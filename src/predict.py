''' StableDiffusion-v1 Predict Module '''
import os
import logging
import torch
import torch.distributed as dist
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import functools

# xFuser imports
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group, initialize_runtime_state,
    get_runtime_state, get_sequence_parallel_world_size,
    get_sequence_parallel_rank, get_sp_group,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank, get_cfg_group
)
from xfuser.model_executor.layers.attention_processor import xFuserFluxAttnProcessor2_0

# Diffusers imports
from diffusers import FluxTransformer2DModel, FluxFillPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, DDIMScheduler

# Configure logging
logging.basicConfig(
    format='%(asctime)s - [RANK %(rank)s] - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
MODEL_CACHE = "hf-cache"

def fit_in_box(img, target_width, target_height, fill_color=(255, 255, 255)):
    """Resize image to fit in target box while preserving aspect ratio"""
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        logger.warning("Input image has zero width or height")
        return Image.new("RGB", (target_width, target_height), fill_color)
    
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (target_width, target_height), fill_color)
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    new_img.paste(resized, (offset_x, offset_y))
    return new_img

def create_inference_image(garment_path, model_path, mask_path, size=(576, 768)):
    """Creates properly formatted input for FLUX Fill model"""
    try:
        width, height = size
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        mask_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])

        # Load and process images
        garment_img = Image.open(garment_path).convert("RGB")
        model_img = Image.open(model_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        # Apply transforms
        garment_tensor = transform(garment_img)
        model_tensor = transform(model_img)
        mask_tensor = (mask_transform(mask_img) > 0.5).float()

        # Create concatenated inputs
        inpaint_image = torch.cat([garment_tensor, model_tensor], dim=2)
        extended_mask = torch.cat([torch.zeros((1, height, width)), mask_tensor], dim=2)

        return (
            transforms.ToPILImage()(inpaint_image * 0.5 + 0.5),
            transforms.ToPILImage()(extended_mask)
        )
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise

class ParallelRunner:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.pipe = None
        self.original_transformer = None

    def setup_distributed(self):
        """Initialize distributed training environment"""
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank
        )
        torch.cuda.set_device(self.local_rank)
        logger.info(f"Initialized process group (rank {self.rank}/{self.world_size})")

    def load_models(self, model_tag="black-forest-labs/FLUX.1-Fill-dev"):
        """Load models with proper device placement"""
        try:
            self.original_transformer = FluxTransformer2DModel.from_pretrained(
                model_tag,
                torch_dtype=torch.bfloat16,
                subfolder="transformer",
                cache_dir=MODEL_CACHE
            ).to(f"cuda:{self.local_rank}")

            self.pipe = FluxFillPipeline.from_pretrained(
                model_tag,
                transformer=self.original_transformer,
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE
            ).to(f"cuda:{self.local_rank}")

            logger.info(f"Models loaded on rank {self.rank}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def configure_parallelization(self, engine_args):
        """Apply xFuser parallelization strategies"""
        try:
            engine_config, _ = engine_args.create_config()
            initialize_runtime_state(self.pipe, engine_config)
            
            if engine_args.ulysses_degree > 1:
                logger.info(f"Applying Ulysses parallelism (degree {engine_args.ulysses_degree})")
                get_runtime_state().set_parallel_config(
                    ulysses_degree=engine_args.ulysses_degree,
                    ring_degree=engine_args.ring_degree,
                    tensor_parallel_degree=engine_args.tensor_parallel_degree
                )

            self.parallelize_transformer()
            logger.info(f"Parallelization configured on rank {self.rank}")
        except Exception as e:
            logger.error(f"Parallelization failed: {str(e)}")
            raise

    def parallelize_transformer(self):
        """Apply custom transformer parallelization"""
        transformer = self.pipe.transformer
        original_forward = transformer.forward

        @functools.wraps(original_forward)
        def new_forward(hidden_states, encoder_hidden_states=None, *args, **kwargs):
            if hidden_states.shape[-2] % get_sequence_parallel_world_size() == 0:
                hidden_states = torch.chunk(
                    hidden_states,
                    get_sequence_parallel_world_size(),
                    dim=-2
                )[get_sequence_parallel_rank()]
                
            for block in transformer.transformer_blocks:
                block.attn.processor = xFuserFluxAttnProcessor2_0()

            output = original_forward(hidden_states, encoder_hidden_states, *args, **kwargs)
            output = list(output)
            output[0] = get_sp_group().all_gather(output[0], dim=-2)
            return tuple(output)

        transformer.forward = new_forward
        logger.info(f"Transformer parallelized on rank {self.rank}")

    def run_inference(self, args):
        """Main inference workflow"""
        try:
            inpaint_image, mask_image = create_inference_image(
                args.garment, args.model_img, args.mask, (args.width, args.height)
            )
            
            output = self.pipe(
                height=args.height,
                width=args.width * 2,
                image=inpaint_image,
                mask_image=mask_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                prompt=args.prompt,
                generator=torch.Generator(device=f"cuda:{self.local_rank}").manual_seed(args.seed)
            )

            return self.save_output(output, args.output)
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise

    def save_output(self, output, path):
        """Process and save output"""
        try:
            result = output.images[0].crop((args.width, 0, args.width * 2, args.height))
            result.save(path)
            logger.info(f"Saved output to {path} on rank {self.rank}")
            
            # Create comparison panel
            panel = self.create_comparison_panel(output.images[0], args)
            panel_path = os.path.splitext(path)[0] + "_panel.png"
            panel.save(panel_path)
            
            return path
        except Exception as e:
            logger.error(f"Output saving failed: {str(e)}")
            raise

    def create_comparison_panel(self, result, args):
        """Create comparison panel for visualization"""
        garment_img = Image.open(args.garment).convert("RGB")
        model_img = Image.open(args.model_img).convert("RGB")
        mask_img = Image.open(args.mask).convert("L")

        # Create blacked-out model
        model_array = np.array(model_img)
        mask_array = np.array(mask_img)
        model_array[np.repeat(mask_array[:, :, np.newaxis], 3, axis=2) > 0] = 0
        model_with_mask = Image.fromarray(model_array)

        # Create panel
        panel = Image.new("RGB", (args.width * 3, args.height), (255, 255, 255))
        panel.paste(fit_in_box(garment_img, args.width, args.height), (0, 0))
        panel.paste(fit_in_box(model_with_mask, args.width, args.height), (args.width, 0))
        panel.paste(fit_in_box(result.crop((args.width, 0, args.width*2, args.height)), 
                             args.width, args.height), (args.width*2, 0))
        return panel

def main():
    runner = ParallelRunner()
    runner.setup_distributed()

    # Argument parsing
    parser = FlexibleArgumentParser(description="xFuser Parallel Inference")
    xFuserArgs.add_cli_args(parser)
    parser.add_argument('--garment', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--model_img', required=True)
    parser.add_argument('--output', required=True)
    # parser.add_argument('--width', type=int, default=1224)
    # parser.add_argument('--height', type=int, default=1632)
    # parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=30.0)
    parser.add_argument('--scheduler', default='K-LMS')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--prompt', required=True)
    
    args = parser.parse_args()

    try:
        runner.load_models()
        runner.configure_parallelization(xFuserArgs(args))
        output_path = runner.run_inference(args)
        
        if runner.rank == 0:
            print(output_path)
            
    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()