''' StableDiffusion-v1 Predict Module '''
import os
import sys
import logging
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
import numpy as np
from PIL import Image
import time
import functools

# xFuser imports
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

# Diffusers imports
from diffusers import (
    FluxTransformer2DModel,
    FluxFillPipeline,
    FlowMatchEulerDiscreteScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
MODEL_CACHE = "hf-cache"

class Predictor:
    def __init__(self, model_tag="black-forest-labs/FLUX.1-Fill-dev"):
        self.model_tag = model_tag
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.pipe = None
        self.initialized = False

        # Configure logger with rank information
        self.logger = logging.LoggerAdapter(logger, {'rank': self.rank})
        self.logger.info(f"Initializing Predictor on rank {self.rank}")

    def setup(self):
        '''One-time setup per process'''
        if self.initialized:
            return
            
        self.logger.info(f"Loading model on rank {self.rank}")
        torch.cuda.set_device(self.local_rank)

        try:
            # Load transformer model
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_tag,
                torch_dtype=torch.bfloat16,
                subfolder="transformer",
                cache_dir=MODEL_CACHE
            )

            # Initialize pipeline
            self.pipe = FluxFillPipeline.from_pretrained(
                self.model_tag,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                cache_dir=MODEL_CACHE
            ).to(f"cuda:{self.local_rank}")

            # Distributed Data Parallel setup
            if self.world_size > 1:
                self.logger.info("Initializing DDP")
                self.pipe.transformer = DDP(
                    self.pipe.transformer,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank
                )

            self.initialized = True
            self.logger.info(f"Setup complete on rank {self.rank}")

        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            raise

    def predict(self, args):
        '''Main prediction function'''
        if not self.initialized:
            raise RuntimeError("Predictor not initialized - call setup() first")

        try:
            self.logger.info(f"Starting prediction on rank {self.rank}")
            torch.cuda.synchronize()

            # Distributed process synchronization
            if self.world_size > 1:
                dist.barrier()

            # Process inputs
            combined_image, mask_image = self.create_inference_image(
                args.garment,
                args.model_img,
                args.mask,
                size=(args.width, args.height)
                )
            
            # Run inference
            output = self.pipe(
                height=args.height,
                width=args.width * 2,
                image=combined_image,
                mask_image=mask_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                prompt=args.prompt,
                generator=torch.Generator(device=f"cuda:{self.local_rank}").manual_seed(args.seed)
                )
            
            # Process output
            result = self.process_output(output, args)
            
            if self.rank == 0:
                return result
            return None

        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def create_inference_image(self, garment_path, model_path, mask_path, size=(576, 768)):
        '''Create formatted input for FLUX model'''
        try:
            # Load images
            garment_img = Image.open(garment_path).convert("RGB")
            model_img = Image.open(model_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("L")

            # Apply transforms
            transform = transforms.Compose([
                transforms.Resize((size[1], size[0])),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            garment_tensor = transform(garment_img)
            model_tensor = transform(model_img)
            mask_tensor = transforms.ToTensor()(mask_img.resize((size[0], size[1]))) > 0.5

            # Create concatenated inputs
            inpaint_image = torch.cat([garment_tensor, model_tensor], dim=2)
            extended_mask = torch.cat([torch.zeros((1, size[1], size[0])), mask_tensor], dim=2)

            return (
                transforms.ToPILImage()(inpaint_image * 0.5 + 0.5),
                transforms.ToPILImage()(extended_mask)
            )

        except Exception as e:
            self.logger.error(f"Input processing failed: {str(e)}")
            raise

    def process_output(self, output, args):
        '''Process and save output'''
        try:
            result = output.images[0]
            width = args.width
            tryon_result = result.crop((width, 0, width * 2, args.height))
            tryon_result.save(args.output)
            
            if self.rank == 0:
                self.logger.info(f"Saved result to {args.output}")
                print(f"PREDICTION_SUCCESS|{args.output}")  # For handler parsing
            
            return args.output

        except Exception as e:
            self.logger.error(f"Output processing failed: {str(e)}")
            raise

def main():
    # Distributed environment setup
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=int(os.environ['WORLD_SIZE']),
        rank=int(os.environ['RANK'])
    )

    # Parse arguments
    parser = FlexibleArgumentParser(description="xFuser and FLUX Fill Virtual Try-On")
    xFuserArgs.add_cli_args(parser)
    
    # Add custom arguments
    parser.add_argument('--garment', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--model_img', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--width', type=int, default=1224)
    parser.add_argument('--height', type=int, default=1632)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=30.0)
    parser.add_argument('--scheduler', type=str, default='K-LMS')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt', type=str, required=True)

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = Predictor()
        predictor.setup()
        
        # Run prediction
        result = predictor.predict(args)
        
        # Cleanup
        dist.destroy_process_group()
        return result

    except Exception as e:
        logger.error(f"Main process failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()