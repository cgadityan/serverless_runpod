''' StableDiffusion-v1 Predict Module '''

import os
from typing import List, Optional
import torch
from torchvision import transforms
import argparse
from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)
from xfuser.model_executor.layers.attention_processor import xFuserFluxAttnProcessor2_0
from diffusers import FluxTransformer2DModel
from diffusers import (
    FluxTransformer2DModel, 
    FluxFillPipeline,
    # StableDiffusionPipeline,
    # StableDiffusionImg2ImgPipeline,
    # StableDiffusionInpaintPipeline,
    # StableDiffusionInpaintPipelineLegacy,


    DDIMScheduler,
    DDPMScheduler,
    # DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    # KarrasVeScheduler,
    PNDMScheduler,
    # RePaintScheduler,
    # ScoreSdeVeScheduler,
    # ScoreSdeVpScheduler,
    # UnCLIPScheduler,
    # VQDiffusionScheduler,
    LMSDiscreteScheduler
)
import time 
import functools
import numpy as np
from PIL import Image
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

MODEL_CACHE = "hf-cache"


class Predictor:
    '''Predictor class for StableDiffusion-v1'''

    def __init__(self, model_tag="runwayml/stable-diffusion-v1-5"):
        '''
        Initialize the Predictor class
        '''
        self.model_tag = model_tag

    def setup(self):
        '''
        Load the model into memory to make running multiple predictions efficient
        '''
        print("Loading pipeline...")

        print("Loading FLUX Fill model...")
        # Initialize the pipeline with the correct model
        self.transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev", 
            torch_dtype=torch.bfloat16,
            subfolder="transformer",
            cache_dir= MODEL_CACHE
        )
        self.pipe = FluxFillPipeline.from_pretrained(
            # pretrained_model_name_or_path=engine_config.model_config.model,
            "black-forest-labs/FLUX.1-Fill-dev",
            transformer=self.transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE
        ).to("cuda")

        # self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
        #     self.model_tag,
        #     safety_checker=None,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        #     torch_dtype=torch.float16,
        # ).to("cuda")
        # self.img2img_pipe = StableDiffusionImg2ImgPipeline(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        #     safety_checker=None,
        #     # safety_checker=self.txt2img_pipe.safety_checker,
        #     feature_extractor=self.txt2img_pipe.feature_extractor,
        # ).to("cuda")
        # self.inpaint_pipe = StableDiffusionInpaintPipelineLegacy(
        #     vae=self.txt2img_pipe.vae,
        #     text_encoder=self.txt2img_pipe.text_encoder,
        #     tokenizer=self.txt2img_pipe.tokenizer,
        #     unet=self.txt2img_pipe.unet,
        #     scheduler=self.txt2img_pipe.scheduler,
        #     safety_checker=None,
        #     # safety_checker=self.txt2img_pipe.safety_checker,
        #     feature_extractor=self.txt2img_pipe.feature_extractor,
        # ).to("cuda")

        # # because lora is loaded for the entire model
        # self.lora_loaded = False
        # self.txt2img_pipe.unet.to(memory_format=torch.channels_last)
        # self.img2img_pipe.unet.to(memory_format=torch.channels_last)
        # self.inpaint_pipe.unet.to(memory_format=torch.channels_last)
        # self.txt2img_pipe.enable_xformers_memory_efficient_attention()
        # self.img2img_pipe.enable_xformers_memory_efficient_attention()
        # self.inpaint_pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, init_image, mask, prompt_strength, num_outputs, num_inference_steps, guidance_scale, scheduler, seed, lora, lora_scale):
        '''
        Run a single prediction on the model
        '''
        # if seed is None:
        #     seed = int.from_bytes(os.urandom(2), "big")

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits."
        #     )

        extra_kwargs = {}
 
        pipe = self.pipe
        pipe.scheduler = make_scheduler(scheduler, pipe.scheduler.config)

        # Create a unified parser with a combined description
        parser = FlexibleArgumentParser(description="xFuser and FLUX Fill Virtual Try-On Arguments")
        
        # Add xFuser-related arguments
        xFuserArgs.add_cli_args(parser)
        
        # Add FLUX Fill Virtual Try-On specific arguments
        parser.add_argument("--model_img", default="/data/model.jpg", help="Path to model image")
        parser.add_argument("--garment", default="/data/garment.jpg", help="Path to garment image")
        parser.add_argument("--mask", default="/data/mask.png", help="Path to mask image")
        parser.add_argument("--output", default="/data/output.jpg", help="Path to save output image")
        # parser.add_argument("--prompt", default="A photo of a person wearing the garment, detailed texture, high quality",
        #                     help="Text prompt for generation")
        parser.add_argument("--size", default="1224,1632", help="Output size as width,height")
        # parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
        parser.add_argument("--cache_dir", default="/hf_cache", help="Cache directory for models")
        
        # Parse all arguments at once
        args = parser.parse_args()
        
        # Process xFuser arguments to create configurations
        engine_args = xFuserArgs.from_cli_args(args)
        engine_config, input_config = engine_args.create_config()
        engine_config.runtime_config.dtype = torch.bfloat16
        local_rank = get_world_group().local_rank

        input_config.num_inference_steps = 50
        engine_args.ulysses_degree = 2
        engine_args.ring_degree = 1

        parallel_info = (
            f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
            f"tp{engine_args.tensor_parallel_degree}_"
            f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
        )

        parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

        try:
            width, height = map(int, args.size.split(','))
            size = (width, height)
        except:
            print("Invalid size format. Using default 1224x1632.")
            # size = (720, 960)
            size = (1224, 1632)
        
        # Use default prompt if None was provided (this is redundant with the default parameter but kept for clarity)
        prompt = args.prompt if args.prompt else """Two-panel image showing a garment on the left and a model wearing the same garment on the right.
[IMAGE1] White Adidas t-shirt with black trefoil logo and text.
[IMAGE2] Model wearing a White Adidas t-shirt with black trefoil logo and text."""

        print("size: ", size)

        ### Parallelize 
        initialize_runtime_state(pipe, engine_config)
        get_runtime_state().set_input_parameters(
            height=size[1],
            width=size[0] * 2,
            batch_size=1,
            num_inference_steps=input_config.num_inference_steps,
            max_condition_sequence_length=512,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
        
        parallelize_transformer(pipe)

        print("Processing virtual try-on...")
        output_paths, peak_memory, elapsed_time = process_virtual_try_on(pipe, engine_args, engine_config, input_config,
            args.garment,
            args.model_img,
            args.mask,
            args.output,
            local_rank,
            prompt=prompt,
            size=size,
            seed=input_config.seed    
        )
        
        if get_world_group().rank == get_world_group().world_size - 1:
            print(
                f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9:.2f} GB"
            )
        get_runtime_state().destory_distributed_env()

        # if not (lora is None):
        #     print("loaded lora")
        #     pipe.unet.load_attn_procs(lora)
        #     self.lora_loaded = True
        #     extra_kwargs['cross_attention_kwargs'] = {"scale": lora_scale}

        # # because lora is retained between requests
        # if (lora is None) and self.lora_loaded:
        #     extra_kwargs['cross_attention_kwargs'] = {"scale": 0}

        # generator = torch.Generator("cuda").manual_seed(seed)
        # output = pipe(
        #     prompt=[prompt] * num_outputs if prompt is not None else None,
        #     negative_prompt=[negative_prompt]*num_outputs if negative_prompt is not None else None,
        #     # width=width,
        #     # height=height,
        #     guidance_scale=guidance_scale,
        #     generator=generator,
        #     num_inference_steps=num_inference_steps,
        #     **extra_kwargs,
        # )

        # output_paths = []
        # for i, sample in enumerate(output.images):
        #     if output.nsfw_content_detected and output.nsfw_content_detected[i] and self.NSFW:
        #         continue

        #     output_path = f"/outputs/out-{i}.png"
        #     sample.save(output_path)
        #     output_paths.append(output_path)

        # if len(output_paths) == 0:
        #     raise Exception(
        #         "NSFW content detected. Try running it again, or try a different prompt."
        #     )

        return output_paths
    

    
##################################################################################

def fit_in_box(img, target_width, target_height, fill_color=(255, 255, 255)):
    """Resize image to fit in target box while preserving aspect ratio"""
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        return Image.new("RGB", (target_width, target_height), fill_color)
    
    scale = min(target_width / float(orig_w), target_height / float(orig_h))
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (target_width, target_height), fill_color)
    offset_x = (target_width - new_w) // 2
    offset_y = (target_height - new_h) // 2
    new_img.paste(resized, (offset_x, offset_y))
    
    return new_img

##################################################################################

def create_inference_image(garment_path, model_path, mask_path, size=(576, 768)):
    """Creates properly formatted input for FLUX Fill model"""
    # Ensure size is a valid tuple with positive dimensions
    if isinstance(size, (int, float)):
        size = (int(size), int(size))
    elif not isinstance(size, tuple) or len(size) != 2:
        size = (576, 768)  # default size
    
    width, height = int(size[0]), int(size[1])
    
    # Add transform
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor()
    ])
    
    try:
        # Load and verify images
        garment_img = Image.open(garment_path).convert("RGB")
        model_img = Image.open(model_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Apply transforms
        garment_tensor = transform(garment_img)
        model_tensor = transform(model_img)
        mask_tensor = mask_transform(mask_img)
        
        # Ensure mask is binary
        mask_tensor = (mask_tensor > 0.5).float()
        
        # Create concatenated image (garment | model)
        inpaint_image = torch.cat([garment_tensor, model_tensor], dim=2)
        
        # Create mask (zeros for garment | mask for model)
        garment_mask = torch.zeros((1, height, width))
        extended_mask = torch.cat([garment_mask, mask_tensor], dim=2)
        
        # Convert tensors back to PIL images
        inpaint_image_pil = transforms.ToPILImage()(inpaint_image * 0.5 + 0.5)
        mask_image_pil = transforms.ToPILImage()(extended_mask)
        
        return inpaint_image_pil, mask_image_pil
        
    except Exception as e:
        raise Exception(f"Error in create_inference_image: {str(e)}")

##################################################################################
    
def parallelize_transformer(pipe: FluxFillPipeline):
    transformer = pipe.transformer
    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):
        assert hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0, \
            f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True
        
        if isinstance(timestep, torch.Tensor) and timestep.ndim != 0 and timestep.shape[0] == hidden_states.shape[0]:
            timestep = torch.chunk(timestep, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(encoder_hidden_states, get_classifier_free_guidance_world_size(),dim=0)[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(encoder_hidden_states, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(),dim=-2)[get_sequence_parallel_rank()]
        
        for block in transformer.transformer_blocks + transformer.single_transformer_blocks:
            block.attn.processor = xFuserFluxAttnProcessor2_0()
        
        output = original_forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])

    new_forward = new_forward.__get__(transformer)
    transformer.forward = new_forward

##################################################################################

def process_virtual_try_on(pipe, engine_args, engine_config, input_config, garment_path, model_path, mask_path, output_path, local_rank, prompt=None, size=(576, 768), seed=42):
    """Process virtual try-on using FLUX Fill model"""
    try:
        # Ensure size is a valid tuple
        if isinstance(size, (int, float)):
            size = (int(size), int(size))
        elif not isinstance(size, tuple) or len(size) != 2:
            size = (576, 768)  # default size
        
        # Create inference images
        combined_image, mask_image = create_inference_image(
            garment_path, 
            model_path, 
            mask_path, 
            size=size
        )
        
        # Use default prompt if none provided
        if prompt is None:
            prompt = "A photo of a person wearing the garment, detailed texture, high quality"
        
        # Run inference
        # generator = torch.Generator(device="cuda").manual_seed(seed)
        

        parallel_info = (
            f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
            f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
            f"tp{engine_args.tensor_parallel_degree}_"
            f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
        )

        print(parallel_info)

        if engine_config.runtime_config.use_torch_compile:
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    
            # one step to warmup the torch compiler
            output = pipe(
                height=size[1],
                width=size[0] * 2,  # Double width for side-by-side images
                image=combined_image,
                mask_image=mask_image,
                num_inference_steps=1,
                max_sequence_length=512,
                guidance_scale=30,
                prompt=prompt,
                output_type=input_config.output_type,
                generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
            ).images
    
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        output = pipe(
            height=size[1],
            width=size[0] * 2,  # Double width for side-by-side images
            image=combined_image,
            mask_image=mask_image,
            num_inference_steps=input_config.num_inference_steps,
            max_sequence_length=512,
            guidance_scale=30,
            prompt=prompt,
            output_type=input_config.output_type,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed)
        )

        result = output.images[0]
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

        
        
        # Extract the right half (try-on result)
        width = size[0]
        tryon_result = result.crop((width, 0, width * 2, size[1]))
        
        # Save the try-on result
        tryon_result = tryon_result.convert('RGB')  # Convert to RGB to ensure compatibility
        tryon_result.save(output_path, format='PNG')
        
        # Create a comparison panel (optional)
        garment_img = Image.open(garment_path).convert("RGB")
        model_img = Image.open(model_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")
        
        # Create blacked-out model
        model_array = np.array(model_img)
        mask_array = np.array(mask_img)
        mask_3d = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2)
        model_array[mask_3d > 0] = 0
        model_with_mask = Image.fromarray(model_array)
        
        # Fit images to panel size
        garment_fitted = fit_in_box(garment_img, size[0], size[1])
        model_fitted = fit_in_box(model_with_mask, size[0], size[1])
        result_fitted = fit_in_box(tryon_result, size[0], size[1])
        
        # Create panel
        panel_width = size[0] * 3  # garment + model + result
        panel_height = size[1]
        panel = Image.new("RGB", (panel_width, panel_height), (255, 255, 255))
        
        # Paste images
        panel.paste(garment_fitted, (0, 0))
        panel.paste(model_fitted, (size[0], 0))
        panel.paste(result_fitted, (size[0] * 2, 0))

        if input_config.output_type == "pil":
            dp_group_index = get_data_parallel_rank()
            num_dp_groups = get_data_parallel_world_size()
            dp_batch_size = (input_config.batch_size + num_dp_groups - 1) // num_dp_groups
            if is_dp_last_group():
                for i, image in enumerate(output.images):
                    image_rank = dp_group_index * dp_batch_size + i
                    image_name = f"flux_result_{parallel_info}_{image_rank}_tc_{engine_args.use_torch_compile}.png"
                    print(image_name)
        
        # Save panel
        panel_path = os.path.splitext(output_path)[0] + "_panel.png"
        panel.save(panel_path, format='PNG')
        
        print(f"Try-on result saved to: {output_path}")
        print(f"Comparison panel saved to: {panel_path}")
        
        return output_path, peak_memory, elapsed_time
        
    except Exception as e:
        raise Exception(f"Error in virtual try-on processing: {str(e)}")


def make_scheduler(name, config):
    '''
    Returns a scheduler from a name and config.
    '''
    return {
        "DDIM": DDIMScheduler.from_config(config),
        "DDPM": DDPMScheduler.from_config(config),
        # "DEIS": DEISMultistepScheduler.from_config(config),
        "DPM-M": DPMSolverMultistepScheduler.from_config(config),
        "DPM-S": DPMSolverSinglestepScheduler.from_config(config),
        "EULER-A": EulerAncestralDiscreteScheduler.from_config(config),
        "EULER-D": EulerDiscreteScheduler.from_config(config),
        "HEUN": HeunDiscreteScheduler.from_config(config),
        "IPNDM": IPNDMScheduler.from_config(config),
        "KDPM2-A": KDPM2AncestralDiscreteScheduler.from_config(config),
        "KDPM2-D": KDPM2DiscreteScheduler.from_config(config),
        # "KARRAS-VE": KarrasVeScheduler.from_config(config),
        "PNDM": PNDMScheduler.from_config(config),
        # "RE-PAINT": RePaintScheduler.from_config(config),
        # "SCORE-VE": ScoreSdeVeScheduler.from_config(config),
        # "SCORE-VP": ScoreSdeVpScheduler.from_config(config),
        # "UN-CLIPS": UnCLIPScheduler.from_config(config),
        # "VQD": VQDiffusionScheduler.from_config(config),
        "K-LMS": LMSDiscreteScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config)
    }[name]
