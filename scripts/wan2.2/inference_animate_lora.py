#!/usr/bin/env python
"""
Wan2.2 Animate LoRA Inference Script
=====================================

This script provides inference for Wan2.2 Animate models with LoRA adapters.
Supports text-to-video (T2V), image-to-video (I2V), and controlled generation.
"""

import argparse
import os
import gc
import logging
from typing import Optional, List, Tuple, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from transformers import T5Tokenizer
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm

from videox_fun.models import (
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
    AutoencoderKLWan,
    FaceAdapter,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler, FlowUniPCMultistepScheduler

logger = logging.getLogger(__name__)


class Wan2_2AnimateLoRAInferenceOutput(BaseOutput):
    """
    Output class for Wan2.2 Animate LoRA inference.
    """

    def __init__(
        self,
        videos: torch.Tensor,
        frames: List[np.ndarray],
        save_path: Optional[str] = None,
    ):
        self.videos = videos
        self.frames = frames
        self.save_path = save_path


def load_lora_transformer(base_model_path: str, lora_path: str, device: torch.device):
    """
    Load base Wan2.2 transformer with LoRA adapter.

    Args:
        base_model_path: Path to base Wan2.2 model
        lora_path: Path to LoRA adapter weights
        device: Target device

    Returns:
        Loaded transformer with LoRA applied
    """
    # Load base model
    transformer = Wan2_2Transformer3DModel.from_pretrained(base_model_path)
    transformer = transformer.to(device)

    # Apply LoRA adapter
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA adapter from {lora_path}")
        transformer = PeftModel.from_pretrained(transformer, lora_path)
        transformer = transformer.to(device)
        transformer = transformer.eval()
    else:
        logger.warning(f"LoRA adapter not found at {lora_path}, using base model only")

    return transformer


def load_pipeline_components(
    base_model_path: str,
    lora_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
):
    """
    Load all pipeline components.

    Args:
        base_model_path: Path to base model
        lora_path: Path to LoRA adapter
        device: Target device
        dtype: Data type

    Returns:
        Tuple of loaded components
    """
    logger.info("Loading pipeline components...")

    # Load tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    text_encoder = WanT5EncoderModel.from_pretrained(base_model_path)
    text_encoder = text_encoder.to(device)

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(base_model_path)
    vae = vae.to(device).eval()

    # Load transformer with LoRA
    transformer = load_lora_transformer(base_model_path, lora_path, device)

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_path)

    # Create face adapter if needed
    face_adapter = None
    if os.path.exists(os.path.join(base_model_path, "face_adapter")):
        face_adapter = FaceAdapter.from_pretrained(os.path.join(base_model_path, "face_adapter"))
        face_adapter = face_adapter.to(device)

    logger.info("All components loaded successfully")

    return tokenizer, text_encoder, vae, transformer, scheduler, face_adapter


def encode_prompt(
    prompt: Union[str, List[str]],
    text_encoder,
    tokenizer,
    device: torch.device,
    max_length: int = 512,
    negative_prompt: Optional[str] = None,
):
    """
    Encode text prompts into embeddings.

    Args:
        prompt: Input prompt(s)
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        device: Target device
        max_length: Maximum sequence length
        negative_prompt: Negative prompt for guidance

    Returns:
        Tuple of positive and negative embeddings
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    # Tokenize prompts
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Encode positive prompts
    with torch.no_grad():
        prompt_embeds = text_encoder(
            text_input_ids=text_inputs.input_ids.to(device),
            attention_mask=text_inputs.attention_mask.to(device),
        )[0]

    # Encode negative prompt if provided
    if negative_prompt is not None:
        negative_inputs = tokenizer(
            [negative_prompt],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            negative_embeds = text_encoder(
                text_input_ids=negative_inputs.input_ids.to(device),
                attention_mask=negative_inputs.attention_mask.to(device),
            )[0]
    else:
        negative_embeds = None

    return prompt_embeds, negative_embeds


def prepare_initial_latents(
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    vae: AutoencoderKLWan,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
):
    """
    Prepare initial latent noise for video generation.

    Args:
        batch_size: Batch size
        num_frames: Number of video frames
        height: Video height
        width: Video width
        vae: VAE model
        device: Target device
        generator: Random generator

    Returns:
        Initial latent noise
    """
    # Calculate latent dimensions
    latent_channels = vae.config.latent_channels
    spatial_compression = vae.spatial_compression_ratio
    temporal_compression = vae.temporal_compression_ratio

    latent_height = height // spatial_compression
    latent_width = width // spatial_compression
    latent_frames = (num_frames - 1) // temporal_compression + 1

    shape = (
        batch_size,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
    )

    # Generate initial noise
    latents = torch.randn(shape, generator=generator, device=device)

    # Scale by VAE scaling factor
    if hasattr(vae, 'config') and hasattr(vae.config, 'scaling_factor'):
        latents = latents * vae.config.scaling_factor

    return latents


def animate_inference(
    prompt: Union[str, List[str]],
    base_model_path: str,
    lora_path: str,
    output_dir: str = "./outputs",
    negative_prompt: Optional[str] = None,
    height: int = 480,
    width: int = 720,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    eta: float = 0.0,
    num_videos_per_prompt: int = 1,
    seed: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    scheduler_type: str = "euler",
    face_adapter_path: Optional[str] = None,
):
    """
    Main inference function for Wan2.2 Animate with LoRA.

    Args:
        prompt: Text prompt(s) for generation
        base_model_path: Path to base Wan2.2 model
        lora_path: Path to LoRA adapter weights
        output_dir: Output directory for saved videos
        negative_prompt: Negative prompt for guidance
        height: Video height
        width: Video width
        num_frames: Number of video frames
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        eta: Eta parameter for scheduler
        num_videos_per_prompt: Number of videos per prompt
        seed: Random seed for reproducibility
        device: Target device
        dtype: Data type
        scheduler_type: Type of scheduler to use
        face_adapter_path: Path to face adapter

    Returns:
        Inference output object
    """
    # Set up device and generator
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if dtype == torch.float16 else "no",
        device_placement=True,
    )

    try:
        # Load pipeline components
        tokenizer, text_encoder, vae, transformer, scheduler, face_adapter = load_pipeline_components(
            base_model_path, lora_path, device, dtype
        )

        # Set scheduler
        if scheduler_type.lower() == "dpm":
            scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(base_model_path)
        elif scheduler_type.lower() == "unipc":
            scheduler = FlowUniPCMultistepScheduler.from_pretrained(base_model_path)

        # Encode prompts
        prompt_embeds, negative_embeds = encode_prompt(
            prompt, text_encoder, tokenizer, device, negative_prompt=negative_prompt
        )

        # Prepare latents
        batch_size = len(prompt) * num_videos_per_prompt
        latents = prepare_initial_latents(
            batch_size, num_frames, height, width, vae, device, generator
        )

        # Set timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)

        # Prepare guidance
        if guidance_scale != 1.0:
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds], dim=0)

        logger.info("Starting diffusion process...")

        # Diffusion loop
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps)):
            # Expand latents for guidance
            latent_model_input = torch.cat([latents] * 2) if guidance_scale != 1.0 else latents

            # Predict noise
            with torch.no_grad():
                noise_pred = transformer(
                    x=latent_model_input,
                    context=prompt_embeds,
                    timesteps=t,
                    seq_len=prompt_embeds.shape[1],
                )

            # Apply guidance
            if guidance_scale != 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Compute previous latents
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        logger.info("Decoding latents...")
        latents = latents / vae.config.scaling_factor
        video = vae.decode(latents).sample

        # Normalize to [0, 1]
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0, 1)

        # Convert to numpy
        frames = video.cpu().float().numpy()

        # Save video
        save_path = save_video(frames, output_dir, prompt[0] if isinstance(prompt, str) else "generated")

        # Create output
        output = Wan2_2AnimateLoRAInferenceOutput(
            videos=video,
            frames=frames,
            save_path=save_path,
        )

        logger.info(f"Generation completed! Video saved to: {save_path}")
        return output

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise
    finally:
        # Clean up
        if 'transformer' in locals():
            del transformer
        if 'vae' in locals():
            del vae
        if 'text_encoder' in locals():
            del text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_video(frames: np.ndarray, output_dir: str, prompt_name: str) -> str:
    """
    Save generated video to disk.

    Args:
        frames: Video frames array
        output_dir: Output directory
        prompt_name: Name derived from prompt

    Returns:
        Path to saved video
    """
    import cv2

    # Create safe filename
    safe_name = "".join(c for c in prompt_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_name = safe_name.replace(' ', '_')[:50]  # Limit length

    output_path = os.path.join(output_dir, f"{safe_name}.mp4")

    # Convert frames to video
    height, width = frames.shape[-3], frames.shape[-2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))

    for frame in frames:
        # Normalize and convert to uint8
        frame_np = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        frame_bgr = frame_np[..., ::-1]
        video_writer.write(frame_bgr)

    video_writer.release()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Animate LoRA Inference")

    # Model paths
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base Wan2.2 model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter weights")

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    # Video parameters
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of video frames")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")

    # Generation settings
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Classifier-free guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta parameter")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Videos per prompt")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Data type")
    parser.add_argument("--scheduler", type=str, default="euler", choices=["euler", "dpm", "unipc"], help="Scheduler type")

    args = parser.parse_args()

    # Convert dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Run inference
    output = animate_inference(
        prompt=args.prompt,
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        num_videos_per_prompt=args.num_videos_per_prompt,
        seed=args.seed,
        device=args.device,
        dtype=dtype,
        scheduler_type=args.scheduler,
    )

    print(f"Generation completed! Video saved to: {output.save_path}")


if __name__ == "__main__":
    main()