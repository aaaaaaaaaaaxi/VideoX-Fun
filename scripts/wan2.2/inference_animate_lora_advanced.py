#!/usr/bin/env python
"""
Advanced Wan2.2 Animate LoRA Inference Script
============================================

This script provides advanced inference for Wan2.2 Animate models with LoRA adapters.
Supports multiple generation modes: T2V, I2V, V2V, and controlled generation.
"""

import argparse
import os
import gc
import logging
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from transformers import T5Tokenizer
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from einops import rearrange

from videox_fun.models import (
    AutoTokenizer,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
    AutoencoderKLWan,
    FaceAdapter,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler, FlowUniPCMultistepScheduler
from videox_fun.pipeline.pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for video generation."""
    mode: str = "text_to_video"  # text_to_video, image_to_video, video_to_video, controlled
    base_model_path: str = ""
    lora_path: str = ""
    output_dir: str = "./outputs"

    # Text inputs
    prompt: str = ""
    negative_prompt: str = ""

    # Video parameters
    height: int = 480
    width: int = 720
    num_frames: int = 49
    fps: int = 8

    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.0
    eta: float = 0.0
    num_videos_per_prompt: int = 1

    # Device and precision
    device: str = "cuda"
    dtype: str = "float16"

    # Scheduler
    scheduler: str = "euler"  # euler, dpm, unipc

    # Advanced settings
    seed: Optional[int] = None
    enable_xformers: bool = True
    enable_teacache: bool = False

    # Control parameters
    control_type: Optional[str] = None  # canny, depth, pose, mlsd
    control_image: Optional[str] = None
    start_image: Optional[str] = None
    reference_image: Optional[str] = None


class Wan2_2AnimateAdvancedInference:
    """Advanced inference class for Wan2.2 Animate with LoRA."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.dtype == "float16" else "no",
            device_placement=True,
        )

        # Initialize components
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.transformer = None
        self.scheduler = None
        self.face_adapter = None

    def load_models(self):
        """Load all required models."""
        logger.info("Loading models...")

        # Load base models
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        self.text_encoder = WanT5EncoderModel.from_pretrained(self.config.base_model_path)
        self.text_encoder = self.text_encoder.to(self.device)

        self.vae = AutoencoderKLWan.from_pretrained(self.config.base_model_path)
        self.vae = self.vae.to(self.device).eval()

        # Load transformer with LoRA
        self.transformer = Wan2_2Transformer3DModel.from_pretrained(self.config.base_model_path)
        if os.path.exists(self.config.lora_path):
            logger.info(f"Loading LoRA from {self.config.lora_path}")
            self.transformer = PeftModel.from_pretrained(self.transformer, self.config.lora_path)
        self.transformer = self.transformer.to(self.device).eval()

        # Load scheduler
        if self.config.scheduler == "dpm":
            self.scheduler = FlowDPMSolverMultistepScheduler.from_pretrained(self.config.base_model_path)
        elif self.config.scheduler == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler.from_pretrained(self.config.base_model_path)
        else:
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.config.base_model_path)

        # Load face adapter if exists
        face_adapter_path = os.path.join(self.config.base_model_path, "face_adapter")
        if os.path.exists(face_adapter_path):
            self.face_adapter = FaceAdapter.from_pretrained(face_adapter_path)
            self.face_adapter = self.face_adapter.to(self.device)

        logger.info("All models loaded successfully")

    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            embeddings = self.text_encoder(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
            )[0]

        return embeddings

    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        image = Image.open(image_path).convert("RGB")

        # Resize to match video dimensions
        image = image.resize((self.config.width, self.config.height))

        # Convert to tensor and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return image_tensor.to(self.device)

    def prepare_latents(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Prepare initial latent noise."""
        latent_channels = self.vae.config.latent_channels
        spatial_compression = self.vae.spatial_compression_ratio
        temporal_compression = self.vae.temporal_compression_ratio

        latent_height = self.config.height // spatial_compression
        latent_width = self.config.width // spatial_compression
        latent_frames = (self.config.num_frames - 1) // temporal_compression + 1

        batch_size = self.config.num_videos_per_prompt

        shape = (
            batch_size,
            latent_channels,
            latent_frames,
            latent_height,
            latent_width,
        )

        latents = torch.randn(shape, generator=generator, device=self.device)
        if hasattr(self.vae, 'config') and hasattr(self.vae.config, 'scaling_factor'):
            latents = latents * self.vae.config.scaling_factor

        return latents

    def load_control_image(self, control_image_path: str) -> torch.Tensor:
        """Load and preprocess control image."""
        control_image = Image.open(control_image_path).convert("RGB")
        control_image = control_image.resize((self.config.width, self.config.height))

        # Encode with VAE
        control_tensor = torch.from_numpy(np.array(control_image)).permute(0, 3, 1, 2).float() / 255.0
        control_tensor = control_tensor.to(self.device)

        with torch.no_grad():
            control_latents = self.vae.encode(control_tensor).sample

        return control_latents

    def generate_text_to_video(self) -> torch.Tensor:
        """Generate video from text prompt."""
        logger.info("Generating video from text prompt...")

        # Encode prompts
        positive_embeds = self.encode_text(self.config.prompt)
        negative_embeds = self.encode_text(self.config.negative_prompt) if self.config.negative_prompt else None

        # Prepare guidance
        if self.config.guidance_scale != 1.0 and negative_embeds is not None:
            embeddings = torch.cat([negative_embeds, positive_embeds], dim=0)
        else:
            embeddings = positive_embeds

        # Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed) if self.config.seed else None
        latents = self.prepare_latents(generator)

        # Set scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)

        # Diffusion loop
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2) if self.config.guidance_scale != 1.0 else latents

            with torch.no_grad():
                noise_pred = self.transformer(
                    x=latent_model_input,
                    context=embeddings,
                    timesteps=t,
                    seq_len=embeddings.shape[1],
                )

            # Apply guidance
            if self.config.guidance_scale != 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = latents / (self.vae.config.scaling_factor if hasattr(self.vae.config, 'scaling_factor') else 1.0)
        video = self.vae.decode(latents).sample

        # Normalize
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0, 1)

        return video

    def generate_image_to_video(self, image_path: str) -> torch.Tensor:
        """Generate video from image prompt."""
        logger.info("Generating video from image prompt...")

        # Load and encode image
        image_tensor = self.load_image(image_path)

        # Encode text prompts
        positive_embeds = self.encode_text(self.config.prompt)
        negative_embeds = self.encode_text(self.config.negative_prompt) if self.config.negative_prompt else None

        # Prepare guidance
        if self.config.guidance_scale != 1.0 and negative_embeds is not None:
            embeddings = torch.cat([negative_embeds, positive_embeds], dim=0)
        else:
            embeddings = positive_embeds

        # Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed) if self.config.seed else None
        latents = self.prepare_latents(generator)

        # Encode image as first frame
        with torch.no_grad():
            image_latents = self.vae.encode(image_tensor.unsqueeze(0)).sample

        # Set scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)

        # Diffusion loop with conditioning
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2) if self.config.guidance_scale != 1.0 else latents

            with torch.no_grad():
                noise_pred = self.transformer(
                    x=latent_model_input,
                    context=embeddings,
                    timesteps=t,
                    seq_len=embeddings.shape[1],
                    # Add image conditioning
                    y=image_latents if i < self.config.num_inference_steps // 2 else None,
                )

            # Apply guidance
            if self.config.guidance_scale != 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = latents / (self.vae.config.scaling_factor if hasattr(self.vae.config, 'scaling_factor') else 1.0)
        video = self.vae.decode(latents).sample

        # Normalize
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0, 1)

        return video

    def generate_controlled_video(self, control_image_path: str) -> torch.Tensor:
        """Generate video with control conditions."""
        logger.info(f"Generating controlled video with {self.config.control_type} control...")

        # Load control image
        control_latents = self.load_control_image(control_image_path)

        # Encode text prompts
        positive_embeds = self.encode_text(self.config.prompt)
        negative_embeds = self.encode_text(self.config.negative_prompt) if self.config.negative_prompt else None

        # Prepare guidance
        if self.config.guidance_scale != 1.0 and negative_embeds is not None:
            embeddings = torch.cat([negative_embeds, positive_embeds], dim=0)
        else:
            embeddings = positive_embeds

        # Prepare latents
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed) if self.config.seed else None
        latents = self.prepare_latents(generator)

        # Set scheduler
        self.scheduler.set_timesteps(self.config.num_inference_steps, device=self.device)

        # Diffusion loop with control
        for i, t in tqdm(enumerate(self.scheduler.timesteps), total=len(self.scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2) if self.config.guidance_scale != 1.0 else latents

            with torch.no_grad():
                noise_pred = self.transformer(
                    x=latent_model_input,
                    context=embeddings,
                    timesteps=t,
                    seq_len=embeddings.shape[1],
                    # Add control conditioning
                    y=control_latents,
                    control_type=self.config.control_type,
                )

            # Apply guidance
            if self.config.guidance_scale != 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = latents / (self.vae.config.scaling_factor if hasattr(self.vae.config, 'scaling_factor') else 1.0)
        video = self.vae.decode(latents).sample

        # Normalize
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0, 1)

        return video

    def save_video(self, video: torch.Tensor, filename: str) -> str:
        """Save generated video to disk."""
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Convert to numpy
        frames = video.cpu().float().numpy()

        # Create filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')[:50]
        output_path = os.path.join(self.config.output_dir, f"{safe_name}.mp4")

        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.config.fps,
                                     (self.config.width, self.config.height))

        for frame in frames:
            frame_np = (frame * 255).astype(np.uint8)
            frame_bgr = frame_np[..., ::-1]
            video_writer.write(frame_bgr)

        video_writer.release()

        return output_path

    def generate(self) -> str:
        """Main generation method."""
        try:
            # Load models
            self.load_models()

            # Generate based on mode
            if self.config.mode == "text_to_video":
                video = self.generate_text_to_video()
            elif self.config.mode == "image_to_video":
                video = self.generate_image_to_video(self.config.start_image)
            elif self.config.mode == "controlled":
                video = self.generate_controlled_video(self.config.control_image)
            else:
                raise ValueError(f"Unsupported generation mode: {self.config.mode}")

            # Save video
            output_path = self.save_video(video, self.config.prompt[:50])

            logger.info(f"Generation completed! Video saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
        finally:
            # Clean up
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.transformer:
            del self.transformer
        if self.vae:
            del self.vae
        if self.text_encoder:
            del self.text_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Advanced Wan2.2 Animate LoRA Inference")

    # Model paths
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to base Wan2.2 model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter")

    # Generation mode
    parser.add_argument("--mode", type=str, default="text_to_video",
                       choices=["text_to_video", "image_to_video", "controlled"],
                       help="Generation mode")

    # Text inputs
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")

    # Control inputs
    parser.add_argument("--control_type", type=str, choices=["canny", "depth", "pose", "mlsd"],
                       help="Type of control for controlled generation")
    parser.add_argument("--control_image", type=str, help="Control image path")
    parser.add_argument("--start_image", type=str, help="Start image path (for I2V)")
    parser.add_argument("--reference_image", type=str, help="Reference image path")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    # Video parameters
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--fps", type=int, default=8, help="Video FPS")

    # Generation settings
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="Eta parameter")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Videos per prompt")

    # Device settings
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"],
                       help="Data type")
    parser.add_argument("--scheduler", type=str, default="euler",
                       choices=["euler", "dpm", "unipc"], help="Scheduler")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # Create config
    config = GenerationConfig(
        mode=args.mode,
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        num_videos_per_prompt=args.num_videos_per_prompt,
        device=args.device,
        dtype=args.dtype,
        scheduler=args.scheduler,
        seed=args.seed,
        control_type=args.control_type,
        control_image=args.control_image,
        start_image=args.start_image,
        reference_image=args.reference_image,
    )

    # Run inference
    generator = Wan2_2AnimateAdvancedInference(config)
    output_path = generator.generate()

    print(f"\n✅ Generation completed!")
    print(f"📁 Video saved to: {output_path}")


if __name__ == "__main__":
    main()