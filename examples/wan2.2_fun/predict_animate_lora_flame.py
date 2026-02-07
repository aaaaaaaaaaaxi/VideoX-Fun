"""
Inference script for Wan2.2 Animate LoRA model with FLAME parameters.
This script loads a trained transformer checkpoint and LoRA weights (without using PEFT)
and performs video generation using FLAME parameters for facial animation.

Based on training script: scripts/wan2.2/train_animate_lora.py
Reference: e:/HKUSTGZ/HoloSoul/InstanceAnimator/inference/predict_video_decouple.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices
from videox_fun.models import (
    AutoencoderKLWan, AutoencoderKLWan3_8, AutoTokenizer,
    CLIPModel, Wan2_2Transformer3DModel_Animate, WanT5EncoderModel
)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2AnimatePipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_image, get_video_to_video_latent, save_videos_grid


# ==============================================================================
# Configuration
# ==============================================================================

# GPU memory mode options:
# - "model_full_load": entire model on GPU
# - "model_full_load_and_qfloat8": GPU + float8 quantization
# - "model_cpu_offload": CPU offload after use
# - "model_cpu_offload_and_qfloat8": CPU offload + float8
# - "sequential_cpu_offload": layer-wise offload (slow but memory efficient)
GPU_memory_mode = "sequential_cpu_offload"

# Multi-GPU config (product must equal number of GPUs)
ulysses_degree = 1
ring_degree = 1

# FSDP config
fsdp_dit = False
fsdp_text_encoder = True

# Compile for speedup (incompatible with fsdp_dit and sequential_cpu_offload)
compile_dit = False

# TeaCache config
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False

# CFG skip ratio for acceleration (0.00 - 0.25)
cfg_skip_ratio = 0

# Riflex config
enable_riflex = False
riflex_k = 6

# Config and model paths
config_path = "config/wan2.2/wan_civitai_animate.yaml"
model_name = "./models/Diffusion_Transformer/Wan2.2-Animate-14B/"

# Sampler selection: "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name = "Flow_Unipc"
shift = 5

# Model checkpoint paths (from training output)
transformer_path = None  # Path to transformer.safetensors
transformer_high_path = None  # Path to high noise transformer.safetensors (optional)
vae_path = None  # Path to VAE checkpoint (optional)

# LoRA checkpoint paths (from training output)
lora_path = None  # Path to checkpoint-{step}.safetensors
lora_high_path = None  # Path to high noise LoRA (optional)
lora_weight = 0.55
lora_high_weight = 0.55

# Input paths
src_pose_path = "asset/wan_animate/replace/process_results/src_pose.mp4"
src_ref_path = "asset/wan_animate/replace/process_results/src_ref.png"
src_bg_path = "asset/wan_animate/replace/process_results/src_bg.mp4"  # Optional
src_mask_path = "asset/wan_animate/replace/process_results/src_mask.mp4"  # Optional

# FLAME parameters path (npy file with shape (T, 56))
# 56 = 50 (expression) + 3 (jaw_pose) + 3 (global_pose)
flame_params_path = "flame_params.npy"

# Generation parameters
sample_size = [480, 832]
video_length = 81
fps = 16
weight_dtype = torch.bfloat16

# Text prompts
prompt = "视频中的人在做动作"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

guidance_scale = 4.0
seed = 43
num_inference_steps = 20
save_path = "samples/wan-videos-animate-flame"


# ==============================================================================
# Model Loading
# ==============================================================================

def load_checkpoint(checkpoint_path):
    """Load checkpoint from safetensors or pytorch file."""
    if checkpoint_path is None:
        return None

    print(f"Loading checkpoint from: {checkpoint_path}")
    if checkpoint_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")

    state_dict = state_dict.get("state_dict", state_dict)
    return state_dict


def load_models():
    """Load all models for inference."""
    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)
    boundary = config['transformer_additional_kwargs'].get('boundary', 0.875)

    print("Loading transformer...")
    transformer = Wan2_2Transformer3DModel_Animate.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get(
            'transformer_low_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Load second transformer for high noise if using MoE
    if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get(
                'transformer_high_noise_model_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None

    # Load transformer checkpoint
    state_dict = load_checkpoint(transformer_path)
    if state_dict is not None:
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"Transformer - missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Load high noise transformer checkpoint
    if transformer_2 is not None:
        state_dict_high = load_checkpoint(transformer_high_path)
        if state_dict_high is not None:
            m, u = transformer_2.load_state_dict(state_dict_high, strict=False)
            print(f"Transformer_2 - missing keys: {len(m)}, unexpected keys: {len(u)}")

    print("Loading VAE...")
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Load VAE checkpoint
    vae_state_dict = load_checkpoint(vae_path)
    if vae_state_dict is not None:
        m, u = vae.load_state_dict(vae_state_dict, strict=False)
        print(f"VAE - missing keys: {len(m)}, unexpected keys: {len(u)}")

    print("Loading text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get(
            'tokenizer_subpath', 'tokenizer'))
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get(
            'text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    print("Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get(
            'image_encoder_subpath', 'image_encoder'))
    ).to(weight_dtype).eval()

    print("Loading scheduler...")
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]

    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1

    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    print("Creating pipeline...")
    pipeline = Wan2_2AnimatePipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        clip_image_encoder=clip_image_encoder,
        scheduler=scheduler,
    )

    return pipeline, device, config, boundary


def setup_pipeline_optimizations(pipeline, device, config):
    """Setup memory optimizations and performance features."""
    # Multi-GPU setup
    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        from videox_fun.dist import shard_model

        pipeline.transformer.enable_multi_gpus_inference()
        if pipeline.transformer_2 is not None:
            pipeline.transformer_2.enable_multi_gpus_inference()

        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
            print("Added FSDP DIT")

        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Added FSDP TEXT ENCODER")

    # Compile
    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        if pipeline.transformer_2 is not None:
            for i in range(len(pipeline.transformer_2.blocks)):
                pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
        print("Added Compile")

    # GPU memory mode
    if GPU_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(pipeline.transformer, ["modulation"], device=device)
        pipeline.transformer.freqs = pipeline.transformer.freqs.to(device=device)
        if pipeline.transformer_2 is not None:
            replace_parameters_by_name(pipeline.transformer_2, ["modulation"], device=device)
            pipeline.transformer_2.freqs = pipeline.transformer_2.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)

    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(pipeline.transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
        if pipeline.transformer_2 is not None:
            convert_model_weight_to_float8(pipeline.transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(pipeline.transformer_2, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)

    elif GPU_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)

    elif GPU_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(pipeline.transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
        if pipeline.transformer_2 is not None:
            convert_model_weight_to_float8(pipeline.transformer_2, exclude_module_name=["modulation"], device=device)
            convert_weight_dtype_wrapper(pipeline.transformer_2, weight_dtype)
        pipeline.to(device=device)

    else:
        pipeline.to(device=device)

    # TeaCache
    if enable_teacache:
        coefficients = get_teacache_coefficients(model_name)
        if coefficients is not None:
            print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
            pipeline.transformer.enable_teacache(
                coefficients, num_inference_steps, teacache_threshold,
                num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
            )
            if pipeline.transformer_2 is not None:
                pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

    # CFG skip
    if cfg_skip_ratio is not None and cfg_skip_ratio > 0:
        print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
        if pipeline.transformer_2 is not None:
            pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

    # Riflex
    if enable_riflex:
        video_length = int((video_length - 1) // pipeline.vae.config.temporal_compression_ratio *
                          pipeline.vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // pipeline.vae.config.temporal_compression_ratio + 1
        pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)
        if pipeline.transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k=riflex_k, L_test=latent_frames)

    return pipeline


def load_flame_params(flame_path, target_length, device, dtype):
    """
    Load FLAME parameters from npy file.

    Args:
        flame_path: Path to npy file containing FLAME parameters
        target_length: Target number of frames
        device: Device to load tensors to
        dtype: Data type for tensors

    Returns:
        FLAME parameters tensor with shape (1, target_length, 56)
    """
    if flame_path is None or not os.path.exists(flame_path):
        raise ValueError(f"FLAME parameters file not found: {flame_path}")

    print(f"Loading FLAME parameters from: {flame_path}")
    flame_params = np.load(flame_path)

    if flame_params.ndim == 2:
        # Shape is (T, 56) - add batch dimension
        flame_params = flame_params[np.newaxis, :]
    elif flame_params.ndim == 1:
        # Shape is (56,) - add batch and frame dimensions
        flame_params = flame_params[np.newaxis, np.newaxis, :]

    T, C = flame_params.shape[1], flame_params.shape[2]
    print(f"Loaded FLAME parameters with shape: {flame_params.shape}")

    if C != 56:
        raise ValueError(f"FLAME parameters should have 56 channels, got {C}")

    # Pad or truncate to target length
    if T < target_length:
        # Pad by repeating the last frame
        padding = target_length - T
        flame_params = np.concatenate([
            flame_params,
            np.repeat(flame_params[:, -1:, :], padding, axis=1)
        ], axis=1)
        print(f"Padded FLAME parameters from {T} to {target_length} frames")
    elif T > target_length:
        # Truncate to target length
        flame_params = flame_params[:, :target_length, :]
        print(f"Truncated FLAME parameters from {T} to {target_length} frames")

    flame_tensor = torch.from_numpy(flame_params).to(device=device, dtype=dtype)
    return flame_tensor


# ==============================================================================
# Custom Pipeline with FLAME Support
# ==============================================================================

class Wan2_2AnimateFlamePipeline(Wan2_2AnimatePipeline):
    """
    Extended Wan2.2 Animate Pipeline with FLAME parameter support.
    Instead of using face_pixel_values, this pipeline uses FLAME parameters
    for facial animation control.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_flame_latents(self, flame_params, batch_size, device, dtype):
        """
        Prepare FLAME parameters for transformer input.

        Args:
            flame_params: FLAME parameters tensor (1, T, 56)
            batch_size: Batch size
            device: Device
            dtype: Data type

        Returns:
            Processed FLAME parameters ready for transformer
        """
        # FLAME params are already in the correct shape (B, T, C)
        # The transformer's flame_projection will handle dimension conversion
        flame = flame_params.to(device=device, dtype=dtype)

        return flame

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        pose_video=None,
        flame_params=None,  # FLAME parameters instead of face_video
        ref_image=None,
        bg_video=None,
        mask_video=None,
        replace_flag=True,
        guidance_scale: float = 6,
        generator=None,
        boundary: float = 0.875,
        shift: int = 5,
        **kwargs
    ):
        """
        Generate video using FLAME parameters for facial animation.
        """
        import math
        import copy
        from decord import VideoReader

        batch_size = 1
        device = self._execution_device
        weight_dtype = self.text_encoder.dtype
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance,
            num_videos_per_prompt=1, device=device, dtype=weight_dtype
        )

        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # Prepare pose video
        if pose_video is not None:
            video_length = pose_video.shape[2]
            pose_video = self.image_processor.preprocess(
                rearrange(pose_video, "b c f h w -> (b f) c h w"),
                height=height, width=width
            )
            pose_video = pose_video.to(dtype=torch.float32)
            pose_video = rearrange(pose_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            pose_video = None

        real_frame_len = pose_video.size()[2]
        target_len = self.get_valid_len(real_frame_len, num_frames, overlap=1)
        print(f'real frames: {real_frame_len} target frames: {target_len}')

        pose_video = self.inputs_padding(pose_video, target_len).to(device, weight_dtype)

        # Prepare FLAME parameters (instead of face_video)
        if flame_params is not None:
            flame_params = self.prepare_flame_latents(flame_params, batch_size, device, weight_dtype)
            # Pad FLAME params to match target length
            if flame_params.size(1) < target_len:
                # Pad by repeating the last frame
                padding = target_len - flame_params.size(1)
                flame_params = torch.cat([
                    flame_params,
                    flame_params[:, -1:, :].repeat(1, padding, 1)
                ], dim=1)
            elif flame_params.size(1) > target_len:
                flame_params = flame_params[:, :target_len, :]
        else:
            raise ValueError("flame_params must be provided for animation")

        # Prepare reference image
        ref_image = self.padding_resize(np.array(ref_image), height=height, width=width)
        ref_image = torch.tensor(ref_image / 127.5 - 1).unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0)
        ref_image = ref_image.to(device, weight_dtype)

        # Prepare background and mask if replace_flag is True
        if replace_flag:
            if bg_video is not None:
                video_length = bg_video.shape[2]
                bg_video = self.image_processor.preprocess(
                    rearrange(bg_video, "b c f h w -> (b f) c h w"),
                    height=height, width=width
                )
                bg_video = bg_video.to(dtype=torch.float32)
                bg_video = rearrange(bg_video, "(b f) c h w -> b c f h w", f=video_length)
                bg_video = self.inputs_padding(bg_video, target_len).to(device, weight_dtype)
            else:
                bg_video = None

            if mask_video is not None:
                mask_video = self.inputs_padding(mask_video, target_len).to(device, weight_dtype)
            else:
                mask_video = None

        # Scheduler setup
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            from videox_fun.utils.fm_solvers import get_sampling_sigmas
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler, device=device, sigmas=sampling_sigmas
            )
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler, num_inference_steps, device
            )

        self._num_timesteps = len(timesteps)

        target_shape = (
            self.vae.latent_channels,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            width // self.vae.spatial_compression_ratio,
            height // self.vae.spatial_compression_ratio
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3]) /
            (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) *
            target_shape[1]
        )

        # Denoising loop with sliding window
        start = 0
        end = num_frames
        all_out_frames = []
        bs = pose_video.size()[0]

        while start + 1 >= min(start + 1, pose_video.size()[2]):
            if start >= pose_video.size()[2]:
                break

            end = min(start + num_frames, pose_video.size()[2])
            if end - start < 2:
                break

            clip_len = end - start

            # Prepare latents
            latent_channels = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size, latent_channels, clip_len, height, width,
                weight_dtype, device, generator, None
            )

            # Prepare conditioning for current clip
            conditioning_pixel_values = pose_video[:, :, start:end]
            flame_clip = flame_params[:, start:end]
            ref_pixel_values = ref_image.clone().detach()

            # Prepare reference latents
            pose_latents, ref_latents = self.prepare_control_latents(
                conditioning_pixel_values, ref_pixel_values,
                batch_size, height, width, weight_dtype, device, generator,
                do_classifier_free_guidance
            )

            # Prepare masks
            mask_ref = self.get_i2v_mask(1, target_shape[-1], target_shape[-2], 1, device=device)
            y_ref = torch.concat([mask_ref, ref_latents], dim=1).to(device=device, dtype=weight_dtype)

            # Prepare y_reft
            if start > 0 and replace_flag and bg_video is not None:
                refer_t_pixel_values = out_frames[:, :, -1:].clone().detach()
                refer_t_pixel_values = (refer_t_pixel_values - 0.5) / 0.5

                bg_pixel_values = bg_video[:, :, start:end]
                y_reft = self.vae.encode(
                    torch.cat([
                        refer_t_pixel_values,
                        bg_pixel_values[:, :, 1:]
                    ], dim=2).to(device=device, dtype=weight_dtype)
                )[0].mode()

                mask_pixel_values = 1 - mask_video[:, :, start:end]
                mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
                mask_pixel_values = F.interpolate(
                    mask_pixel_values, size=(target_shape[-1], target_shape[-2]), mode='nearest'
                )
                mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b c t h w", b=bs)[:, 0]
                msk_reft = self.get_i2v_mask(
                    int((clip_len - 1) // self.vae.temporal_compression_ratio + 1),
                    target_shape[-1], target_shape[-2], 1,
                    mask_pixel_values=mask_pixel_values, device=device
                )
            elif start > 0:
                refer_t_pixel_values = out_frames[:, :, -1:].clone().detach()
                refer_t_pixel_values = (refer_t_pixel_values - 0.5) / 0.5
                refer_t_pixel_values = rearrange(refer_t_pixel_values, "b c t h w -> (b t) c h w")
                refer_t_pixel_values = F.interpolate(refer_t_pixel_values, size=(height, width), mode="bicubic")
                refer_t_pixel_values = rearrange(refer_t_pixel_values, "(b t) c h w -> b c t h w", b=bs)

                y_reft = self.vae.encode(
                    torch.cat([
                        refer_t_pixel_values,
                        torch.zeros(bs, 3, clip_len - 1, height, width).to(device=device, dtype=weight_dtype)
                    ], dim=2).to(device=device, dtype=weight_dtype)
                )[0].mode()
                msk_reft = self.get_i2v_mask(
                    int((clip_len - 1) // self.vae.temporal_compression_ratio + 1),
                    target_shape[-1], target_shape[-2], 1, device=device
                )
            else:
                if replace_flag and bg_video is not None:
                    bg_pixel_values = bg_video[:, :, start:end]
                    y_reft = self.vae.encode(
                        bg_pixel_values.to(device=device, dtype=weight_dtype)
                    )[0].mode()

                    mask_pixel_values = 1 - mask_video[:, :, start:end]
                    mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
                    mask_pixel_values = F.interpolate(
                        mask_pixel_values, size=(target_shape[-1], target_shape[-2]), mode='nearest'
                    )
                    mask_pixel_values = rearrange(mask_pixel_values, "(b t) c h w -> b c t h w", b=bs)[:, 0]
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1),
                        target_shape[-1], target_shape[-2], 0,
                        mask_pixel_values=mask_pixel_values, device=device
                    )
                else:
                    y_reft = self.vae.encode(
                        torch.zeros(bs, 3, clip_len, height, width).to(device=device, dtype=weight_dtype)
                    )[0].mode()
                    msk_reft = self.get_i2v_mask(
                        int((clip_len - 1) // self.vae.temporal_compression_ratio + 1),
                        target_shape[-1], target_shape[-2], 0, device=device
                    )

            y_reft = torch.concat([msk_reft, y_reft], dim=1).to(device=device, dtype=weight_dtype)
            y = torch.concat([y_ref, y_reft], dim=2)

            clip_context = self.clip_image_encoder([ref_pixel_values[0, :, :, :]]).to(device=device, dtype=weight_dtype)

            # Denoising loop
            self.transformer.num_inference_steps = num_inference_steps

            for i, t in enumerate(timesteps):
                self.transformer.current_steps = i

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                y_in = torch.cat([y] * 2) if do_classifier_free_guidance else y
                clip_context_input = torch.cat([clip_context] * 2) if do_classifier_free_guidance else clip_context
                pose_latents_input = torch.cat([pose_latents] * 2) if do_classifier_free_guidance else pose_latents

                # Prepare FLAME input (negative gets -1 mask, positive gets actual params)
                if do_classifier_free_guidance:
                    # For CFG, concatenate a masked version (all -1) with the actual flame params
                    flame_input = torch.cat([
                        torch.ones_like(flame_clip) * -1,  # negative: masked
                        flame_clip  # positive: actual FLAME params
                    ])
                else:
                    flame_input = flame_clip

                timestep = t.expand(latent_model_input.shape[0])

                # Select transformer based on timestep (for MoE)
                if self.transformer_2 is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                    else:
                        local_transformer = self.transformer
                else:
                    local_transformer = self.transformer

                # Predict noise with FLAME parameters
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = local_transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=y_in,
                        clip_fea=clip_context_input,
                        pose_latents=pose_latents_input,
                        flame=flame_input,  # FLAME parameters instead of face_pixel_values
                    )

                # Apply guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            out_frames = self.decode_latents(latents[:, :, 1:])
            all_out_frames.append(out_frames.cpu())

            start += clip_len - 1
            if start >= pose_video.size()[2]:
                break

        videos = torch.cat(all_out_frames, dim=2)[:, :, :real_frame_len]
        self.maybe_free_model_hooks()

        return WanPipelineOutput(videos=videos.float().cpu())


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None, **kwargs):
    """Helper function to retrieve timesteps from scheduler."""
    import inspect

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


# ==============================================================================
# Main Inference Function
# ==============================================================================

def run_inference():
    """Main inference function."""
    print("=" * 60)
    print("Wan2.2 Animate LoRA Inference with FLAME Parameters")
    print("=" * 60)

    # Load models
    pipeline, device, config, boundary = load_models()

    # Setup optimizations
    pipeline = setup_pipeline_optimizations(pipeline, device, config)

    # Replace pipeline with FLAME-supporting version
    flame_pipeline = Wan2_2AnimateFlamePipeline(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        vae=pipeline.vae,
        transformer=pipeline.transformer,
        transformer_2=pipeline.transformer_2,
        clip_image_encoder=pipeline.clip_image_encoder,
        scheduler=pipeline.scheduler,
    )

    # Copy over device hooks
    flame_pipeline._device = pipeline._device
    flame_pipeline._execution_device = pipeline._execution_device

    # Merge LoRA weights (without using PEFT)
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Merging LoRA weights from: {lora_path}")
        flame_pipeline = merge_lora(flame_pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if flame_pipeline.transformer_2 is not None and lora_high_path is not None:
            flame_pipeline = merge_lora(
                flame_pipeline, lora_high_path, lora_high_weight,
                device=device, dtype=weight_dtype, sub_transformer_name="transformer_2"
            )

    # Set generator
    generator = torch.Generator(device=device).manual_seed(seed)

    # Prepare inputs
    print("Preparing input data...")

    # Adjust video length
    video_length = int(
        (video_length - 1) // flame_pipeline.vae.config.temporal_compression_ratio *
        flame_pipeline.vae.config.temporal_compression_ratio
    ) + 1 if video_length != 1 else 1

    # Load pose video
    pose_video, _, _, _ = get_video_to_video_latent(
        src_pose_path, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None
    )

    # Load reference image
    ref_image = get_image(src_ref_path)

    # Load FLAME parameters
    flame_params = load_flame_params(flame_params_path, video_length, device, weight_dtype)

    # Load background and mask (optional)
    if os.path.exists(src_bg_path):
        bg_video, _, _, _ = get_video_to_video_latent(
            src_bg_path, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None
        )
        mask_video, _, _, _ = get_video_to_video_latent(
            src_mask_path, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None
        )
        mask_video = mask_video[:, :1]
        replace_flag = True
    else:
        bg_video = None
        mask_video = None
        replace_flag = False

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        sample = flame_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            num_frames=video_length,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            boundary=boundary,
            pose_video=pose_video,
            flame_params=flame_params,  # FLAME parameters instead of face_video
            ref_image=ref_image,
            bg_video=bg_video,
            mask_video=mask_video,
            replace_flag=replace_flag,
            shift=shift,
        ).videos

    # Unmerge LoRA weights
    if lora_path is not None and os.path.exists(lora_path):
        flame_pipeline = unmerge_lora(flame_pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
        if flame_pipeline.transformer_2 is not None and lora_high_path is not None:
            flame_pipeline = unmerge_lora(
                flame_pipeline, lora_high_path, lora_high_weight,
                device=device, dtype=weight_dtype, sub_transformer_name="transformer_2"
            )

    # Save results
    print("Saving results...")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")
        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

    print(f"Saved result to: {video_path}")
    print("=" * 60)
    print("Inference completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_inference()
