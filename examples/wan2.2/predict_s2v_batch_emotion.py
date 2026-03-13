import os
import sys
import json
import argparse
from tqdm import tqdm
from contextlib import contextmanager

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               AutoTokenizer, CLIPModel,
                               Wan2_2Transformer3DModel_S2V, WanAudioEncoder,
                               WanT5EncoderModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2S2VPipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    merge_video_audio, save_videos_grid)

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
GPU_memory_mode     = "model_cpu_offload"

# Multi GPUs config
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory.
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = True
teacache_threshold  = 0.15
num_skip_start_steps = 5
teacache_offload    = False

# Skip some cfg steps in inference
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_s2v.yaml"
model_name          = "models/Diffusion_Transformer/Wan2.2-S2V-14B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
shift               = 3

# Load pretrained model if need
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
lora_path               = None
lora_high_path          = None

# Other params
# sample_size         = [832, 480]
sample_size         = [512, 512]
video_length        = 80
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype            = torch.bfloat16

# The path of the pose control video
control_video           = None

# ============================================
# Parse command line arguments
# ============================================
parser = argparse.ArgumentParser(description='Batch generate videos with S2V model')
parser.add_argument('--block', type=str, default='01', help='Block number (01-30)')
parser.add_argument('--ref_image_folder', type=str, default='path/to/ref/images', help='Folder containing reference images')
parser.add_argument('--audio_folder', type=str, default='path/to/audio', help='Folder containing audio files')
parser.add_argument('--ref_image_extension', type=str, default='.png', help='Reference image file extension')
parser.add_argument('--audio_extension', type=str, default='.wav', help='Audio file extension')
args = parser.parse_args()

# ============================================
# Batch processing parameters
# ============================================
# Block number (from command line argument)
block = args.block
# Path to the block JSON file (auto-generated from block number)
block_json_file_path = f"data_predict/blocks/{block}.json"
# Folder containing reference images (from command line argument)
ref_image_folder = args.ref_image_folder
# Folder containing audio files (from command line argument)
audio_folder = args.audio_folder
# Use ref_image as the first frame
init_first_frame = False

# Reference image and audio file extensions (from command line arguments)
ref_image_extension = args.ref_image_extension
audio_extension = args.audio_extension

# ============================================
# Emotion to prompt mapping
# ============================================
emotion_prompts = {
    "angry": "A person is speaking with an angry expression.",
    "contempt": "A person is speaking with a contemptuous expression.",
    "disgusted": "A person is speaking with a disgusted expression.",
    "fear": "A person is speaking with a scared expression.",
    "happy": "A person is speaking with a joyful expression.",
    "neutral": "A person is speaking with a neutral expression.",
    "sad": "A person is speaking with a sorrowful expression.",
    "surprised": "A person is speaking with a surprised expression.",
}
# Default prompt when emotion is not found
default_prompt = "A person is speaking."

# ============================================
# Prompts
# ============================================
negative_prompt     = "Over-saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, static frame, grayish overall, worst quality, low quality, JPEG compression artifacts, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static scene, messy background, three legs, crowded background, walking backwards"
guidance_scale      = 4.5
seed                = 43
num_inference_steps = 40
lora_weight         = 0.55
lora_high_weight    = 0.55
# Save path (block number will be appended)
save_path           = "samples/wan-videos-speech2v"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

config = OmegaConf.load(config_path)
boundary = config['transformer_additional_kwargs'].get('boundary', 0.900)

transformer = Wan2_2Transformer3DModel_S2V.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
else:
    transformer_2 = None

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None:
    if transformer_high_path is not None:
        print(f"From checkpoint: {transformer_high_path}")
        if transformer_high_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_high_path)
        else:
            state_dict = torch.load(transformer_high_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer_2.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

audio_encoder = WanAudioEncoder(
    os.path.join(model_name, config['audio_encoder_kwargs'].get('audio_encoder_subpath', 'audio_encoder')),
    "cpu"
)

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = Wan2_2S2VPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    audio_encoder=audio_encoder,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
    if transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, dtype=weight_dtype, sub_transformer_name="transformer_2")

# ============================================
# Batch Processing
# ============================================

# Load block JSON file
with open(block_json_file_path, 'r', encoding='utf-8') as f:
    block_data = json.load(f)

metadata = block_data['metadata']
mapping = block_data['mapping']

print(f"\n{'='*60}")
print(f"Batch Processing - Block {metadata['block_id']}/{metadata['total_blocks']}")
print(f"{'='*60}")
print(f"Total files in this block: {metadata['block_size']}")
print(f"Dataset: {', '.join(metadata['datasets'])}")
print(f"Emotions: {', '.join(metadata['emotions'])}")
print(f"{'='*60}\n")

# Create save path with block number
save_path_block = os.path.join(save_path, f"block_{block}")

# Process each file in the block
processed_count = 0
skipped_count = 0
failed_count = 0

# Convert mapping to list for tqdm
items_list = list(mapping.items())

# Disable pipeline internal progress bar
os.environ['DIFFUSERS_DISABLE_PROGRESS_BAR'] = '1'

# Create outer progress bar for batch processing
pbar = tqdm(items_list, total=metadata['block_size'], desc=f"Block {block}", unit="video", position=0)

for filename, info in pbar:
    ref_image_path = os.path.join(ref_image_folder, filename + ref_image_extension)
    audio_path = os.path.join(audio_folder, filename + audio_extension)

    # Update progress bar with current file info
    pbar.set_postfix({
        'file': filename[:12] if len(filename) > 12 else filename,
        'data': info.get('dataset', 'N/A')[:6],
        'emo': info.get('emotion', 'N/A')[:6],
        'ok': processed_count,
        'skip': skipped_count,
        'fail': failed_count
    })

    # Check if both files exist
    if not os.path.exists(ref_image_path):
        skipped_count += 1
        continue
    if not os.path.exists(audio_path):
        skipped_count += 1
        continue

    try:
        generator = torch.Generator(device=device).manual_seed(seed)

        # Get emotion-specific prompt
        emotion = info.get('emotion', 'neutral').lower()
        current_prompt = emotion_prompts.get(emotion, default_prompt)

        # Create inner progress bar for inference steps
        step_pbar = tqdm(total=num_inference_steps, desc=f"  {filename[:15]} [{emotion}]", unit="step", position=1, leave=False, ncols=100)

        # Define callback to update step progress
        def step_callback(pipe, i, t, callback_kwargs):
            step_pbar.update(1)
            step_pbar.set_postfix({'step': f'{i+1}/{num_inference_steps}'})
            return callback_kwargs

        # Get emotion-specific prompt
        emotion = info.get('emotion', 'neutral').lower()
        current_prompt = emotion_prompts.get(emotion, default_prompt)

        with torch.no_grad():
            video_length_processed = video_length // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio if video_length != 1 else 1
            latent_frames = video_length_processed // vae.config.temporal_compression_ratio

            if enable_riflex:
                pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
                if transformer_2 is not None:
                    pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

            # Load reference image
            ref_image = get_image_latent(ref_image_path, sample_size=sample_size)

            pose_video, _, _, _ = get_video_to_video_latent(control_video, video_length=video_length_processed, sample_size=sample_size, fps=fps, ref_image=None)

            sample = pipeline(
                current_prompt,
                num_frames = video_length_processed,
                negative_prompt = negative_prompt,
                height      = sample_size[0],
                width       = sample_size[1],
                generator   = generator,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps,
                boundary = boundary,
                callback_on_step_end = step_callback,
                callback_on_step_end_tensor_inputs = ["latents"],

                ref_image = ref_image,
                pose_video = pose_video,
                audio_path = audio_path,
                shift = shift,
                fps = fps,
                init_first_frame = init_first_frame
            ).videos

        # Close inner progress bar
        step_pbar.close()

        # Save results
        if not os.path.exists(save_path_block):
            os.makedirs(save_path_block, exist_ok=True)

        index = len([path for path in os.listdir(save_path_block)]) + 1
        # 不需要index
        prefix = f"{filename}"
        video_path = os.path.join(save_path_block, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

        # Merge video with audio
        merge_video_audio(video_path=video_path, audio_path=audio_path)

        processed_count += 1

    except Exception as e:
        if 'step_pbar' in locals():
            step_pbar.close()
        failed_count += 1
        continue

# Print summary
print(f"\n{'='*60}")
print(f"Batch Processing Complete - Block {block}")
print(f"{'='*60}")
print(f"Processed successfully: {processed_count}")
print(f"Skipped (missing files): {skipped_count}")
print(f"Failed (errors): {failed_count}")
print(f"Total in block: {metadata['block_size']}")
print(f"Output directory: {save_path_block}")
print(f"{'='*60}\n")
