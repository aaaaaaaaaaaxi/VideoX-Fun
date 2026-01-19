# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VideoX-Fun is a comprehensive AI/ML video generation pipeline focused on Diffusion Transformer models. It supports text-to-video (T2V), image-to-video (I2V), video-to-video (V2V) generation, and controlled video generation using various conditions (Canny, Depth, Pose, MLSD, etc.).

## Common Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install for development
pip install -e .

# Optional: Install xfuser for multi-GPU support
pip install xfuser==0.4.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
pip install yunchang==0.6.2 --progress-bar off -i https://mirrors.aliyun.com/pypi/simple/
```

### Inference
```bash
# Single-GPU inference (example for CogVideoX T2V)
python examples/cogvideox_fun/predict_t2v.py

# Multi-GPU inference (8 GPUs example)
torchrun --nproc-per-node=8 examples/wan2.1_fun/predict_t2v.py

# Run web UI (example for CogVideoX)
python examples/cogvideox_fun/app.py
```

### Training
```bash
# Train baseline model (example)
sh scripts/cogvideox_fun/train.sh

# Train LoRA model (example)
sh scripts/cogvideox_fun/train_lora.sh

# Train control model (example)
sh scripts/cogvideox_fun/train_control.sh

# Multi-GPU training (accelerate)
accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --config_path="config/cogvideox/config.yaml" \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1
```

### Environment Setup and Validation
```bash
# Validate Python environment
python --version  # Should be 3.10 or 3.11

# Check PyTorch CUDA support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install development dependencies
pip install -e ".[dev]"
```

### Model Management
```bash
# Create model directories
mkdir -p models/Diffusion_Transformer
mkdir -p models/Personalized_Model

# ComfyUI model structure
mkdir -p ComfyUI/models/Fun_Models/
```

## Code Architecture

### Directory Structure
- `examples/` - Model-specific inference scripts and web UIs organized by model type
  - Each model has its own subdirectory (cogvideox_fun, wan2.1_fun, flux2, etc.)
  - Contains predict_t2v.py, predict_i2v.py, predict_v2v.py, and app.py for web interface
- `scripts/` - Training scripts with parallel structure to examples
  - train.sh, train_lora.sh, train_control.sh for each model
- `videox_fun/` - Core package with main functionality
  - `api/` - API endpoints (single-node and multi-node)
  - `models/` - Model implementations (transformers, VAEs, encoders)
  - `pipeline/` - Inference pipelines for all models
  - `ui/` - Gradio UI components
  - `utils/` - Utility functions (LoRA, FP8, optimization)
  - `dist/` - Distributed training and parallelization utilities
  - `data/` - Dataset handling and sampling strategies
- `comfyui/` - ComfyUI integration for node-based workflows
- `config/` - YAML configuration files for each model
- `models/` - Pre-trained model storage location
- `datasets/` - Dataset storage and organization
- `reports/` - Training reports and logs

### Core Package Structure (`videox_fun/`)

#### Models
- **Transformer implementations**: All model architectures are supported
  - `cogvideox_transformer3d.py` - CogVideoX video generation
  - `wan_transformer3d.py`, `wan_transformer3d_animate.py`, `wan_transformer3d_vace.py` - Wan models
  - `flux_transformer2d.py`, `flux2_transformer2d.py` - Flux image generation
  - `qwenimage_transformer2d.py` - Qwen-Image
  - `z_image_transformer2d.py` - Z-Image
- **VAE implementations**: Video/image compression for each model
- **Encoders**: Text, image, and audio encoders

#### Pipelines
Each model has dedicated pipelines supporting:
- Standard generation (T2V, I2V, T2I)
- ControlNet-based controlled generation
- Animate and inpaint variants
- Camera control and trajectory control

#### Distributed Training (`dist/`)
- **xfuser integration**: Multi-GPU parallelization
- **FSDP support**: Model sharding for large models
- Model-specific implementations for optimal performance

#### Key Features
- **LoRA support**: Merging/unmerging LoRA weights
- **Memory optimization**: FP8 quantization, CPU offloading
- **Training optimizations**: Bucket training, gradient checkpointing

#### GPU Memory Optimization
All prediction scripts support memory optimization modes:
- `model_full_load` - Full GPU loading
- `model_full_load_and_qfloat8` - Full loading with float8 quantization
- `model_cpu_offload` - Model offloaded to CPU after use
- `model_cpu_offload_and_qfloat8` - CPU offload with float8 quantization
- `sequential_cpu_offload` - Layer-wise CPU offload (slow but memory efficient)

#### Multi-GPU Support
Uses xfuser for distributed inference with configurable parallelization:
- `ulysses_degree` - Head-split parallelization
- `ring_degree` - Sequence-split parallelization
- Product must equal number of GPUs

### Training Modes and Configurations

#### Training Types
All models support multiple training approaches:
- **Baseline training**: `train.py` - Full model training
- **LoRA fine-tuning**: `train_lora.py` - Parameter-efficient adaptation
- **Control model training**: `train_control.py` - ControlNet integration
- **Reward-guided training**: `train_reward_lora.py` - Human preference optimization
- **Distillation**: `train_distill.py` - Knowledge distillation
- **Animate models**: `train_animate.py` - Animation-specific training

#### Training Configuration
Key parameters in training scripts:
- `--enable_bucket` - Bucket training for variable resolutions
- `--random_hw_adapt` - Automatic height/width scaling
- `--video_sample_n_frames` - Frames (49, 81, 121 based on model)
- `--video_sample_stride` - Frame sampling rate
- `--train_mode` - "normal", "inpaint", "control", "animate"
- `--gradient_checkpointing` - Memory optimization during training

#### Configuration Format
YAML configs specify:
- `format`: civitai or diffusers
- `pipeline`: Model type (Wan, flux2, qwenimage, etc.)
- Component-specific parameters for transformer, VAE, text encoder, scheduler
- Path mappings for model components

### Web UI
- Gradio-based interface available for each model
- Access at http://localhost:7860 when running app.py
- Supports all generation modes with interactive controls

### ComfyUI Integration
- Complete node-based workflow support
- Models can be loaded directly in ComfyUI or chunked across directories
- Additional nodes for preprocessing (ControlNet weights)

## Important Notes

### Model Zoo
Models are available on Hugging Face and ModelScope - see README.md for complete list with download links.

### Environment Requirements
- Python 3.10/3.11
- PyTorch 2.2.0
- CUDA 11.8/12.1
- ~60GB disk space for model weights

### Data Format
Training datasets require JSON metadata with:
```json
[
    {
        "file_path": "train/00000001.mp4",
        "text": "Description of the video",
        "type": "video"
    }
]
```

### Output Directories
- T2V: `samples/{model_name}-videos`
- I2V: `samples/{model_name}-videos_i2v`
- V2V: `samples/{model_name}-videos_v2v`
- Control: `samples/{model_name}-videos_v2v_control`

### Model Optimization Techniques

#### Memory Management
- **TeaCache**: Wan model optimization for inference (enable_teacache=True)
- **CFG Skip Ratio**: Skip some CFG steps for faster generation (cfg_skip_ratio>0)
- **Sparse Attention**: Optional PAI fuser kernels for memory efficiency
- **Fast RoPE Kernels**: Optional PAI fuser optimization

#### Performance Optimizations
- **Model Compilation**: Optional torch.compile() support
- **Float8 Quantization**: FP8 precision for memory savings
- **Layer-wise CPU Offloading**: Sequential offloading for extreme memory efficiency
- **Mixed Precision Training**: BF16 for faster training with same accuracy

#### Generation Parameters
Common parameters across all models:
- `sample_size` - [height, width]
- `video_length` - Number of frames (model-dependent)
- `fps` - Frames per second (model-dependent)
- `guidance_scale` - CFG guidance strength
- `num_inference_steps` - Sampling steps
- `enable_teacache` - Wan model caching
- `teacache_threshold` - Cache threshold for TeaCache