# Wan2.2 Animate LoRA Inference

本目录包含 Wan2.2 Animate 模型与 LoRA 适配器的推理脚本。

## 文件说明

### 1. `inference_animate_lora.py`
基础推理脚本，提供文本到视频生成功能：
- 文本提示生成视频 (T2V)
- 基本的 LoRA 加载
- 简单的扩散推理循环
- 视频保存功能

### 2. `inference_animate_lora_advanced.py`
高级推理脚本，支持多种生成模式：
- 文本到视频 (T2V)
- 图像到视频 (I2V)
- 控制生成 (Controlled)
- 高级参数配置
- 更多控制选项

## 基础使用

### 文本到视频生成

```bash
python inference_animate_lora.py \
    --base_model_path /path/to/wan2.2/base/model \
    --lora_path /path/to/your/lora/weights \
    --prompt "A cat walking in the garden" \
    --negative_prompt "blurry, low quality" \
    --num_frames 49 \
    --height 480 \
    --width 720 \
    --num_inference_steps 50 \
    --guidance_scale 7.0 \
    --output_dir ./outputs \
    --seed 42
```

### 高级功能使用

#### 图像到视频生成

```bash
python inference_animate_lora_advanced.py \
    --base_model_path /path/to/wan2.2/base/model \
    --lora_path /path/to/your/lora/weights \
    --mode image_to_video \
    --prompt "A lion roaring in the savanna" \
    --start_image ./input_image.png \
    --num_frames 25 \
    --height 512 \
    --width 512 \
    --guidance_scale 8.0
```

#### 控制生成

```bash
python inference_animate_lora_advanced.py \
    --base_model_path /path/to/wan2.2/base/model \
    --lora_path /path/to/your/lora/weights \
    --mode controlled \
    --prompt "A person walking" \
    --control_type pose \
    --control_image ./pose_control.png \
    --num_frames 16 \
    --height 256 \
    --width 256 \
    --num_inference_steps 30
```

## 参数说明

### 必需参数
- `--base_model_path`: Wan2.2 基础模型路径
- `--lora_path`: LoRA 权重文件路径
- `--prompt`: 文本提示

### 视频参数
- `--height`: 视频高度 (默认: 480)
- `--width`: 视频宽度 (默认: 720)
- `--num_frames`: 帧数 (默认: 49)
- `--fps`: 帧率 (默认: 8)

### 生成参数
- `--num_inference_steps`: 扩散步数 (默认: 50)
- `--guidance_scale`: 引导强度 (默认: 7.0)
- `--eta`: 调度器 eta 参数 (默认: 0.0)
- `--num_videos_per_prompt`: 每个提示生成的视频数 (默认: 1)

### 设备设置
- `--device`: 目标设备 (cuda/cpu)
- `--dtype`: 数据类型 (float16/float32)
- `--scheduler`: 调度器类型 (euler/dpm/unipc)
- `--seed`: 随机种子

### 高级选项
- `--control_type`: 控制类型 (canny/depth/pose/mlsd)
- `--control_image`: 控制图像路径
- `--start_image`: 起始图像路径 (I2V 模式)
- `--reference_image`: 参考图像路径

## 输出

生成的视频会保存在指定的输出目录中，文件名基于提示文本自动生成。

## 依赖要求

确保已安装以下依赖：
```bash
pip install torch diffusers transformers accelerate peft opencv-python
```

## 性能优化

### 1. 内存优化
- 使用 `--dtype float16` 减少内存使用
- 启用模型卸载：`--device cuda` + 混合精度

### 2. 速度优化
- 使用更少的扩散步数：`--num_inference_steps 30`
- 使用 DPM 或 UniPC 调度器：`--scheduler dpm`
- 启用 xformers（如果可用）

### 3. 质量优化
- 增加引导强度：`--guidance_scale 8.0-10.0`
- 增加扩散步数：`--num_inference_steps 50-100`
- 使用更好的调度器：`--scheduler unipc`

## 故障排除

### 1. 内存不足
- 减少 `--num_frames`
- 使用 `--dtype float16`
- 减少 `--height` 和 `--width`

### 2. LoRA 加载失败
- 检查 LoRA 文件路径是否正确
- 确保 LoRA 文件格式正确 (safetensors)
- 检查基础模型版本是否兼容

### 3. 生成质量差
- 增加 `--num_inference_steps`
- 调整 `--guidance_scale`
- 提供更详细的提示
- 使用更好的调度器

## 示例命令

### 1. 基础文本生成
```bash
python inference_animate_lora.py \
    --base_model_path ./models/wan2.2-base \
    --lora_path ./models/your-lora.safetensors \
    --prompt "A beautiful sunset over the ocean" \
    --num_frames 49 \
    --guidance_scale 7.0
```

### 2. 高质量生成
```bash
python inference_animate_lora_advanced.py \
    --base_model_path ./models/wan2.2-base \
    --lora_path ./models/your-lora.safetensors \
    --prompt "A majestic dragon flying through the clouds, detailed scales, cinematic lighting" \
    --negative_prompt "blurry, low quality, bad art" \
    --num_frames 81 \
    --num_inference_steps 100 \
    --guidance_scale 9.0 \
    --scheduler dpm \
    --seed 1234
```

### 3. 图像到视频
```bash
python inference_animate_lora_advanced.py \
    --base_model_path ./models/wan2.2-base \
    --lora_path ./models/your-lora.safetensors \
    --mode image_to_video \
    --prompt "A person walking in the park" \
    --start_image ./input.png \
    --num_frames 25 \
    --height 512 \
    --width 512
```

## 注意事项

1. 确保基础模型和 LoRA 版本兼容
2. LoRA 文件应为 safetensors 格式
3. 生成视频可能需要较大的 GPU 内存
4. 建议使用至少 16GB GPU 内存的设备