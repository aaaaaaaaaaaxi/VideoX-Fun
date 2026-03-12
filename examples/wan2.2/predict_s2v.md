# predict_s2v.py 代码分析

## 概述

`predict_s2v.py` 是 **Wan2.2-S2V (Speech-to-Video)** 的推理脚本，用于根据语音、姿势视频和参考图像生成说话人视频。

### 核心功能流程

```
输入：音频 + 姿势视频 + 参考图像 -> 输出：合成说话视频
```

---

## 重要参数分类说明

### 1. GPU内存优化模式 (第46行)

| 模式 | 说明 | 推荐场景 |
|------|------|----------|
| `model_full_load` | 完整加载到GPU | 显存充足时 |
| `model_full_load_and_qfloat8` | float8量化+完整加载 | 节省显存，轻微精度损失 |
| `model_cpu_offload` | 用完后移到CPU | 中等显存 |
| `model_cpu_offload_and_qfloat8` | CPU offload + float8 | 显存紧张 |
| `sequential_cpu_offload` | 逐层CPU卸载 | **极低显存，速度慢** |

```python
GPU_memory_mode = "sequential_cpu_offload"
```

### 2. 多GPU配置 (第51-54行)

```python
ulysses_degree = 1      # 注意力头切分（head-split）
ring_degree = 1         # 序列切分（sequence-split）
```

**重要**：`ulysses_degree × ring_degree = GPU数量`

示例：
- 1 GPU: `ulysses_degree=1, ring_degree=1`
- 8 GPU: `ulysses_degree=2, ring_degree=4`

```python
fsdp_dit = False              # 使用FSDP节省GPU显存（transformer）
fsdp_text_encoder = True      # 使用FSDP节省GPU显存（text_encoder）
```

### 3. TeaCache加速 (第61-73行)

Wan模型专用的推理加速技术，通过缓存中间结果减少计算。

```python
enable_teacache = True           # 启用缓存加速
teacache_threshold = 0.10        # 缓存阈值（0.05~0.30）
num_skip_start_steps = 5         # 开始跳过的步数
teacache_offload = False         # 将缓存张量移到CPU
```

**推荐阈值设置**：

| 模型 | 推荐阈值 |
|------|----------|
| Wan2.2-T2V-A14B | 0.10~0.15 |
| Wan2.2-I2V-A14B | 0.15~0.20 |

- **阈值越大**：缓存步数越多，速度越快，但可能与原始结果有差异
- **num_skip_start_steps**：前几步不使用缓存，保证生成质量

### 4. CFG Skip Ratio (第77行)

```python
cfg_skip_ratio = 0        # 0.00~0.25，跳过部分CFG步骤以加速
```

### 5. Riflex配置 (第80-82行)

```python
enable_riflex = False     # 启用Riflex（内在频率调制）
riflex_k = 6              # 内在频率索引
```

### 6. 模型配置 (第85-99行)

```python
config_path = "config/wan2.2/wan_civitai_s2v.yaml"     # 配置文件路径
model_name = "models/Diffusion_Transformer/Wan2.2-S2V-14B"  # 模型路径

# 预训练模型加载（可选）
transformer_path = None           # 低噪声模型权重
transformer_high_path = None      # 高噪声模型权重（MoE架构）
vae_path = None                   # VAE权重
```

### 7. 采样器配置 (第89-93行)

```python
sampler_name = "Flow"        # 选项: Flow / Flow_Unipc / Flow_DPM++
shift = 3                    # 噪声调度偏移（影响时序动态）
```

- **Flow**: 默认Euler离散求解器
- **Flow_Unipc**: UniPC多步求解器
- **Flow_DPM++**: DPM++多步求解器

`shift` 参数影响时间维度的动态表现。

### 8. LoRA配置 (第100-103, 129-130行)

```python
# LoRA路径
lora_path = None              # 低噪声模型LoRA
lora_high_path = None         # 高噪声模型LoRA

# LoRA权重
lora_weight = 0.55            # 低噪声模型LoRA权重
lora_high_weight = 0.55       # 高噪声模型LoRA权重
```

### 9. 生成参数 (第106-127行)

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `sample_size` | 输出分辨率 [高, 宽] | `[832, 480]` |
| `video_length` | 生成帧数 | `80` |
| `fps` | 帧率 | `16` |
| `guidance_scale` | CFG引导强度 | `4.5` |
| `num_inference_steps` | 采样步数 | `40` |
| `seed` | 随机种子 | `43` |
| `weight_dtype` | 数据类型 | `torch.bfloat16` |

```python
sample_size = [832, 480]
video_length = 80
fps = 16
guidance_scale = 4.5
num_inference_steps = 40
seed = 43
weight_dtype = torch.bfloat16  # V100/2080Ti需改为torch.float16
```

### 10. 输入路径 (第113-120行)

```python
control_video = "asset/pose.mp4"      # 姿势控制视频
ref_image = "asset/8.png"             # 参考图像（人物外观）
audio_path = "asset/talk.wav"         # 输入语音
init_first_frame = False              # 是否用ref_image作为首帧
```

### 11. 提示词 (第122-124行)

```python
prompt = "一个女孩在海边说话。"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
```

### 12. 输出配置 (第131行)

```python
save_path = "samples/wan-videos-speech2v"
```

---

## 代码执行流程

### 1. 模型加载 (137-217行)

```python
# 加载双transformer（MoE架构支持）
transformer = Wan2_2Transformer3DModel_S2V.from_pretrained(...)

# 加载VAE
vae = AutoencoderKLWan.from_pretrained(...)

# 加载文本编码器
text_encoder = WanT5EncoderModel.from_pretrained(...)

# 加载音频编码器
audio_encoder = WanAudioEncoder(...)
```

### 2. 采样器初始化 (220-229行)

```python
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)
```

### 3. Pipeline构建 (232-240行)

```python
pipeline = Wan2_2S2VPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    audio_encoder=audio_encoder,
)
```

### 4. 多GPU配置 (241-255行)

```python
if ulysses_degree > 1 or ring_degree > 1:
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
```

### 5. 模型编译 (257-263行)

```python
if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
```

### 6. 内存优化应用 (265-289行)

根据 `GPU_memory_mode` 应用对应的优化策略：

```python
if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    pipeline.enable_model_cpu_offload(device=device)
# ... 其他模式
```

### 7. TeaCache启用 (291-298行)

```python
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold,
        num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
```

### 8. CFG Skip启用 (300-304行)

```python
if cfg_skip_ratio is not None:
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
```

### 9. LoRA合并 (308-311行)

```python
if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
```

### 10. 推理执行 (313-344行)

```python
with torch.no_grad():
    # 处理参考图像
    if ref_image is not None:
        ref_image = get_image_latent(ref_image, sample_size=sample_size)

    # 处理姿势视频
    pose_video, _, _, _ = get_video_to_video_latent(control_video, ...)

    # 执行生成
    sample = pipeline(
        prompt,
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height = sample_size[0],
        width = sample_size[1],
        generator = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,
        boundary = boundary,
        ref_image = ref_image,
        pose_video = pose_video,
        audio_path = audio_path,
        shift = shift,
        fps = fps,
        init_first_frame = init_first_frame
    ).videos
```

### 11. LoRA卸载 (346-349行)

```python
if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
```

### 12. 结果保存 (351-376行)

```python
def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)

    # 合并音频和视频
    merge_video_audio(video_path=video_path, audio_path=audio_path)
```

---

## 使用示例

### 单GPU推理（默认配置）

```bash
python examples/wan2.2/predict_s2v.py
```

### 多GPU推理（8卡）

```bash
torchrun --nproc-per-node=8 examples/wan2.2/predict_s2v.py
```

需在脚本中设置：
```python
ulysses_degree = 2
ring_degree = 4
```

---

## 注意事项

1. **显存不足**：使用 `sequential_cpu_offload` 或 `model_cpu_offload_and_qfloat8`
2. **V100/2080Ti显卡**：设置 `weight_dtype = torch.float16`
3. **加速生成**：启用 `enable_teacache` 和合适的 `cfg_skip_ratio`
4. **MoE架构**：同时加载 `transformer`（低噪声）和 `transformer_2`（高噪声）
5. **输入要求**：
   - 姿势视频：控制人物动作
   - 参考图像：定义人物外观
   - 音频：驱动说话口型
