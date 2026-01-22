# 构建动画训练数据集指南

本文档介绍如何使用 `build_animate_dataset.py` 脚本来构建训练数据集的JSON文件。

## 目录结构要求

在运行脚本之前，请确保您的数据目录结构如下：

```
dataset/
├── videos/          # 存放所有训练视频文件 (.mp4)
│   ├── 00000001.mp4
│   ├── 00000002.mp4
│   └── ...
├── references/      # 存放参考图像文件 (.png)
│   ├── 00000001.png
│   ├── 00000002.png
│   └── ...
├── flame_params/    # 存放FLAME参数文件 (.pkl)
│   ├── 00000001.pkl
│   ├── 00000002.pkl
│   └── ...
└── dataset.json     # 生成的数据集文件
```

## 使用方法

### 方法1：使用命令行（推荐）

```bash
python build_animate_dataset.py \
    --video_dir path/to/videos \
    --ref_dir path/to/references \
    --flame_dir path/to/flame_params \
    --output dataset.json \
    --text "视频中的人在做讲话" \
    --height 512 \
    --width 512
```

```bash
python build_animate_dataset.py \
    --video_dir /hpc2hdd/home/ntang745/workspace/CelebV-HQ/35666_resample \
    --ref_dir /hpc2hdd/home/ntang745/workspace/CelebV-HQ/35666_first_frame \
    --flame_dir /hpc2hdd/home/ntang745/workspace/CelebV-HQ/35666_resample_flame_params \
    --output dataset.json \
    --text "视频中的人在讲话" \
    --height 512 \
    --width 512
```

```bash
python build_animate_dataset.py \
    --video_dir /hpc2hdd/home/ntang745/workspace/MEAD/crop_head_512 \
    --ref_dir /hpc2hdd/home/ntang745/workspace/MEAD/crop_head_512_first_frame \
    --flame_dir /hpc2hdd/home/ntang745/workspace/MEAD/smirk_processed/flame_params \
    --output dataset_MEAD.json \
    --text "视频中的人在讲话" \
    --height 512 \
    --width 512
```

### 方法2：使用Python函数调用

```python
from build_animate_dataset import build_dataset_json

build_dataset_json(
    video_dir="path/to/videos",
    ref_dir="path/to/references",
    flame_dir="path/to/flame_params",
    output_file="dataset.json",
    text_template="视频中的人在做讲话",
    height=512,
    width=512
)
```

## 参数说明

| 参数 | 类型 | 必需 | 默认值 | 描述 |
|------|------|------|--------|------|
| `--video_dir` | str | 是 | - | 视频文件所在的目录 |
| `--ref_dir` | str | 是 | - | 参考图像所在的目录 |
| `--flame_dir` | str | 是 | - | FLAME参数文件所在的目录 |
| `--output` | str | 是 | - | 输出的JSON文件路径 |
| `--text` | str | 否 | "视频中的人在做讲话" | 描述文本模板 |
| `--height` | int | 否 | 512 | 视频高度 |
| `--width` | int | 否 | 512 | 视频宽度 |
| `--pattern` | str | 否 | "*.mp4" | 视频文件匹配模式 |

## 输出格式

生成的JSON文件格式如下：

```json
[
    {
        "file_path": "path/to/videos/00000001.mp4",
        "control_file_path": "",
        "face_file_path": "",
        "ref_file_path": "path/to/references/00000001.png",
        "text": "视频中的人在做讲话",
        "type": "video",
        "height": 512,
        "width": 512
    },
    {
        "file_path": "path/to/videos/00000002.mp4",
        "control_file_path": "",
        "face_file_path": "",
        "ref_file_path": "path/to/references/00000002.png",
        "text": "视频中的人在做讲话",
        "type": "video",
        "height": 512,
        "width": 512
    }
]
```

## 注意事项

1. **文件名匹配**：脚本会根据视频文件的文件名（不带扩展名）来查找对应的参考图像和FLAME参数文件
2. **文件存在检查**：脚本会自动检查所有必需的文件是否存在，缺少文件的条目会被跳过
3. **错误处理**：如果目录不存在，脚本会报错并退出
4. **编码格式**：JSON文件使用UTF-8编码，支持中文

## 常见问题

### Q: 如何处理不同类型的视频格式？
A: 使用 `--pattern` 参数来指定不同的文件匹配模式，例如：
   - `--pattern "*.avi"` 用于AVI格式
   - `--pattern "*.mov"` 用于MOV格式

### Q: 如何自定义描述文本？
A: 使用 `--text` 参数来自定义描述文本，例如：
   - `--text "A person is speaking in the video"`
   - `--text "视频中的角色在说话"`

### Q: 如何批量处理多个数据集？
A: 可以使用循环来处理多个数据集，参考 `example_build_dataset.py` 中的示例。

### Q: 生成的JSON文件在哪里使用？
A: 生成的JSON文件可以用作训练脚本的数据配置文件，通过 `--train_data_meta` 参数指定。