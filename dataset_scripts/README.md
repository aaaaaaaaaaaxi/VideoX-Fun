# 数据集构建工具

这个目录包含用于构建动画训练数据集的工具脚本。

## 文件说明

- `build_animate_dataset.py` - 主要的数据集构建脚本
- `example_build_dataset.py` - 使用示例脚本
- `README_BUILD_DATASET.md` - 详细的使用说明文档

## 快速开始

```bash
# 基本用法
python build_animate_dataset.py \
    --video_dir path/to/videos \
    --ref_dir path/to/references \
    --flame_dir path/to/flame_params \
    --output dataset.json
```

## 详细文档

请参考 `README_BUILD_DATASET.md` 获取详细的使用说明。