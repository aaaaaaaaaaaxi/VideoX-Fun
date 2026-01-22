#!/usr/bin/env python3
"""
构建动画训练数据集的使用示例
"""

from build_animate_dataset import build_dataset_json
import os


def example_usage():
    """
    演示如何使用 build_dataset_json 函数
    """

    # 示例目录结构（请根据您的实际路径修改）
    # 假设您的目录结构如下：
    # dataset/
    # ├── videos/          # 存放视频文件
    # ├── references/      # 存放参考图像（PNG格式）
    # ├── flame_params/    # 存放FLAME参数（PKL格式）
    # └── dataset.json    # 生成的数据集文件

    video_dir = "path/to/your/videos"
    ref_dir = "path/to/your/references"
    flame_dir = "path/to/your/flame_params"
    output_file = "path/to/your/dataset.json"

    # 方式1：使用函数调用
    try:
        count = build_dataset_json(
            video_dir=video_dir,
            ref_dir=ref_dir,
            flame_dir=flame_dir,
            output_file=output_file,
            text_template="视频中的人在做讲话",
            height=512,
            width=512
        )
        print(f"成功生成 {count} 个数据条目")

    except Exception as e:
        print(f"生成数据集时出错: {e}")

    # 方式2：使用命令行（推荐）
    print("\n命令行使用方法:")
    print("python build_animate_dataset.py \\")
    print("    --video_dir path/to/your/videos \\")
    print("    --ref_dir path/to/your/references \\")
    print("    --flame_dir path/to/your/flame_params \\")
    print("    --output dataset.json \\")
    print("    --text \"视频中的人在做讲话\" \\")
    print("    --height 512 \\")
    print("    --width 512")


def batch_process_example():
    """
    批量处理多个数据集的示例
    """
    datasets = [
        {
            "name": "train_dataset",
            "video_dir": "data/train/videos",
            "ref_dir": "data/train/references",
            "flame_dir": "data/train/flame_params",
            "output": "data/train/dataset.json"
        },
        {
            "name": "val_dataset",
            "video_dir": "data/val/videos",
            "ref_dir": "data/val/references",
            "flame_dir": "data/val/flame_params",
            "output": "data/val/dataset.json"
        }
    ]

    for dataset in datasets:
        print(f"\n正在处理 {dataset['name']}...")
        try:
            count = build_dataset_json(
                video_dir=dataset["video_dir"],
                ref_dir=dataset["ref_dir"],
                flame_dir=dataset["flame_dir"],
                output_file=dataset["output"]
            )
            print(f"✅ {dataset['name']} 完成，共 {count} 个条目")
        except Exception as e:
            print(f"❌ {dataset['name']} 失败: {e}")


if __name__ == "__main__":
    print("=== 构建动画训练数据集示例 ===")
    example_usage()

    print("\n=== 批量处理示例 ===")
    batch_process_example()