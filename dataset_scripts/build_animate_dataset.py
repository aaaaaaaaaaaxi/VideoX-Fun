#!/usr/bin/env python3
"""
构建动画训练数据集的JSON文件
根据README_TRAIN_ANIMATE.md的格式要求，将视频、参考图像和FLAME参数文件映射为训练数据
"""

import os
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional
import argparse


def build_dataset_json(
    video_dir: str,
    ref_dir: str,
    flame_dir: str,
    output_file: str,
    text_template: str = "视频中的人在讲话",
    height: int = 512,
    width: int = 512,
    file_pattern: str = "*.mp4"
) -> int:
    """
    构建训练数据集的JSON文件

    Args:
        video_dir: 视频文件所在的目录
        ref_dir: 参考图像所在的目录
        flame_dir: FLAME参数文件所在的目录
        output_file: 输出的JSON文件路径
        text_template: 描述文本模板
        height: 视频高度
        width: 视频宽度
        file_pattern: 视频文件匹配模式

    Returns:
        生成的数据条目数量
    """
    # 确保所有目录都存在
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")

    if not os.path.exists(ref_dir):
        raise FileNotFoundError(f"参考图像目录不存在: {ref_dir}")

    if not os.path.exists(flame_dir):
        raise FileNotFoundError(f"FLAME参数目录不存在: {flame_dir}")

    # 查找所有视频文件
    video_pattern = os.path.join(video_dir, file_pattern)
    video_files = glob.glob(video_pattern)

    if not video_files:
        raise FileNotFoundError(f"在 {video_dir} 中没有找到匹配 {file_pattern} 的视频文件")

    dataset = []

    # 获取所有视频文件的文件名（不带扩展名）
    video_basenames = [Path(f).stem for f in video_files]

    for video_basename in video_basenames:
        # 构建各文件的完整路径
        video_path = os.path.join(video_dir, f"{video_basename}.mp4")
        ref_path = os.path.join(ref_dir, f"{video_basename}.png")
        flame_path = os.path.join(flame_dir, f"{video_basename}.pkl")

        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"警告: 视频文件不存在，跳过: {video_path}")
            continue

        if not os.path.exists(ref_path):
            print(f"警告: 参考图像文件不存在，跳过: {ref_path}")
            continue

        if not os.path.exists(flame_path):
            print(f"警告: FLAME参数文件不存在，跳过: {flame_path}")
            continue

        # 构建数据条目
        data_entry = {
            "file_path": video_path,
            "control_file_path": "",  # 设为空，因为不需要控制
            "face_file_path": "",     # 设为空，因为不需要面部文件
            "ref_file_path": ref_path,
            "text": text_template,
            "type": "video",
            "height": height,
            "width": width
        }

        dataset.append(data_entry)
        print(f"成功添加数据条目: {video_basename}")

    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n成功生成数据集文件: {output_file}")
    print(f"共包含 {len(dataset)} 个数据条目")

    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="构建动画训练数据集的JSON文件")
    parser.add_argument("--video_dir", type=str, required=True, help="视频文件目录")
    parser.add_argument("--ref_dir", type=str, required=True, help="参考图像目录")
    parser.add_argument("--flame_dir", type=str, required=True, help="FLAME参数文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出的JSON文件路径")
    parser.add_argument("--text", type=str, default="视频中的人在做讲话", help="描述文本")
    parser.add_argument("--height", type=int, default=512, help="视频高度")
    parser.add_argument("--width", type=int, default=512, help="视频宽度")
    parser.add_argument("--pattern", type=str, default="*.mp4", help="视频文件匹配模式")

    args = parser.parse_args()

    try:
        count = build_dataset_json(
            video_dir=args.video_dir,
            ref_dir=args.ref_dir,
            flame_dir=args.flame_dir,
            output_file=args.output,
            text_template=args.text,
            height=args.height,
            width=args.width,
            file_pattern=args.pattern
        )

        print(f"\n✅ 成功构建数据集，共 {count} 个条目")

    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())