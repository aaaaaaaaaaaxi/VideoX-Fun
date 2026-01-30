#!/usr/bin/env python
"""
批量 Wan2.2 Animate LoRA 推理脚本
=====================================

用于批量生成多个视频，支持从配置文件读取参数。
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from inference_animate_lora_advanced import Wan2_2AnimateAdvancedInference, GenerationConfig


logger = logging.getLogger(__name__)


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载配置列表。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    if isinstance(config_data, dict):
        if 'prompts' in config_data:
            # 格式1: {"prompts": [...], "common": {...}}
            common = config_data.get('common', {})
            prompts = config_data['prompts']
            # 合并通用配置
            for prompt_config in prompts:
                prompt_config.update(common)
            return prompts
        else:
            # 格式2: 单个配置
            return [config_data]
    elif isinstance(config_data, list):
        # 格式3: 配置列表
        return config_data
    else:
        raise ValueError("配置文件格式错误")


def save_config_to_json(configs: List[Dict[str, Any]], output_path: str):
    """将配置保存为JSON文件。"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)


def create_sample_config(output_path: str = "sample_config.json"):
    """创建示例配置文件。"""
    sample_configs = [
        {
            "name": "nature_video",
            "prompt": "A beautiful waterfall in a lush green forest, sunlight filtering through trees",
            "negative_prompt": "blurry, low quality, bad art, distorted",
            "num_frames": 49,
            "height": 480,
            "width": 720,
            "guidance_scale": 7.5,
            "seed": 42
        },
        {
            "name": "cityscape",
            "prompt": "Futuristic cityscape at night, neon lights, flying cars",
            "negative_prompt": "poor quality, blurry, cartoonish",
            "num_frames": 81,
            "height": 512,
            "width": 512,
            "guidance_scale": 8.0,
            "seed": 123
        },
        {
            "name": "animal_animation",
            "prompt": "A cute cat playing with a ball of yarn, smooth animation",
            "negative_prompt": "static, blurry, deformed",
            "num_frames": 25,
            "height": 384,
            "width": 640,
            "guidance_scale": 6.5,
            "seed": 456
        }
    ]

    save_config_to_json(sample_configs, output_path)
    print(f"示例配置已保存到: {output_path}")


def batch_inference(
    base_model_path: str,
    lora_path: str,
    config_file: str,
    output_dir: str = "./batch_outputs",
    max_workers: int = 1,
    device: str = "cuda",
    dtype: str = "float16"
):
    """批量推理。"""
    # 加载配置
    configs = load_config(config_file)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建成功和失败记录
    successful_generations = []
    failed_generations = []

    logger.info(f"开始批量生成 {len(configs)} 个视频...")

    # 为每个配置生成视频
    for config in tqdm(configs, desc="生成进度"):
        try:
            # 跳过没有prompt的配置
            if 'prompt' not in config:
                logger.warning(f"配置 {config.get('name', 'unknown')} 缺少prompt，跳过")
                failed_generations.append({
                    'config': config,
                    'error': "Missing prompt in config"
                })
                continue

            # 创建生成配置
            gen_config = GenerationConfig(
                base_model_path=base_model_path,
                lora_path=lora_path,
                output_dir=output_dir,
                prompt=config['prompt'],
                negative_prompt=config.get('negative_prompt', ''),
                height=config.get('height', 480),
                width=config.get('width', 720),
                num_frames=config.get('num_frames', 49),
                fps=config.get('fps', 8),
                num_inference_steps=config.get('num_inference_steps', 50),
                guidance_scale=config.get('guidance_scale', 7.0),
                eta=config.get('eta', 0.0),
                num_videos_per_prompt=config.get('num_videos_per_prompt', 1),
                device=device,
                dtype=dtype,
                scheduler=config.get('scheduler', 'euler'),
                seed=config.get('seed'),
                mode=config.get('mode', 'text_to_video'),
                control_type=config.get('control_type'),
                control_image=config.get('control_image'),
                start_image=config.get('start_image'),
                reference_image=config.get('reference_image')
            )

            # 生成视频
            generator = Wan2_2AnimateAdvancedInference(gen_config)
            output_path = generator.generate()

            # 记录成功
            result = {
                'name': config.get('name', 'unnamed'),
                'prompt': config['prompt'],
                'output_path': output_path,
                'config': {k: v for k, v in config.items() if k != 'prompt'}
            }
            successful_generations.append(result)

            logger.info(f"✅ {config.get('name', 'unknown')} 生成成功: {output_path}")

        except Exception as e:
            logger.error(f"❌ {config.get('name', 'unknown')} 生成失败: {str(e)}")
            failed_generations.append({
                'name': config.get('name', 'unknown'),
                'prompt': config.get('prompt', ''),
                'error': str(e),
                'config': config
            })

    # 保存结果
    results = {
        'successful': successful_generations,
        'failed': failed_generations,
        'summary': {
            'total': len(configs),
            'successful': len(successful_generations),
            'failed': len(failed_generations),
            'success_rate': len(successful_generations) / len(configs) if configs else 0
        }
    }

    results_path = os.path.join(output_dir, 'batch_results.json')
    save_config_to_json([results], results_path)

    # 打印摘要
    print(f"\n📊 批量生成摘要:")
    print(f"   总数: {results['summary']['total']}")
    print(f"   成功: {results['summary']['successful']}")
    print(f"   失败: {results['summary']['failed']}")
    print(f"   成功率: {results['summary']['success_rate']:.2%}")

    if failed_generations:
        print(f"\n❌ 失败的配置:")
        for failure in failed_generations:
            print(f"   - {failure['name']}: {failure['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="批量 Wan2.2 Animate LoRA 推理")

    # 基础参数
    parser.add_argument("--base_model_path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 权重路径")

    # 配置参数
    parser.add_argument("--config_file", type=str, help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="./batch_outputs", help="输出目录")

    # 生成参数
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="数据类型")
    parser.add_argument("--max_workers", type=int, default=1, help="最大并行数")

    # 工具选项
    parser.add_argument("--create_sample_config", action="store_true", help="创建示例配置文件")
    parser.add_argument("--sample_config_path", type=str, default="sample_config.json", help="示例配置路径")

    args = parser.parse_args()

    if args.create_sample_config:
        create_sample_config(args.sample_config_path)
        return

    if not args.config_file:
        print("错误: 必须提供配置文件路径，或使用 --create_sample_config 创建示例配置")
        return

    # 运行批量推理
    results = batch_inference(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path,
        config_file=args.config_file,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        device=args.device,
        dtype=args.dtype
    )

    print(f"\n🎉 批量生成完成! 结果保存在: {os.path.join(args.output_dir, 'batch_results.json')}")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()