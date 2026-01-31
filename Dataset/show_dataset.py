import argparse
import json
import os
from typing import List, Optional

import datasets
import numpy as np


def show_text_stats(data_column: List[str]) -> None:
    """
    显示文本列的统计信息，包括长度的均值、极值和分位数

    Args:
        data_column: 包含文本的列表
    """
    # 计算每个文本的长度
    text_lengths = [len(text.split()) for text in data_column]

    # 计算统计信息
    stats = {
        'mean': float(np.mean(text_lengths)),
        'min': float(np.min(text_lengths)),
        'max': float(np.max(text_lengths)),
        'q1': float(np.percentile(text_lengths, 25)),  # 第一四分位数
        'q2': float(np.percentile(text_lengths, 50)),  # 中位数
        'q3': float(np.percentile(text_lengths, 75)),  # 第三四分位数
        'q95': float(np.percentile(text_lengths, 95)),  # 95% 分位数
        'q99': float(np.percentile(text_lengths, 99)),  # 99% 分位数
    }

    print("Text Statistics:")
    print(json.dumps(stats, indent=4, ensure_ascii=False))


def show_selected_items(
        dataset_split: datasets.Dataset,
        idx_start: int,
        idx_end: int,
        target_ids: Optional[List[str]] = None,
        data_key: Optional[str] = None
) -> None:
    """
    显示数据集中选定范围或指定ID的项目

    Args:
        dataset_split: 数据集分割
        idx_start: 起始索引
        idx_end: 结束索引
        target_ids: 要筛选的ID列表（可选）
        data_key: 用于显示长度的字段名（可选）
    """
    # 如果提供了ID列表，进行筛选
    selected_items = dataset_split

    print(f"target ids: {target_ids}")

    # 显示选定项目
    for idx, item in enumerate(selected_items):
        print(f"\nItem {idx + idx_start}:")
        print(json.dumps(item, indent=4, ensure_ascii=False))

        # 如果提供了数据键，显示文本长度
        # if data_key and data_key in item:
        #     print(f"Text Length (words): {len(item[data_key].split())}")


def extract_text(item: dict) -> dict:
    """
    从翻译字段中提取文本（用于特定数据集如wmt14-de-en）

    Args:
        item: 数据集中的项目

    Returns:
        提取文本后的项目
    """
    return item["translation"]


def get_dataset_path(dataset_name: str) -> str:
    """
    根据当前操作系统获取数据集的路径

    Args:
        dataset_name: 数据集名称

    Returns:
        数据集的完整路径
    """
    if os.name == "nt":  # Windows系统
        return os.path.join("Dataset", dataset_name)
    else:  # Linux系统
        return os.path.join("/longzi05/houshilong/Dataset", dataset_name)


def main():
    """
    主函数，用于解析命令行参数并执行数据集查看操作
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="View and analyze dataset information")
    parser.add_argument("--dataset", "-d", type=str, default="glue-mnli", help="Dataset name")
    parser.add_argument("--split", "-s", type=str, default="validation_matched", help="Dataset split (train/validation/test)")
    parser.add_argument("--key", "-k", type=list, default=["premise", "hypothesis"], help="Data column key to analyze")
    parser.add_argument("--start", type=int, default=14, help="Start index for analysis")
    parser.add_argument("--end", type=int, default=15, help="End index for analysis")
    parser.add_argument("--ids", type=str, nargs="*", default=["116"], help="Target IDs to filter")
    parser.add_argument("--add-idx", type=int, default=1, help="Add index column to dataset")

    args = parser.parse_args()

    try:
        # 获取数据集路径
        dataset_path = get_dataset_path(args.dataset)

        # 加载数据集
        print(f"Loading dataset from {dataset_path}...")
        dataset_full = datasets.load_from_disk(dataset_path)
        print(f"Full dataset shape: {dataset_full.shape}")

        # 获取指定分割
        if args.split not in dataset_full:
            raise ValueError(f"Split '{args.split}' not found in dataset")

        dataset_split = dataset_full[args.split]
        print(f"Split '{args.split}' shape: {dataset_split.shape}")

        # 验证索引范围
        if args.end > len(dataset_split):
            print(
                f"Warning: End index {args.end} exceeds dataset length {len(dataset_split)}, adjusting to {len(dataset_split)}")
            args.end = len(dataset_split)

        if args.start < 0 or args.start >= args.end:
            raise ValueError(f"Invalid index range: start={args.start}, end={args.end}")

        # 范围截取
        dataset_split = dataset_split.select(range(args.start, args.end))

        # 添加索引列（如果需要）
        if args.add_idx:
            dataset_split = dataset_split.add_column("id", list(range(args.start, args.end)))
        print("add id column success")

        # 对特定数据集进行特殊处理
        if args.dataset == "wmt14-de-en":
            dataset_split = dataset_split.map(extract_text, remove_columns="translation")

        # 显示选定项目
        if args.ids or args.end > args.start:
            show_selected_items(dataset_split, args.start, args.end, args.ids, args.key)

        # 显示文本统计信息
        # if args.key in dataset_split.features:
        #     show_text_stats(dataset_split[args.key])
        # else:
        #     print(f"Warning: Column '{args.key}' not found in dataset")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
