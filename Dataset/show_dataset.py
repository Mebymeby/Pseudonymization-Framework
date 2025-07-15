import json
import os

import datasets
import numpy as np


def show_text_stats(data_column):
    # 计算每个文本的长度
    text_lengths = [len(text.split()) for text in data_column]

    # 计算均值、最大最小值等统计信息
    mean_length = np.mean(text_lengths)
    min_length = np.min(text_lengths)
    max_length = np.max(text_lengths)

    # 计算四分位点
    q1 = np.percentile(text_lengths, 25)  # 第一四分位数
    q2 = np.percentile(text_lengths, 50)  # 第二四分位数（中位数）
    q3 = np.percentile(text_lengths, 75)  # 第三四分位数
    q95 = np.percentile(text_lengths, 95)  # 95% 最大值
    q99 = np.percentile(text_lengths, 99)  # 99% 最大值

    status = {
        'mean': float(mean_length),
        'min': float(min_length),
        'max': float(max_length),
        'q1': float(q1),
        'q2': float(q2),
        'q3': float(q3),
        'q95': float(q95),
        'q99': float(q99),
    }

    print(json.dumps(status, indent=4, ensure_ascii=False))


def show_selected_items(idx_start, idx_end):
    selected_items = dataset_split.select(range(idx_start, idx_end))
    for idx, item in enumerate(selected_items):
        print(json.dumps(item, indent=4, ensure_ascii=False))
        print(len(item[dataset_key].split()))


def extract_text(item):
    return item["translation"]


if __name__ == "__main__":

    dataset_name = "xsum"
    dataset_split = "test"
    dataset_key = "document"

    if os.name == 'nt':
        dataset_path = os.path.join("Dataset", dataset_name)
    else:
        dataset_path = os.path.join("/data4/houshilong/Dataset", dataset_name)

    dataset_full = datasets.load_from_disk(dataset_path)
    print(f"Full: {dataset_full.shape}")

    dataset_split = dataset_full[dataset_split]
    print(f"Split: {dataset_split.shape}")

    if dataset_name == "wmt14-de-en":
        dataset_split = dataset_split.map(extract_text, remove_columns="translation")

    show_selected_items(16, 17)

    show_text_stats(dataset_split[dataset_key])
