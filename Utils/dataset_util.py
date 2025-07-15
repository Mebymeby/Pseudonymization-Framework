import json
import os

import datasets

from Config import prompt


# 加载: 数据集指定split
## 1. 数据集合法性判断
## 2. 数据集格式重整
## 3. 数据集子集截取
## 4. 有效数据量计算
def load_dataset(args, dataset_config):
    dataset_name = args.dataset  # 数据集名称
    dataset_path = os.path.join(args.dataset_base_path, dataset_name)  # 数据集路径
    dataset_full = datasets.load_from_disk(dataset_path)  # 加载完整数据集
    dataset_split = dataset_full[args.data_split]  # 加载所需split

    # 对指定数据做些处理
    if dataset_name == "wikitext-103":
        dataset = dataset_split.filter(lambda item: len(item["text"].split()) > 100)
    elif dataset_config.name == "wmt14-de-en":
        dataset = dataset_split.map(lambda item: item["translation"], remove_columns="translation")
    else:
        dataset = dataset_split

    dataset_full_len = len(dataset)  # 计算有效数据集长度
    if args.data_start > dataset_full_len:
        print("WARNING: data_start > data_len")
        exit(0)

    data_end = args.data_start + args.data_size
    if args.data_size <= 0 or data_end > dataset_full_len:
        data_end = dataset_full_len

    dataset = dataset.select(range(args.data_start, data_end))  # 根据参数截取数据
    dataset_len = len(dataset)

    return dataset, dataset_len


# 获取:
## 1. 数据集 item 用于执行 detection 的输入部分
## 2. 数据集 item 嵌入 prompt 的输入部分
def get_item_input_content_list(item, dataset_config):
    input_content_list = []
    for key in dataset_config.input_keys:
        if dataset_config.name == "xsum":
            value_split = item[key].split()
            if len(value_split) > 2048:
                value_split = value_split[:2048]
            input_content_list.append(" ".join(value_split))
            item[key] = " ".join(value_split)  # 裁切数据集
        elif dataset_config.name == "wikitext-103":
            value_split = item[key].split()[:50]
            input_content_list.append(" ".join(value_split))
        elif dataset_config.name == "cnn_dailymail":
            input_len = int(len(item[key].split()) * 0.3)
            value_split = item[key].split()[:input_len]
            input_content_list.append(" ".join(value_split))
        else:
            input_content_list.append(item[key])

    return input_content_list


# 获取: 数据集 llm_prompt 推理时的 infer_prompt
def generate_dataset_infer_prompt(item, dataset_config):
    item_input_content_list = get_item_input_content_list(item, dataset_config)
    dataset_infer_prompt = prompt.DATASET[dataset_config.name].format(*item_input_content_list)
    return dataset_infer_prompt


# 加载: 数据集预抽取的实体字典
def load_entity_dict(args):
    entity_dict_path = os.path.join("Output", "Entity_dict", "merged.json")
    with open(entity_dict_path, "r", encoding="utf-8") as file:
        entity_dict = json.load(file)
    return entity_dict
