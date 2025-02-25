import json
import os

import datasets

from Config import prompt


def load_dataset(args, dataset_config):
    data = datasets.load_from_disk(os.path.join(args.dataset_base_path, dataset_config.path))
    data = data[args.data_split]
    data_len = len(data)

    if args.data_start > data_len:
        return None

    args.data_end = args.data_start + args.data_size

    if args.data_size <= 0 or args.data_end > data_len:
        args.data_end = data_len

    data = data.select(range(args.data_start, args.data_end))  # 根据参数截取数据

    return data


def load_entity_dict(dataset_config):
    entity_dict_path = os.path.join("Output", "Entity_dict", dataset_config.name + ".json")
    with open(entity_dict_path, "r", encoding="utf-8") as file:
        entity_dict = json.load(file)
    return entity_dict


def generate_task_input(item, input_keys):
    task_input = ""
    for key in input_keys:
        task_input += ("\n" + key + ": " + item[key])

    if len(task_input) > 8192:
        task_input = task_input[:8192]  # 截断过长的输入，防止炸显存

    return task_input


def generate_dataset_prompt(item, dataset_config):
    return f"""{prompt.DATASET[dataset_config.name]}
{generate_task_input(item, dataset_config.input_keys)}
"""
