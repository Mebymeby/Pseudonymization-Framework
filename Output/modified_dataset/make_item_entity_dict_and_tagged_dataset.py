import argparse
import json
import os
import re

import datasets
from transformers import pipeline

from Utils import config_util


def process_item(item, process_type):
    item[modify_key + "_ori"] = item[modify_key]  # 保留原始数据，方便对照

    item_entity_map = {key: [] for key in ner_keys}

    item_ner_result = ner_pipe(item[modify_key])
    item_ner_result.sort(key=lambda x: len(x["word"]), reverse=True)  # 根据实体文本长度降序排列，方便后面替换

    for entity in item_ner_result:
        if (
                entity["entity_group"] in ner_keys and
                entity["score"] > args.entity_score_threshold and
                len(entity["word"]) >= args.entity_min_len and
                entity["word"] not in item_entity_map[entity["entity_group"]]  # 去重
        ):
            item_entity_map[entity["entity_group"]].append(entity["word"])

            if process_type == "only_rep":
                item[modify_key] = item[modify_key].replace(entity["word"], "<" + entity["entity_group"] + ">")
            if process_type == "tag_wrap":
                item[modify_key] = item[modify_key].replace(entity["word"], "<" + entity["entity_group"] + ">"
                                                            + entity["word"] + "</" + entity["entity_group"] + ">")

    # 正则表达式处理嵌套标签问题
    if process_type == "tag_wrap":
        patterns = ["<" + ner_key + ">(.*?)<" + ner_key + ">(.*?)</" + ner_key + ">(.*?)</" + ner_key + ">"
                    for ner_key in ner_keys]
        for pattern, ner_key in zip(patterns, ner_keys):
            while re.search(pattern, item[modify_key]):
                item[modify_key] = re.sub(pattern, "<" + ner_key + r">\1\2\3</" + ner_key + ">", item[modify_key])

    # 只记录一次实体字典文件
    if process_type == "only_rep":
        line = json.dumps(item_entity_map, ensure_ascii=False)
        with open(item_entity_dict_path, 'a', encoding='utf-8') as file:
            file.write(line + '\n')

    return item


if __name__ == '__main__':
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--data_size", default=-1, type=int, help="选部分数据进行测试")
    parser.add_argument("--entity_score_threshold", default=0.8, type=float, help="有效实体置信度下限")
    parser.add_argument("--entity_min_len", default=2, type=int, help="单个实体长度下限，排除单个字母的情况")
    args = parser.parse_args()

    # 根据操作系统配置资源路径
    if os.name == 'nt':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
    if os.name == 'posix':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"

    # 加载数据
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", args.dataset))
    data = datasets.load_from_disk(os.path.join(args.dataset_base_path, dataset_config.name))["train"]
    if args.data_size > 0:
        data = data.select(range(args.data_size))

    modify_key = dataset_config.input_keys[0]  # 数据集的待操作属性
    ner_keys = ["PER", "LOC", "ORG"]  # ner 类型列表

    # 加载 NER pipeline
    model_path = os.path.join(args.model_base_path, "bert-large-cased-finetuned-conll03-english")
    ner_pipe = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",  # grouped_entities=True,
        device=0,
    )

    # 处理后数据的存放路径
    item_entity_dict_path = os.path.join("Output", "modified_dataset",
                                         args.dataset + str(args.data_size) if args.data_size > 0 else "",
                                         "item_entity_dict.txt")
    modified_dataset_path = os.path.join("Output", "modified_dataset",
                                         args.dataset + str(args.data_size) if args.data_size > 0 else "")
    os.makedirs(modified_dataset_path, exist_ok=True)

    # 执行处理
    only_rep_dataset = data.map(process_item, fn_kwargs={"process_type": "only_rep"})
    tag_wrap_dataset = data.map(process_item, fn_kwargs={"process_type": "tag_wrap"})

    # 保存处理后的数据集
    only_rep_dataset.save_to_disk(os.path.join(modified_dataset_path, "only_rep"))
    tag_wrap_dataset.save_to_disk(os.path.join(modified_dataset_path, "tag_wrap"))
