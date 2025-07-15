import argparse
import json
import os
import datasets

from tqdm import tqdm
from transformers import pipeline
from Utils import config_util


# 获取数据需要经过NER处理的部分
def generate_item_input(item, input_keys):
    ner_content = ""
    for key in input_keys:
        ner_content += item[key] + " "
    return ner_content


# 提取 entity dict
def extract_entity_dict_from_dataset():
    ner_keys = ["PER", "LOC", "ORG"]
    entity_dict = {key: set() for key in ner_keys}
    for item in tqdm(data):
        if args.dataset == "wmt14-de-en":
            item = item["translation"]
        if ner_keys:
            ner_result = ner_pipe(generate_item_input(item, dataset_config.input_keys))
            for entity in ner_result:
                if (
                        entity["entity_group"] in ner_keys and
                        entity["score"] > args.entity_score_threshold and
                        len(entity["word"]) >= args.entity_min_len
                ):
                    if len(entity_dict[entity["entity_group"]]) < args.entity_group_max_len:
                        entity_dict[entity["entity_group"]].add(entity["word"])
                    else:
                        ner_keys.remove(entity["entity_group"])  # 不再新增该类型实体
        else:
            print("entity num is satisfied")
            break
    print({key: len(entity_dict[key]) for key in entity_dict})
    return entity_dict


# 持久化 entity_dict
def save_ner_dict(entity_dict):
    save_path = os.path.join("Output", "Entity_dict", args.dataset + ".json")
    serializable_dict = {key: list(value) for key, value in entity_dict.items()}
    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(serializable_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 参数配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="wikitext-103")
    parser.add_argument("--data_split", type=str, default="train", help="用于制作实体字典的数据split")
    parser.add_argument("--entity_score_threshold", default=0.9, type=float, help="有效实体置信度下限")
    parser.add_argument("--entity_group_max_len", default=10000, type=int, help="实体类别组实体数上限")
    parser.add_argument("--entity_min_len", default=2, type=int, help="单个实体长度下限")

    args = parser.parse_args()

    # 系统资源路径配置
    if os.name == 'nt':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
    if os.name == 'posix':
        args.dataset_base_path = "/data4/houshilong/Dataset"
        args.model_base_path = "/data4/houshilong/Model"

    dataset_path = os.path.join(args.dataset_base_path, args.dataset)
    model_path = os.path.join(args.model_base_path, "bert-large-cased-finetuned-conll03-english")

    # 加载数据
    data = datasets.load_from_disk(dataset_path)["train"]
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", args.dataset))

    # 加载 NER pipeline
    ner_pipe = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple",  # grouped_entities=True,
        device=0,
    )

    # 执行提取
    save_ner_dict(extract_entity_dict_from_dataset())
