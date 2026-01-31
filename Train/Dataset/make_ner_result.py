import argparse
import json
import os

import datasets
from tqdm import tqdm


def load_ner_pipeline(model_path):
    return pipeline(task="ner", model=model_path, aggregation_strategy="simple")


def detect_by_ner_pipe(ner_pipe, item_content):
    entity_types = ["PER", "ORG", "LOC"]
    entity_score_threshold = 0.9

    entity_list_map = {}

    ner_result = ner_pipe(item_content)
    for entity_item in ner_result:
        if entity_item["entity_group"] in entity_types and entity_item["score"] >= entity_score_threshold:
            entity_list_map[entity_item["word"]] = entity_item["entity_group"]

    return entity_list_map


def make_ner_result(ner_model_path, dataset, dataset_key, ner_result_file_path):
    ner_result = {}

    # 加载 NER Pipeline
    ner_pipe = load_ner_pipeline(ner_model_path)

    # 遍历处理数据集
    for ner_item in tqdm(dataset, desc="Processing NER"):
        ner_item_id = ner_item["id"]
        entity_list_map = detect_by_ner_pipe(ner_pipe, ner_item[dataset_key])
        ner_result[ner_item_id] = entity_list_map

    # 写入文件
    os.makedirs(os.path.dirname(ner_result_file_path), exist_ok=True)
    with open(ner_result_file_path, "w", encoding="utf-8") as f:
        json.dump(ner_result, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--dataset_key", type=str, default="document")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_size", type=int, default=5000)
    parser.add_argument(
        "--ner_model", type=str, default="bert-large-cased-finetuned-conll03-english"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    args.resource_path = "" if os.name == "nt" else "/longzi05/houshilong/"

    # 加载数据集
    dataset_path = os.path.join(args.resource_path, "Dataset", args.dataset)
    dataset = datasets.load_from_disk(dataset_path)[args.data_split].select(
        range(args.data_start, args.data_size)
    )

    # 生成或读取ner结果文件
    ner_result_file_path = os.path.join(
        args.dpo_dataset_path,
        f"{args.dataset}_{args.data_split}_({args.data_start},{args.data_size})_ner_result.json",
    )
    ner_model_path = os.path.join(args.resource_path, "Model", args.ner_model)

    make_ner_result(ner_model_path, dataset, args.dataset_key, ner_result_file_path)
