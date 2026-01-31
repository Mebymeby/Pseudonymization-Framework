"""
1、读取模型在不同种子、不同批次情况下生成的结果文件
2、调用评估方法为数据打分
    2.1 NED质量：隐私实体检出率与错检率（以 NER 模型的结果为准）
    2.2 ERM质量：替换实体重复率、和隐私实体的文本相似度惩罚
    2.3 ATG质量：隐私实体替换准确度、语句通顺度
3、打分后的结果写入 rank 文件
"""

import argparse
import json
import math
import os.path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_eval_model(model_path):
    """加载评估模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return tokenizer, model


def ned_eval(ner_dict: dict, ner_reference: dict) -> dict:
    """
    NED评估：计算实体检测的精确率、召回率和F1分数

    Args:
        ner_dict: 模型检测到的实体字典 {entity_word: entity_type}
        ner_reference: 参考实体字典 {entity_word: entity_type}

    Returns:
        包含评估指标的字典
    """
    TP = 0  # 检出正确实体
    FP = 0  # 检出错误实体
    FN = 0  # 漏检正确实体

    for entity_word, entity_type in ner_dict.items():
        if entity_word in ner_reference:
            if entity_type == ner_reference[entity_word]:
                TP += 1
            else:
                TP += 0.5  # 类型错误算 0.5 分
        else:
            FP += 1

    for entity in ner_reference.keys():
        if entity not in ner_dict:
            FN += 1

    # 计算 F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def erm_eval(ner_dict: dict, erm_dict: dict) -> dict:
    """
    ERM评估：计算实体替换的质量

    Args:
        ner_dict: NER检测结果 {entity_word: entity_type}
        erm_dict: 实体替换结果 {original_entity: pseudo_entity}

    Returns:
        包含评估指标的字典
    """
    entity_num = len(erm_dict.keys())

    # 以 erm_dict 作为预测，ner_dict 作为参考，计算F1，体现前两阶段的一致性
    TP = 0  # 处理正确实体
    FP = 0  # 处理错误实体
    FN = 0  # 遗漏处理实体
    for entity_word, entity_type in ner_dict.items():
        if entity_word in erm_dict:
            TP += 1
        else:
            FP += 1

    for entity in erm_dict.keys():
        if entity not in ner_dict:
            FN += 1

    # 计算 F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # 原实体和假名化实体重叠惩罚
    overlap = 0
    for ori_entity, pseu_entity in erm_dict.items():
        if ori_entity in pseu_entity or pseu_entity in ori_entity:
            overlap += 1

    # 计算假名化实体的有效率（避免假名化实体与原实体重叠）
    valid_rate = 1 - (overlap / entity_num) if entity_num > 0 else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "valid_rate": valid_rate,
    }


@torch.inference_mode()
def atg_eval(erm_dict: dict, atg_seq: str, eval_tokenizer, eval_model) -> dict:
    """
    ATG评估：计算文本生成的质量

    Args:
        erm_dict: 实体替换结果 {original_entity: pseudo_entity}
        atg_seq: 生成的文本序列
        eval_tokenizer: 评估模型的tokenizer
        eval_model: 评估模型

    Returns:
        包含评估指标的字典
    """
    # 以 atg 序列为预测，erm_dict 为参考，计算替换准确率
    TP = 0  # 正确替换
    FN = 0  # 遗漏替换

    for ori_entity, pseu_entity in erm_dict.items():
        if pseu_entity in atg_seq and ori_entity not in atg_seq:
            TP += 1
        else:
            FN += 1

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    inputs = eval_tokenizer(atg_seq, return_tensors="pt").to(eval_model.device)
    outputs = eval_model(**inputs, labels=inputs["input_ids"])
    perplexity = math.exp(outputs.loss.item())

    return {"recall": recall, "perplexity": perplexity}


def calc_rank_score(ned_score, erm_score, atg_score) -> float:
    """计算加权分数"""
    rank_score = (
            2 * ned_score["f1"]
            + erm_score["f1"]
            + erm_score["valid_rate"]
            + atg_score["recall"]
            - 0.05 * atg_score["perplexity"]
    )
    return rank_score


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
    parser.add_argument("--eval_model", type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--dpo_dataset_path", type=str, default=os.path.join("Train", "Dataset")
    )
    parser.add_argument("--e2e_result_file_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    args.resource_path = "" if os.name == "nt" else "/longzi05/houshilong/"

    # 读取rank文件，如果不存在则创建空字典
    rank_file = os.path.join(
        args.dpo_dataset_path, f"{args.dataset}_{args.data_split}_rank.json"
    )
    rank_data = {}
    if os.path.exists(rank_file):
        with open(rank_file, "r", encoding="utf-8") as f:
            rank_data = json.load(f)

    # 读取 ner 结果文件
    ner_result_file_path = os.path.join(
        args.dpo_dataset_path,
        f"{args.dataset}_{args.data_split}_({args.data_start},{args.data_size})_ner_result.json",
    )
    with open(ner_result_file_path, "r", encoding="utf-8") as f:
        ner_result_data = json.load(f)

    # 读取端到端推理结果文件
    with open(args.e2e_result_file_path, "r", encoding="utf-8") as f:
        e2e_result = json.load(f)

    # 解析模型名和种子
    model_name = args.e2e_result_file_path.split("_")[0]
    seed = args.e2e_result_file_path.split("/")[0].split("_")[-1]

    # 加载 eval 模型
    eval_model_path = os.path.join(args.resource_path, "Model", args.eval_model)
    eval_tokenizer, eval_model = load_eval_model(eval_model_path)
    eval_model.eval()

    # 为每个 item 打分
    for item in tqdm(e2e_result, desc="Processing DPO data"):
        item_id = item["key"]

        # 调用评估方法为数据打分
        ned_score = ned_eval(item["NED"], ner_result_data[item_id])
        erm_score = erm_eval(item["NED"], item["ERM"])
        atg_score = atg_eval(item["ERM"], item["ATG"], eval_tokenizer, eval_model)

        # 计算加权分数
        dpo_rank_score = calc_rank_score(ned_score, erm_score, atg_score)

        # 将打分结果写入 item 中
        item["model"] = model_name
        item["seed"] = seed
        item["ned_score"] = ned_score
        item["erm_score"] = erm_score
        item["atg_score"] = atg_score
        item["dpo_rank_score"] = dpo_rank_score

        # 如果 rank_data 没有保存当前 item，直接保存
        if item_id not in rank_data:
            rank_data[item_id] = [item]
            continue  # 只有一个样本无法制作 dpo 数据
        else:
            # 保存 item 到 rank 文件，根据 dpo_rank_score 排序，保留排名前 5 供查看
            rank_data[item_id].append(item)
            cur_item_rank_list = sorted(
                rank_data[item_id], key=lambda x: x["dpo_rank_score"], reverse=True
            )
            rank_data[item_id] = cur_item_rank_list[:5]

    # 保存 rank 数据
    os.makedirs(os.path.dirname(rank_file), exist_ok=True)
    with open(rank_file, "w", encoding="utf-8") as f:
        json.dump(rank_data, f, ensure_ascii=False, indent=4)
