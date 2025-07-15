import argparse
import json
import math
import os.path

import datasets
import numpy as np
import torch
from sentence_transformers import util, SentenceTransformer
from tqdm import tqdm

from Utils import config_util, model_util


def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]


# 计算整个句子的困惑度
def calculate_perplexity(sentence):
    if sentence == "":
        return 100
    inputs = eval_model_tokenizer(sentence, return_tensors="pt").to(eval_model.device)
    outputs = eval_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())


# 计算两个文本的 "不相似度"
def calculate_pseu_efficiency(item, pseu_item):
    ori_seq_list, pseu_seq_list = [], []

    for key in dataset_config.input_keys:
        ori_seq_list.append(item[key])
        pseu_seq_list.append(next(iter(pseu_item.values())))

    ori_seq = " ".join(ori_seq_list)
    pseu_seq = " ".join(pseu_seq_list)

    emb1 = embedding_model.encode(ori_seq, convert_to_tensor=True)
    emb2 = embedding_model.encode(pseu_seq, convert_to_tensor=True)

    sim = util.pytorch_cos_sim(emb1, emb2).squeeze(0).item()

    return 1.0 - sim


def calculate_pseu_leak(consensus_entities_list, pseu_item):
    # 计算 原实体的出现过 & 候选实体未出现 的实体比例
    entity_list_len = len(consensus_entities_list)
    if entity_list_len == 0:
        return 0

    leak_count = 0
    for ori_entity in consensus_entities_list:
        cur_item_input = next(iter(pseu_item.values()))
        if ori_entity in cur_item_input:
            leak_count += 1

    return leak_count / entity_list_len


def calculate_metrics():
    replace_metric_scores = {replace_method: {
        "Pse": 0.0,  # 假名化效率
        "Psp": 0.0,  # 语句困惑度
        "Psl": 0.0,  # 隐私泄露率
    } for replace_method in args.replace_method}

    pse_score_list_map = {replace_method: [] for replace_method in args.replace_method}
    plx_score_list_map = {replace_method: [] for replace_method in args.replace_method}
    psl_score_list_map = {replace_method: [] for replace_method in args.replace_method}

    for idx, item in enumerate(tqdm(dataset)):
        for i, replace_cache_file in enumerate(replace_cache_files):
            cur_pseu_item = replace_cache_file[idx]
            sim = calculate_pseu_efficiency(item, cur_pseu_item)
            pse_score_list_map[args.replace_method[i]].append(sim)
            plx = calculate_perplexity(next(iter(cur_pseu_item.values())))
            plx_score_list_map[args.replace_method[i]].append(plx)
            psl = calculate_pseu_leak(consensus_entity[idx]["consensus"], cur_pseu_item)
            psl_score_list_map[args.replace_method[i]].append(psl)

    pseu_efficiency_scores = {key: np.mean(val) for key, val in pse_score_list_map.items()}
    perplexity_scores = {key: np.mean(val) for key, val in plx_score_list_map.items()}
    pseu_leak_scores = {key: np.mean(val) for key, val in psl_score_list_map.items()}

    for replace_method, pps in pseu_efficiency_scores.items():
        replace_metric_scores[replace_method]["Pse"] = float(pps)
    for replace_method, plx in perplexity_scores.items():
        replace_metric_scores[replace_method]["Psp"] = float(plx)
    for replace_method, psl in pseu_leak_scores.items():
        replace_metric_scores[replace_method]["Psl"] = float(psl)

    return replace_metric_scores


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--eval_model", type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--replace_method", type=str, nargs="+", default=["ori_model", "ft_model"])
    parser.add_argument("--data_size", type=int, default=5000)
    parser.add_argument("--exp_lambda", type=int, default=100)
    args = parser.parse_args()
    args.data_start = 0

    # 资源路径
    args.resource_base_path = "" if os.name == "nt" else os.path.join("/data4", "houshilong")
    args.dataset_path = os.path.join(args.resource_base_path, "Dataset", args.dataset)

    # 数据集配置
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", args.dataset))  # 数据集相关配置

    # 加载 embedding 模型
    embedding_model_path = os.path.join(args.resource_base_path, "Model", args.embedding_model)
    embedding_model = SentenceTransformer(embedding_model_path).to("cuda").eval()

    # 加载评估模型
    eval_model_path = os.path.join(args.resource_base_path, "Model", args.eval_model)
    eval_model_tokenizer, eval_model = model_util.load_causal_lm(eval_model_path, -1)

    # 读取 cache 文件
    ## 共识实体列表
    consensus_entity_path = os.path.join("Train", "e2e_result", "consensus_entities.txt")
    consensus_entity = load_cache_file(consensus_entity_path)

    ## replace item
    replace_cache_file_dir = os.path.join("Train", "e2e_result")
    replace_cache_file_names = ["Qwen2.5-1.5B-Instruct_xsum(0,10000).txt",
                                "Qwen2.5-1.5B-Instruct-xsum_train_5000_ner_rand_direct_xsum(0,5000).txt"]
    replace_cache_file_paths = [os.path.join(replace_cache_file_dir, replace_cache_file)
                                for replace_cache_file in replace_cache_file_names]
    replace_cache_files = [load_cache_file(replace_path) for replace_path in replace_cache_file_paths]

    # 加载数据集
    dataset = datasets.load_from_disk(args.dataset_path)[args.dataset_split].select(range(0, args.data_size))

    # 开始计算
    with torch.inference_mode():
        replace_metric_scores = calculate_metrics()

    # 输出指标值
    for method, metric in replace_metric_scores.items():
        print(f"{method:^{10}}: {metric}")
