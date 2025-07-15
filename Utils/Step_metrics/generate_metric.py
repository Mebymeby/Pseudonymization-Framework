import argparse
import json
import math
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from Utils import model_util, config_util


def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]


# 计算两个实体的语义不相似度
def calculate_privacy_preservation(ent1, ent2):
    eb1 = embedding_model.encode(ent1, convert_to_tensor=True)
    eb2 = embedding_model.encode(ent2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(eb1, eb2).squeeze(0).item()
    return 1 - sim


# 计算整个句子的困惑度
def calculate_perplexity(sentence):
    inputs = eval_model_tokenizer(sentence, return_tensors="pt").to(eval_model.device)
    outputs = eval_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())


def calculate_metrics():
    generate_metric_scores = {generate_method: {
        "Pps": 0.0,  # 隐私保护效率
        "Plx": 0.0,  # 语句困惑度
        "Cpps": 0.0,  # 上下文隐私保护效率
    } for generate_method in args.generate_method}

    pps_score_list_map = {generate_method: [] for generate_method in args.generate_method}
    pls_score_list_map = {generate_method: [] for generate_method in args.generate_method}

    for idx in tqdm(range(args.data_size)):
        for i, generate_cache_file in enumerate(generate_cache_files):
            cur_item_entity_map = generate_cache_file[idx]
            for key, value in cur_item_entity_map.items():
                sim = calculate_privacy_preservation(key, value)
                pps_score_list_map[args.generate_method[i]].append(sim)

        for i, replace_cache_file in enumerate(replace_cache_files):
            perplexity = calculate_perplexity(replace_cache_file[idx][dataset_config.input_keys[0]])
            pls_score_list_map[args.generate_method[i]].append(perplexity)

    privacy_preservation_scores = {key: np.mean(val) for key, val in pps_score_list_map.items()}
    semantic_correctness_score = {key: np.mean(val) for key, val in pls_score_list_map.items()}

    for generate_method, pps in privacy_preservation_scores.items():
        generate_metric_scores[generate_method]["Pps"] = float(pps)
    for generate_method, plx in semantic_correctness_score.items():
        generate_metric_scores[generate_method]["Plx"] = float(plx)
    for generate_method, score_dict in generate_metric_scores.items():
        generate_metric_scores[generate_method]["Cpps"] = math.exp(
            args.exp_lambda *
            generate_metric_scores[generate_method]["Pps"] / generate_metric_scores[generate_method]["Plx"])

    return generate_metric_scores


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--eval_model", type=str, default="Qwen2.5-1.5B-Instruct")
    parser.add_argument("--detect_method", type=str, default="ner")  # 控制变量
    parser.add_argument("--generate_method", type=str, nargs="+", default=["rand", "prompt_entity_and_type"])
    parser.add_argument("--replace_method", type=str, default="direct")  # 控制变量
    parser.add_argument("--data_size", type=int, default=10000)
    parser.add_argument("--exp_lambda", type=int, default=10)
    args = parser.parse_args()
    # 资源路径
    args.resource_base_path = "" if os.name == "nt" else os.path.join("/data4", "houshilong")

    # 数据集配置
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", args.dataset))

    # 加载 embedding 模型
    embedding_model_path = os.path.join(args.resource_base_path, "Model", args.embedding_model)
    embedding_model = SentenceTransformer(embedding_model_path).to("cuda").eval()

    # 加载评估模型
    eval_model_path = os.path.join(args.resource_base_path, "Model", args.eval_model)
    eval_model_tokenizer, eval_model = model_util.load_causal_lm(eval_model_path, -1)

    # 读取 cache 文件
    generate_cache_file_dir = os.path.join("Output", "Cache", args.dataset, "Generate")
    replace_cache_file_dir = os.path.join("Output", "Cache", args.dataset, "Replace")

    generate_cache_file_names = [args.detect_method + "-" + generate_method + ".txt"
                                 for generate_method in args.generate_method]
    replace_cache_file_names = [args.detect_method + "-" + generate_method + "-" + args.replace_method + ".txt"
                                for generate_method in args.generate_method]

    generate_cache_file_paths = [
        os.path.join(generate_cache_file_dir, generate_cache_file) for generate_cache_file in generate_cache_file_names]
    replace_cache_file_paths = [
        os.path.join(replace_cache_file_dir, replace_cache_file) for replace_cache_file in replace_cache_file_names]

    generate_cache_files = [load_cache_file(generate_path) for generate_path in generate_cache_file_paths]
    replace_cache_files = [load_cache_file(replace_path) for replace_path in replace_cache_file_paths]

    # 开始计算
    with torch.inference_mode():
        generate_metric_scores = calculate_metrics()

    # 输出指标值
    for method, metric in generate_metric_scores.items():
        print(f"{method:^{22}}: {metric}")
