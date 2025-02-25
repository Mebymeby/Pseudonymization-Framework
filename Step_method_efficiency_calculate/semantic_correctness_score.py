import json
import math
import os.path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def score(sentence):  # 计算整个句子的困惑度
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return math.exp(loss.item())


if __name__ == '__main__':
    # 加载大模型
    model_path = os.path.join("Model", "Qwen2.5-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    model.eval()

    # 读取各方案的替换后句子
    direct_rep_sentence = []
    prompt_rep_sentence = []
    gen_rep_sentence = []

    with open(os.path.join("Step_method_efficiency_calculate", "pseudonymization_sentence", "direct_replace.txt"), "r",
              encoding="utf-8") as f:
        for line in f:
            direct_rep_sentence.append(json.loads(line.strip())["document"])
    with open(os.path.join("Step_method_efficiency_calculate", "pseudonymization_sentence", "prompt_replace.txt"), "r",
              encoding="utf-8") as f:
        for line in f:
            prompt_rep_sentence.append(json.loads(line.strip())["document"])
    with open(os.path.join("Step_method_efficiency_calculate", "pseudonymization_sentence", "gen_replace.txt"), "r",
              encoding="utf-8") as f:
        for line in f:
            gen_rep_sentence.append(json.loads(line.strip())["document"])

    # 开始计算
    semantic_correctness_score_list_map = {
        "direct": [],
        "prompt": [],
        "gen"   : []
    }

    with torch.inference_mode():
        for idx in tqdm(range(len(direct_rep_sentence))):
            direct_replace_score = score(direct_rep_sentence[idx])
            prompt_replace_score = score(prompt_rep_sentence[idx])
            gen_replace_score = score(gen_rep_sentence[idx])
            semantic_correctness_score_list_map["direct"].append(direct_replace_score)
            semantic_correctness_score_list_map["prompt"].append(prompt_replace_score)
            semantic_correctness_score_list_map["gen"].append(gen_replace_score)

    semantic_correctness_score = {key: np.mean(val) for key, val in semantic_correctness_score_list_map.items()}

    print(semantic_correctness_score)
