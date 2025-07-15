import argparse
import json
import os

import datasets
import torch
from tqdm import tqdm

from Config import prompt
from Utils import model_util, infer_util, config_util

if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_size", type=int, default=10000)
    parser.add_argument("--end2end_model", type=str, default="Qwen2.5-1.5B-Instruct_sft_merged_v3") # Qwen2.5-1.5B-Instruct_sft_merged
    parser.add_argument("--model_gpu", type=int, default=-1)
    args = parser.parse_args()

    if os.name == 'nt':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
    else:
        args.dataset_base_path = "/data4/houshilong/Dataset"
        args.model_base_path = "/data4/houshilong/Model"

    # 数据集
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", args.dataset))
    dataset_full = datasets.load_from_disk(os.path.join(args.dataset_base_path, args.dataset))  # 加载完整数据集
    dataset = dataset_full[args.data_split].select(range(args.data_start, args.data_start + args.data_size))

    # 模型
    end2end_model_tokenizer, end2end_model = model_util.load_causal_lm(
        os.path.join(args.model_base_path, args.end2end_model), args.model_gpu)

    # 记录结果
    cache_path = os.path.join("Train", "e2e_result")
    cache_file = os.path.join(cache_path,
                              args.end2end_model + "_" + args.dataset + "(" + str(args.data_start) + "," + str(
                                  args.data_start + args.data_size) + ").txt")
    os.makedirs(cache_path, exist_ok=True)

    # 调 prompt 推理
    with torch.inference_mode():
        with open(cache_file, "a", encoding="utf-8") as f:
            for item in tqdm(dataset):
                torch.cuda.empty_cache()
                prompt_with_data = prompt.TASK["llm_end2end"].format(item[dataset_config.input_keys[0]])
                res = infer_util.prompt_inference(
                    end2end_model, end2end_model_tokenizer, prompt_with_data, dataset_config.infer_max_new_tokens)
                f.write(json.dumps({item["id"]: res}, ensure_ascii=False) + '\n')
