import argparse
import json
import os
import random
from typing import Any

import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from Config import prompt
from Utils import model_util, config_util


def pretty_output(model_output: str) -> str:
    """提取模型输出中</think>标签之后的内容"""
    return model_output.split("</think>")[-1].strip()


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="End-to-end replacement script for pseudonymization framework"
    )

    # 核心参数
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_size", type=int, default=1000)

    # 模型参数
    parser.add_argument("--end2end_model", type=str, default="Qwen3-4B")
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="multinomial_sampling",
        choices=["greedy", "beam_search", "contrastive_search", "multinomial_sampling"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_gpu", type=int, default=-1)

    # 路径参数
    parser.add_argument(
        "--dataset_base_path", type=str, default="/longzi05/houshilong/Dataset"
    )
    parser.add_argument(
        "--model_base_path", type=str, default="/longzi05/houshilong/Model"
    )

    return parser.parse_args()


def set_random_seed(seed: int = 42) -> None:
    """设置随机种子以确保结果可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(seed)


def end2end_inference(
    decode_mode: str,
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_text: str,
    max_new_tokens: int = 2048,
) -> str:
    """
    使用指定模型和参数进行端到端推理

    Args:
        decode_mode: 解码策略
        model: 加载的语言模型
        tokenizer: 对应的分词器
        prompt_text: 输入提示文本
        max_new_tokens: 最大生成 tokens 数

    Returns:
        模型生成的文本结果
    """
    input_tokens = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    # 生成配置字典，避免重复代码
    generate_kwargs = {
        **input_tokens,
        "max_new_tokens": max_new_tokens,
    }

    if decode_mode == "greedy":
        generate_kwargs.update({"num_beams": 1, "do_sample": False})
    elif decode_mode == "beam_search":
        generate_kwargs.update({"num_beams": 4, "do_sample": False})
    elif decode_mode == "contrastive_search":
        generate_kwargs.update({"penalty_alpha": 0.5, "top_k": 30})
    elif decode_mode == "multinomial_sampling":
        generate_kwargs.update(
            {
                "num_beams": 1,
                "do_sample": True,
                "temperature": 0.7,
            }
        )
    else:
        raise ValueError(f"未知的解码模式: {decode_mode}")

    raw_output = model.generate(**generate_kwargs)

    # 提取并解码生成的部分
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(input_tokens.input_ids, raw_output)
    ]
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return decoded_output


if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 固定随机种子
    set_random_seed(args.seed)

    # 加载数据集配置
    dataset_config = config_util.Config(
        os.path.join("Config", "dataset_config", args.dataset)
    )

    # 加载数据集
    dataset_full = datasets.load_from_disk(
        os.path.join(args.dataset_base_path, args.dataset)
    )

    # 选择指定范围的数据
    data_range = range(args.data_start, args.data_start + args.data_size)
    dataset = dataset_full[args.data_split].select(data_range)

    # 加载模型和分词器
    model_tokenizer, model = model_util.load_causal_lm(
        os.path.join(args.model_base_path, args.end2end_model), args.model_gpu
    )

    # 创建结果目录
    result_dir = os.path.join(
        "Train",
        "Result",
        f"{args.end2end_model}_{args.dataset}_{args.data_split}_{args.seed}",
    )
    os.makedirs(result_dir, exist_ok=True)

    # 结果文件路径
    result_file = os.path.join(
        result_dir, f"{args.data_start}_{args.data_start + args.data_size}.txt"
    )

    # 执行推理
    with torch.inference_mode():
        with open(result_file, "a", encoding="utf-8") as f:
            for idx, item in enumerate(tqdm(dataset, desc="Processing data")):
                torch.cuda.empty_cache()

                # 构建提示
                think_skip_tag = ""
                if args.end2end_model.startswith("Qwen3"):
                    think_skip_tag = """

<think>
</think>
                    
"""
                prompt_template = prompt.TASK["llm_end2end_long"]
                prompt_text = prompt_template.format(
                    item[dataset_config.input_keys[0]], think_skip_tag
                )

                # 执行推理
                raw_result = end2end_inference(
                    args.decode_mode,
                    model,
                    model_tokenizer,
                    prompt_text,
                    dataset_config.infer_max_new_tokens,
                )

                # 处理结果
                processed_result = pretty_output(raw_result)

                # 写入结果
                f.write(
                    json.dumps({item["id"]: processed_result}, ensure_ascii=False)
                    + "\n"
                )
