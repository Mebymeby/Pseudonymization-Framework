import argparse
import os
import random

import numpy as np
import torch
from transformers import set_seed


def init_parser():
    parser = argparse.ArgumentParser(description="接收脚本形参，控制脚本逻辑")

    # ========== 基础参数 ========== #
    ## 数据集相关参数
    parser.add_argument("--dataset", type=str, required=True, help="选择数据集")
    parser.add_argument("--data_split", type=str, required=True, help="选择数据集split")
    parser.add_argument("--data_start", type=int, default=0, help="加载数据时的起始索引")
    parser.add_argument("--data_size", type=int, default=-1, help="实验数据量")

    ## 实验描述
    parser.add_argument("--seed", type=int, default=42, help="设置种子值")
    parser.add_argument("--comment", type=str, default="[DEBUG_RUN]", help="当前实验的描述")

    ## base_run 控制参数
    parser.add_argument("--base_run", action="store_true", help="是否只进行基准测试")

    ## pseu_run 控制参数
    parser.add_argument("--run_step", type=int, default=[1, 1, 1, 1], nargs="+", help="各步骤是否执行")
    parser.add_argument("--fast_run_step", type=int, default=[0, 0, 0, 0], nargs="+", help="各步骤是否加载缓存")
    parser.add_argument("--run_eval", type=int, default=[1, 1], nargs="+", help="两次推理是否执行")
    parser.add_argument("--fast_run_eval", type=int, default=[0, 0], nargs="+", help="两次推理是否加载缓存")
    parser.add_argument("--record_cache", action="store_true", help="标记当前实验是否记录缓存")

    # ========== 模型参数 ========== #
    ## eval_model
    parser.add_argument("--eval_model", type=str, default="Qwen2.5-14B-Instruct", help="评估模型")
    parser.add_argument("--eval_model_gpu", type=int, default=-1, help="为评估模型指定设备")

    ## ner_model
    parser.add_argument("--ner_model", type=str, default="bert-large-cased-finetuned-conll03-english", help="NER模型")
    parser.add_argument("--ner_model_gpu", type=int, default=0, help="为NER模型指定设备")

    ## tool_model
    parser.add_argument("--tool_model", type=str, default="Qwen2.5-1.5B-Instruct", help="选择工具模型")
    parser.add_argument("--tool_model_gpu", type=int, default=0, help="为工具模型指定设备")

    ## ft_repeat_model
    parser.add_argument("--repeat_model", type=str, default="ft_repeat_model", help="")
    parser.add_argument("--repeat_model_gpu", type=int, default=0, help="为ft_repeat_model指定设备")

    ## tag_model
    parser.add_argument("--tag_model", type=str, default="", help="tag_model")
    parser.add_argument("--tag_model_gpu", type=int, default=0, help="为tag_model指定设备")
    parser.add_argument("--tag_model_type", type=str, default="", help="tag_model的类型")

    ## embedding 模型
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2", help="SentenceTransformer模型")

    # ========== Pseudonymization 参数 ========== #
    ## [Detect] process
    parser.add_argument("--detect_method", type=str, help="Detect方案选择")
    parser.add_argument("--entity_types", type=str, default=["PER", "LOC", "ORG"], nargs="+", help="实体类型列表")
    parser.add_argument("--entity_score_threshold", type=float, default=[0.9], nargs="+", help="实体置信度门限")

    ## [Generate] process
    parser.add_argument("--generate_method", type=str, help="Generate方案选择")
    parser.add_argument("--llm_generate_entity_prompt_input", type=str, default="entity", help="生成实体时prompt的输入")
    parser.add_argument("--top_k_range", type=int, default=30, help="top_k 搜索范围")

    ## [Replace] process
    parser.add_argument("--replace_method", type=str, help="Replace方案选择")
    parser.add_argument("--generate_early_stop", action="store_true", help="前向传播生成过程提前终止标记")

    ## [Recover] process
    parser.add_argument("--recover_method", type=str, help="Recover方案选择")

    parser.add_argument("--end2end", action="store_true", help="标记当前实验是否为端到端数据")

    # ========== generative 过程子参数 ========== #
    parser.add_argument("--search_diff_length", type=int, default=50, help="")
    parser.add_argument("--search_suffix_threshold", type=int, default=2, help="")

    args = parser.parse_args()

    # ========== 系统资源路径配置 ========== #
    if os.name == 'nt':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
        args.metric_base_path = "Metrics"
    else:
        args.dataset_base_path = "/data4/houshilong/Dataset"
        args.model_base_path = "/data4/houshilong/Model"
        args.metric_base_path = "/data4/houshilong/Metrics"

    # 配置文件路径
    args.dataset_path = os.path.join("Config", "dataset_config", args.dataset)

    return args


# 设定种子值
def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)
