import argparse
import os
import random

import numpy as np
import torch
from transformers import set_seed


def init_parser():
    parser = argparse.ArgumentParser(description="接收脚本形参，控制脚本逻辑")

    # ========== base_run 系列参数 ========== #
    ## 数据集相关参数
    parser.add_argument("--dataset", type=str, required=True, help="选择数据集")
    parser.add_argument("--data_split", type=str, required=True, help="选择数据集split")
    parser.add_argument("--data_start", type=int, default=0, help="加载数据时的起始索引")
    parser.add_argument("--data_size", type=int, default=-1, help="指定实验数据量")

    ## 评估模型相关参数
    parser.add_argument("--eval_model", type=str, default="Qwen2.5-14B-Instruct", help="选择评估模型")
    parser.add_argument("--eval_model_gpu", type=int, default=-1, help="为评估模型指定设备")

    ## 实验描述
    parser.add_argument("--seed", type=int, default=42, help="设置种子值")
    parser.add_argument("--comment", type=str, default="debug", help="当前实验的描述")
    parser.add_argument("--cache_comment", type=str, default="debug", help="当前记录或者读取缓存的文件夹名")

    ## base run 控制参数
    parser.add_argument("--base_run", action="store_true", help="是否进行基准测试")
    parser.add_argument("--fast_run", action="store_true", help="是否使用缓存数据加快推理")

    # ========== sanitization 系列参数 ========== #
    ## saniti方案参数     选项: step:按步骤进行  end2end:使用小模型端到端实现
    parser.add_argument("--san_method", type=str, default="step", help="sanitization方案选择")

    ## 资源加载
    ### embedding 模型加载
    parser.add_argument("--embedding_model", type=str, default="", help="SentenceTransformer模型")

    ### ner 模型加载
    parser.add_argument("--ner_model", type=str, default="", help="选择 NER 模型")
    parser.add_argument("--ner_model_gpu", type=int, default=0, help="为 ner 模型指定设备")

    ### 工具 LLM 模型加载
    parser.add_argument("--tool_model", type=str, default="", help="工具模型")
    parser.add_argument("--tool_model_gpu", type=int, default=0, help="为工具模型指定设备")

    ### repeat 模型加载
    parser.add_argument("--repeat_model", type=str, default="", help="repeat模型")
    parser.add_argument("--repeat_model_gpu", type=int, default=0, help="为repeat模型指定设备")

    ### ner_generate 模型加载
    parser.add_argument("--ner_gen_model", type=str, default="", help="ner_repeat模型")
    parser.add_argument("--ner_gen_model_gpu", type=int, default=0, help="为ner_repeat模型指定设备")
    # [tag_wrap、only_rep]
    parser.add_argument("--ner_gen_model_type", type=str, default="", help="finetune ner_repeat模型的类型")

    ## ner 参数
    parser.add_argument("--ner_keys", type=str, default=["PER", "LOC", "ORG"], nargs="+", help="实体类型列表")
    parser.add_argument("--entity_score_threshold", type=float, default=[0.8], nargs="+", help="实体置信度门限")

    ## entity dict 过程参数     选项: ner_dict_select: 在预抽取的实体字典中选择  llm_generate: LLM prompt 生成
    parser.add_argument("--entity_map_method", type=str, default="", help="候选实体的获取方式")

    ## replace/generate 过程参数 [str_replace、llm_generate]
    ### 选项: str_replace 直接对文本调用字符串replace操作   llm 通过小型llm模型根据输入的实体字典执行生成式替换
    parser.add_argument("--rep_gen_method", type=str, default="", help="replace/generate过程的方式")

    ## 小型llm replace/generate 过程参数
    ### 选项: prompt 使用prompt指导小型llm生成候选实体   top_k 通过语言模型生成next_token时的候选token中选择候选token
    parser.add_argument("--gen_next_token_type", type=str, default="prompt", help="选择生成next_token的方法")
    parser.add_argument("--top_k_range", type=int, default=20, help="使用top_k方式获取实体时的搜索范围")

    ### 选项: generate_with_entity_and_type_JSON、generate_with_entity_and_type、generate_with_entity、generate_with_type
    parser.add_argument("--gen_entity_input", type=str, default="", help="prompt指导生成候选实体时的参考内容")
    parser.add_argument("--early_stop", action="store_true", help="控制rep/gen过程提前结束")
    # search相关参数
    parser.add_argument("--search_suffix_thresold", type=int, default=1, help="")
    parser.add_argument("--search_range", type=float, default=0.2, help="")
    parser.add_argument("--search_diff_length", type=int, default=20, help="")
    parser.add_argument("--output_text_head", type=str, default="Input:")
    parser.add_argument("--search_replace_constraint", type=str, default="context")

    ## recover 过程参数     选项: str_replace 直接对文本调用字符串replace操作   llm_generate 通过生成式替换实现
    parser.add_argument("--recover_method", type=str, default="", help="recover过程的方式")

    args = parser.parse_args()

    # ========== 互斥参数校验与优化 ========== #
    ## 基准测试时，清空 sanitization 相关参数
    if args.base_run:
        args.san_method = ""
        args.embedding_model = ""
        args.tool_model = ""
        args.tool_model_gpu = -1
        args.ner_keys = []
        args.entity_score_threshold = []
        args.gen_entity_input = ""
    if args.base_run or args.san_method.startswith("end2end"):
        args.ner_model = ""
        args.ner_model_gpu = -1
        args.entity_map_method = ""
        args.rep_gen_method = ""
        args.gen_next_token_type = ""
        args.top_k_range = -1
        args.early_stop = False

    need_tool_model = False
    if (
            args.san_method.startswith("end2end") or
            args.entity_map_method == "llm_generate" or
            args.rep_gen_method.endswith("llm_generate") or
            args.dataset == "wmt14-de-en"
    ):
        assert args.tool_model != "" and args.tool_model_gpu != -1
        need_tool_model = True

    # ========== 系统资源路径配置 ========== #
    if os.name == 'nt':
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
        args.metric_base_path = "Metrics"
    else:
        args.dataset_base_path = "Dataset"
        args.model_base_path = "Model"
        args.metric_base_path = "Metrics"

    # 配置文件路径
    args.dataset = os.path.join("Config", "dataset_config", args.dataset)
    args.eval_model = os.path.join("Config", "model_config", args.eval_model)
    args.tool_model = os.path.join("Config", "model_config", args.tool_model) if args.tool_model else args.tool_model
    args.ner_model = os.path.join("Config", "model_config", args.ner_model) if args.ner_model else args.ner_model

    return args, need_tool_model


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


# 修改该方法中的参数，获取终端执行时的命令字符串
def get_cmd():
    # 创建一个参数字典
    visible_gpus = "0"  # 脚本可见GPU
    script_file = "main.py"  # 指定脚本
    arg_dict = {
        # ========== base_run 系列参数 ========== #
        ## 数据集相关参数
        "--dataset": "wmt14-de-en",  # 选择数据集
        "--data_split": "test",  # 选择数据集split [train、validation、test]
        "--data_start": -1,  # 试验数据起始截取下标，默认为0
        "--data_size": 1000,  # 指定实验数据量，不指定（全集）则填-1

        ## 评估模型相关参数
        "--eval_model": "Qwen2.5-1.5B-Instruct",  # 选择评估模型 [Qwen2.5-1.5B-Instruct、Qwen2.5-14B-Instruct]
        "--eval_model_gpu": 0,  # 为评估模型指定GPU，不指定（-1）则默认device="auto"

        ## 实验描述
        "--seed": -1,  # 设置种子值，默认即可
        "--comment": "debug-翻译",  # 当前实验的描述，会拼接到日志文件名
        "--cache_comment": "debug-翻译",  # 当前实验的中间变量保存路径名，和comment保持一致即可

        ## base run 控制参数
        "--base_run": True,  # 是否进行基准测试，即不执行 sanitization 过程
        "--fast_run": -1,  # 是否加载实验中间结果（缓存）加快推理速度（方案四）

        # ========== sanitization 系列参数 ========== #
        ## saniti方案参数
        ### 选项: step:按步骤进行(默认)   end2end:使用小模型端到端实现   end2end_step_prompt:使用小模型端到端分步prompt
        "--san_method": "end2end_step_prompt",

        # embedding 模型加载（已经为必须项）
        "--embedding_model": "all-mpnet-base-v2",  # [all-mpnet-base-v2]

        # ner 模型加载
        "--ner_model": "",  # 选择 NER 模型 [ner_pipe]
        "--ner_model_gpu": -1,  # 为 ner 模型指定设备，默认选第一张卡

        # 工具 LLM 模型加载
        "--tool_model": "Qwen2.5-1.5B-Instruct",  # 小LLM工具模型 [Qwen2.5-1.5B-Instruct]
        "--tool_model_gpu": -1,  # 为小LLM工具模型指定设备，默认选第一张卡

        # repeat 模型加载
        "--repeat_model": "",  # 加载finetune repeat模型 [ft_repeat_model]
        "--repeat_model_gpu": -1,

        # ner_generate 模型加载（方案四的微调模型）
        "--ner_gen_model": "",  # [ft_ner_gen_model_only_rep、ft_ner_gen_model_tag_wrap]
        "--ner_gen_model_gpu": -1,
        "--ner_gen_model_type": "",

        ## ner 过程参数
        "--ner_keys": "entity",  # 待消除隐私实体类型列表
        "--entity_score_threshold": "0.8",  # 从 ner result 提取实体时的最低置信度

        # entity dict 过程参数
        "--entity_map_method": "",  # 候选实体的获取方式 [ner_dict_select（方案一）、llm_generate（方案1.5）]

        # replace/generate 过程参数
        "--rep_gen_method": "",  # replace/generate过程的方式 [str_replace、llm_generate、ft_llm_generate]
        "--gen_next_token_type": "",  # 选择 rep/gen 过程中生成 next_token 的方法 [prompt、top_k]
        "--gen_entity_input": "entity",  # 生成候选实体时的输入内容 [entity_and_type_JSON、entity_and_type、entity、type]
        "--top_k_range": -1,  # 使用top_k方式获取实体时的搜索范围
        "--early_stop": -1,  # 控制rep/gen过程提前结束 [True]

        # 方案四相关参数，其他方案这块不需要
        "--search_suffix_thresold": -1,  # only_rep模型填1，tag_wrap模型填2
        "--search_range": -1,  # only_rep模型填0.2，tag_wrap模型填0.5
        "--search_diff_length": -1,  # 固定[20]
        "--output_text_head": "",  # 固定[Input:]
        "--search_replace_constraint": "",  # 固定[context]，仅tag_wrap模型需要

        # recover 过程参数     选项: str_replace 直接对文本调用字符串replace操作   llm_generate 通过生成式替换实现   llm_end2end_step3:通过llm prompt实现
        "--recover_method": "llm_end2end_step3",  # 方案一、二、四均需要 str_replace; 方案三需要llm_generate; 方案四可结合后者
    }

    cmd = "CUDA_VISIBLE_DEVICES=" + visible_gpus + " python " + script_file
    for k, v in arg_dict.items():
        if v == -1 or v == "":  # 跳过参数
            continue
        if k == "--base_run":
            cmd += " " + k
            break
        if k == "--early_stop":
            cmd += " " + k
            continue
        cmd += " " + k + " " + str(v)
    return cmd


if __name__ == "__main__":
    print(get_cmd())
