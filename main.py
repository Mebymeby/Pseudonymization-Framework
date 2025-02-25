import cProfile
import os
import pstats
import json

import torch
from tqdm import tqdm

from Utils import config_util, logging_util, argparse_util, dataset_util, model_util, score_util, saniti_func, \
    infer_util


def run():
    # 初始化分数记录数据结构
    scores_dict = score_util.init_score(dataset_config)

    # 遍历数据集
    data_size = len(data)
    for idx, item in enumerate(tqdm(data)):
        # 每个item开始前清空缓存
        torch.cuda.empty_cache()

        if dataset_config.name == "wikitext-103":
            if len(item["text"].split()) < 100:  # 续写数据小于100个词，视为无效数据
                data_size -= 1
                continue
        if dataset_config.name == "wmt14-de-en":
            item = item["translation"]

        # ========== ↓ ========== base run ========== ↓ ========== #

        # 执行推理，得到可直接用于评估的输出
        ori_outputs, _ = infer_util.eval_inference(item, eval_model, eval_tokenizer, eval_model_config, dataset_config)

        # 计算评估值，并记录结果
        ori_result = score_util.compute_score(item, ori_outputs, metric, dataset_config, eval_model, eval_tokenizer)
        ori_result.update({"san_effect": 0.0})
        for key in scores_dict["ori_scores"].keys():
            scores_dict["ori_scores"][key].append(ori_result[key])

        # base run 截断后续 sanitization 过程
        if args.base_run:
            continue

        # ========== ↓ ========== sanitization run ========== ↓ ========== #
        if args.fast_run:  # 读取缓存
            sanitized_item = sanitized_item_record[idx]
            entity_map = entity_map_record[idx]
            san_effect = san_effect_record[idx]
        else:  # 无缓存则直接推理
            # 构造 sanitized_item，其他行为保持与基准测试一致
            sanitized_item, entity_map, san_effect = saniti_func.sanitize_item(
                args, item, entity_dict, dataset_config, ner_model, ner_key, threshold, embedding_model,
                tool_model, tool_tokenizer, repeat_model, repeat_tokenizer, ner_gen_model, ner_gen_tokenizer)

        # 获取德语entity_map
        entity_map_de = {}
        if dataset_config.name == "wmt14-de-en":
            entity_map_de = saniti_func.gen_de_map(item["en"], sanitized_item["en"], tool_model, tool_tokenizer)

        # 执行推理，得到可直接用于评估的输出
        sanitized_outputs, raw_output = infer_util.eval_inference(sanitized_item, eval_model, eval_tokenizer,
                                                                  eval_model_config, dataset_config)

        # sanitization 标记
        sanitized_tag = False  # 标记当前样本是否 sanitized
        if entity_map:  # sanitized 样本
            sanitized_tag = True
            scores_dict["sanitized_count"] += 1  # sanitized 样本计数

        # 还原模型输出中的隐私信息
        if dataset_config.name == "glue-mnli":  # 分类任务的输出只有一个标签，不需要recover
            recovered_outputs = sanitized_outputs
        else:
            if dataset_config.name == "wmt14-de-en":
                recover_entity_map = entity_map_de
            else:
                recover_entity_map = entity_map
            recovered_outputs = saniti_func.recover_output(args, recover_entity_map, sanitized_outputs,
                                                           tool_model, tool_tokenizer, repeat_model, repeat_tokenizer)

        if sanitized_tag and recovered_outputs == ori_outputs:
            scores_dict["transparent_sanitized_count"] += 1

        # 计算评估值，并记录结果
        san_result = score_util.compute_score(item, recovered_outputs, metric, dataset_config,
                                              eval_model, eval_tokenizer)
        san_result.update(san_effect)

        for key in scores_dict["san_scores"].keys():
            scores_dict["san_scores"][key].append(san_result[key])
            if sanitized_tag:
                scores_dict["sanitized_ori_scores"][key].append(ori_result[key])
                scores_dict["sanitized_san_scores"][key].append(san_result[key])

        # badcase.log 记录
        if sanitized_tag:
            logging_util.add_badcase_log(badcase_logger, ori_result, san_result, dataset_config,
                                         item, sanitized_item, entity_map, ori_outputs, raw_output, sanitized_outputs)

        # 缓存当前item的中间结果
        logging_util.add_item_result(ori_result, san_result, entity_map, sanitized_item, san_effect, cache_file_path)

    # 记录总结果
    # for key, value in scores_dict.items():
    #     if key != "sanitized_count" and key != "transparent_sanitized_count":
    #         for k, v in value.items():
    #             weighted_mean_score[key][k].extend(v)

    # result.log 记录
    logging_util.add_result_log(args, result_logger, scores_dict, data_size)

    return


if __name__ == '__main__':

    # ========== ↓ ========== 资源初始化 ========== ↓ ========== #

    # 初始化参数解析器，读取 cmd 脚本形参，用于控制验证流程
    args, need_tool_model = argparse_util.init_parser()

    # 设置种子
    argparse_util.set_random_seed(args.seed)

    # 读取配置文件，初始化配置对象
    dataset_config = config_util.Config(args.dataset)  # 数据集相关配置
    eval_model_config = config_util.Config(args.eval_model)  # 评估模型相关配置
    if args.ner_model:
        ner_model_config = config_util.Config(args.ner_model)  # ner 模型相关配置
    if need_tool_model:
        tool_model_config = config_util.Config(args.tool_model)  # 工具模型相关配置

    # 配置并获取日志执行器，记录配置信息
    result_logger, badcase_logger, dataset_log_dir = logging_util.config_logger(args, dataset_config)
    result_logger.info("args: {}".format(json.dumps(vars(args), indent=4, ensure_ascii=False)))
    result_logger.info("dataset_config: {}".format(dataset_config.print_config()))
    result_logger.info("eval_model_config: {}".format(eval_model_config.print_config()))
    if args.ner_model:
        result_logger.info("ner_model_config: {}".format(ner_model_config.print_config()))
    if need_tool_model:
        result_logger.info("tool_model_config: {}".format(tool_model_config.print_config()))

    # 加载数据集，以及相应的评估指标
    data = dataset_util.load_dataset(args, dataset_config)
    metric = score_util.load_metric(args, dataset_config)

    # 读取缓存文件
    cache_file_path = os.path.join("Output", "Log", dataset_config.name, "item_cache", args.cache_comment)
    os.makedirs(cache_file_path, exist_ok=True)
    sanitized_item_record, entity_map_record, san_effect_record = [], [], []
    if args.fast_run:
        sanitized_item_record, entity_map_record, san_effect_record = logging_util.load_item_cache(cache_file_path)

    # 开启性能分析
    profiler = cProfile.Profile()
    profiler.enable()

    # weighted_mean_score = score_util.init_score(dataset_config)  # 加权平均总结果，区分实体时使用

    with torch.inference_mode():
        # 加载评估模型，finetune模型 or 通用LLM，由配置文件区分
        eval_model, eval_tokenizer = model_util.load_eval_model(args, eval_model_config)

        # ========== base run ========== #
        if args.base_run:
            run()
        # ========== sanitization run ========== #
        else:
            # 资源定义
            entity_dict = {}  # 方案一
            ner_model, ner_tokenizer = None, None  # 方案一、方案三:提取输入数据的实体
            tool_model, tool_tokenizer = None, None  # 方案二:端到端模型; 方案三:实体生成、repeat模型; 方案四:实体生成模型
            repeat_model, repeat_tokenizer = None, None  # 方案三:finetune repeat 模型
            ner_gen_model, ner_gen_tokenizer = None, None  # 方案四:finetune ner_generate 模型

            # 资源加载

            # sentence transformer embedding模型
            embedding_model = model_util.load_embedding_model(args)  # 方案一:辅助筛选实体; 全方案:计算隐私净化效率

            # 方案二只需要一个工具模型
            if args.san_method == "end2end":
                if args.tool_model == args.eval_model:  # 模型对象重复利用
                    tool_model, tool_tokenizer = eval_model, eval_tokenizer
                else:
                    tool_model, tool_tokenizer = model_util.load_tool_model(args, tool_model_config)
            else:
                # 加载实体字典
                if args.entity_map_method == "ner_dict_select" or args.gen_next_token_type == "dict_select":
                    entity_dict = dataset_util.load_entity_dict(dataset_config)

                # 加载ner模型
                if args.ner_model:
                    ner_model, ner_tokenizer = model_util.load_ner_model(args, ner_model_config)

                # 加载工具模型
                if need_tool_model:
                    if args.tool_model == args.eval_model:  # 模型对象重复利用
                        tool_model, tool_tokenizer = eval_model, eval_tokenizer
                    else:
                        tool_model, tool_tokenizer = model_util.load_tool_model(args, tool_model_config)

                # 加载 finetune repeat_model
                if args.repeat_model:
                    if args.repeat_model == args.tool_model:
                        repeat_model, repeat_tokenizer = tool_model, tool_tokenizer
                    else:
                        repeat_model, repeat_tokenizer = model_util.load_repeat_model(args)

                # 加载 finetune ner_gen_model
                if args.ner_gen_model:
                    ner_gen_model, ner_gen_tokenizer = model_util.load_ner_gen_model(args)

            # 实验附加维度：ner_keys * entity_score_threshold
            for ner_key in args.ner_keys:
                for threshold in args.entity_score_threshold:
                    result_logger.info("ner_key: %s ,confidence_threshold: %s", ner_key, threshold)
                    run()

    # result_logger.info("多实体加权得分")
    # logging_util.add_result_log(args, result_logger, weighted_mean_score, 1)

    # 性能分析统计
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumtime").print_stats(20)  # 仅显示 cumtime 最多的 20 个函数
