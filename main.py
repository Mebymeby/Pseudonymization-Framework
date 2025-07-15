import cProfile
import io
import json
import os
import pstats

import torch
from tqdm import tqdm

from Utils import config_util, logging_util, argparse_util, dataset_util, model_util, score_util, infer_util, \
    detect_process, cache_util, generate_process, replace_process, recover_process


# 评估主函数
def run():
    # 初始化分数记录数据结构
    scores_dict = score_util.init_score(dataset_config)

    # 遍历数据集
    for idx, item in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()  # 每个 item 开始前清空 torch 分配的显存缓存
        idx += args.data_start

        ## 获取 item 中所有需要 detection 的部分，并裁切部分过大的 item
        item_input_content_list = dataset_util.get_item_input_content_list(item, dataset_config)
        item_input_content = " ".join(item_input_content_list)

        # ========== ↓ ========== base run ========== ↓ ========== #

        # +++++ 执行推理 +++++ #
        ori_output_list = []
        ori_result = {}
        if args.run_eval[0]:
            if args.fast_run_eval[0]:
                if not ori_output_list:
                    ori_output_list = cache_util.load_ori_eval_result(args)
                ori_output = ori_output_list[idx]["ori_eval"]
            else:
                ori_output = infer_util.eval_inference(item, eval_model, eval_tokenizer, dataset_config)

            if args.record_cache:
                cache_util.save_ori_eval_result(args, ori_output, cache_file_dir)

            # 计算评估值，并记录结果
            if args.run_step[3] and args.run_eval[0] and args.run_eval[1]:
                ori_result = score_util.compute_score(args, item, ori_output, dataset_config)
                for key in scores_dict["full_ori_scores"].keys():
                    scores_dict["full_ori_scores"][key].append(ori_result[key])

        # base run 截断后续 Pseudonymization 过程
        if args.base_run:
            continue

        # ========== ↓ ========== Pseudonymization run ========== ↓ ========== #

        # ========== [Detect] entities ========== #
        entity_list_dict = {}
        entity_list_map = {}
        if args.run_step[0] and not args.fast_run_step[1]:
            if args.fast_run_step[0]:
                if not entity_list_dict:
                    entity_list_dict = cache_util.load_detect_result(args)
                entity_list_map = entity_list_dict[idx]
            else:
                entity_list_map = detect_process.detect_entity_list(
                    args, dataset_config, item_input_content, ner_model, tool_model, tool_model_tokenizer)

        # ========== [Generate] candidate entities ========== #
        candidate_entity_dict = {}
        candidate_entity_map = {}
        if args.run_step[1] and not args.detect_method.startswith("tag"):
            if args.fast_run_step[1]:
                if not candidate_entity_dict:
                    candidate_entity_dict = cache_util.load_generate_result(args)
                candidate_entity_map = candidate_entity_dict[idx]
            else:
                candidate_entity_map = generate_process.generate_candidate_entity_map(
                    args, entity_list_map, dataset_entity_dict, item_input_content,
                    tool_model, tool_model_tokenizer, tag_model, tag_model_tokenizer, embedding_model)

        # ========== [Replace] privacy entity in text ========== #
        pseu_item_list = []
        pseu_item = {}
        if args.run_step[2] or args.detect_method.startswith("tag"):
            if args.fast_run_step[2]:
                if not pseu_item_list:
                    pseu_item_list = cache_util.load_replace_result(args)
                pseu_item = pseu_item_list[idx]
            else:
                pseu_item = replace_process.replace_privacy_entity(
                    args, dataset_config, item, entity_list_map, candidate_entity_map, dataset_entity_dict,
                    tool_model, tool_model_tokenizer, repeat_model, repeat_model_tokenizer,
                    tag_model, tag_model_tokenizer, ner_model, embedding_model)

        # +++++ 执行推理 +++++ #
        pseu_output_list = []
        pseu_output = ""
        if args.run_eval[1]:
            if args.fast_run_eval[1]:
                if not pseu_output_list:
                    pseu_output_list = cache_util.load_pseu_eval_result(args)
                pseu_output = pseu_output_list[idx].get("pseu_eval")
            else:
                pseu_output = infer_util.eval_inference(pseu_item, eval_model, eval_tokenizer, dataset_config)

        # ========== [Recover] privacy entity in text ========== #
        recover_output_list = []
        recover_output = ""
        if args.run_step[3]:
            if args.fast_run_step[3]:
                if not recover_output_list:
                    recover_output_list = cache_util.load_recover_result(args)
                recover_output = recover_output_list[idx]["recover"]
            else:
                # 获取 recover_entity_candidate_map
                if args.dataset == "wmt14-de-en":
                    recover_entity_candidate_map = generate_process.gen_entity_map_trans(
                        dataset_config, item[dataset_config.input_keys[0]], pseu_item[dataset_config.input_keys[0]],
                        tool_model, tool_model_tokenizer)
                else:
                    recover_entity_candidate_map = candidate_entity_map
                # 执行 recover
                recover_output = recover_process.recover_privacy_entity(
                    args, dataset_config, pseu_output, recover_entity_candidate_map,
                    tool_model, tool_model_tokenizer, repeat_model, repeat_model_tokenizer)

        # 计算评估值，并记录结果
        if args.run_step[3] and args.run_eval[0] and args.run_eval[1]:  # 执行全部流程后计算分数
            pseu_result = score_util.compute_score(args, item, recover_output, dataset_config)
            log_str_fmt = " [ori_result] : { "
            for key, val in ori_result.items():
                log_str_fmt += key + ": " + f"{round(val, 5):.5f}" + " "
            log_str_fmt += "} | [pseu_result] : { "
            for key, val in pseu_result.items():
                log_str_fmt += key + ": " + f"{round(val, 5):.5f}" + " "
            log_str_fmt += "}"

            for key in scores_dict["full_pseu_scores"].keys():
                scores_dict["full_pseu_scores"][key].append(pseu_result[key])
                if entity_list_map:
                    scores_dict["pseu_ori_scores"][key].append(ori_result[key])
                    scores_dict["pseu_pseu_scores"][key].append(pseu_result[key])

            result_logger.info(log_str_fmt)

        # 记录中间结果
        if args.record_cache:
            next_cache_file_prefix = ""
            if args.run_step[0]:
                next_cache_file_prefix = cache_util.save_detect_result(
                    args, entity_list_map, cache_file_dir)
            if args.run_step[1]:
                next_cache_file_prefix = cache_util.save_generate_result(
                    args, candidate_entity_map, cache_file_dir, next_cache_file_prefix)
            if args.run_step[2]:
                next_cache_file_prefix = cache_util.save_replace_result(
                    args, pseu_item, cache_file_dir, next_cache_file_prefix)
            if args.run_eval[1]:
                cache_util.save_pseu_eval_result(args, pseu_output, cache_file_dir, next_cache_file_prefix)
            if args.run_step[3]:
                cache_util.save_recover_result(args, recover_output, cache_file_dir, next_cache_file_prefix)

    if len(next(iter(next(iter(scores_dict.values())).values()))) > 0:
        # 计算总结果
        score_list = score_util.compute_avg_score(scores_dict)

        # result.log 记录
        logging_util.add_result_log(result_logger, score_list)

    return


if __name__ == '__main__':

    # 开启性能分析
    profiler = cProfile.Profile()
    profiler.enable()

    # ========== ↓ ========== 通用资源初始化 ========== ↓ ========== #

    # 初始化参数解析器，读取脚本形参，控制验证流程
    args = argparse_util.init_parser()

    # 设置种子
    argparse_util.set_random_seed(args.seed)

    # 初始化日志执行器
    result_logger = logging_util.config_logger(args)
    result_logger.info("args: {}".format(json.dumps(vars(args), indent=4, ensure_ascii=False)))

    # 初始化中间结果文件(cache_file)路径
    cache_file_dir = os.path.join("Output", "Cache", args.dataset)
    os.makedirs(cache_file_dir, exist_ok=True)

    # 初始化配置对象，读取数据集/NLP任务配置文件
    dataset_config = config_util.Config(args.dataset_path)  # 数据集相关配置

    # 加载数据集，以及相应的评估指标
    dataset, dataset_length = dataset_util.load_dataset(args, dataset_config)

    # 过滤端到端有效数据
    if args.end2end:
        end_to_end_ids = json.load(open(os.path.join(cache_file_dir, "End2End_id_list.json"), "r", encoding="utf-8"))
        dataset = dataset.filter(lambda item: item["id"] in end_to_end_ids)

    # 加载评估模型
    if ((args.run_eval[0] and not args.fast_run_eval[0])
            or args.run_eval[1] and not args.fast_run_eval[1]):
        eval_tokenizer, eval_model = model_util.load_eval_model(args)

    # ========== ↓ ========== 模块资源初始化 ========== ↓ ========== #

    dataset_entity_dict = {}
    ner_model = None
    tool_model_tokenizer, tool_model = None, None
    repeat_model_tokenizer, repeat_model = None, None
    tag_model_tokenizer, tag_model = None, None

    if not args.base_run:

        if args.run_step[0] and not args.fast_run_step[0]:
            # Detection privacy entity 过程   seq -> entity list
            if args.detect_method == "ner" or args.detect_method.startswith("tag"):  # 使用专用 NER 模型识别
                assert args.ner_model
                ner_model = model_util.load_ner_model(args)
            elif args.detect_method == "prompt":  # 使用通用语言模型 + 提示工程识别
                assert args.tool_model
                tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
            if args.detect_method == "tag_mask" or args.detect_method == "tag_mark":  # 使用微调 tag_mark 或 tag_mask 模型在生成时识别
                assert args.tag_model
                tag_model_tokenizer, tag_model = model_util.load_tag_model(args)

        if args.run_step[1] and not args.fast_run_step[1]:
            # Generation candidate entity 过程   entity list -> entity map
            if args.generate_method == "rand":  # 从字典中随机选取
                dataset_entity_dict = dataset_util.load_entity_dict(args)
            elif args.generate_method == "prompt":  # 使用通用语言模型 + 提示工程生成
                assert args.tool_model
                tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
            elif args.generate_method == "top_k":  # 使用语言模型 next token top_k 特性生成
                assert args.tool_model
                tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
                assert args.tag_model
                tag_model_tokenizer, tag_model = model_util.load_tag_model(args)

        if args.run_step[2] and not args.fast_run_step[2]:
            # Replace privacy entity 过程
            if args.replace_method == "direct":
                pass  # 字符串替换
            elif args.replace_method == "prompt":  # 使用通用语言模型 + 提示工程替换
                assert args.tool_model
                tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
            elif args.replace_method == "gen":  # 使用 repeat 模型在生成过程中替换
                if args.detect_method == "tag_mask" or args.detect_method == "tag_mark":
                    assert args.tag_model
                    tag_model_tokenizer, tag_model = model_util.load_tag_model(args)
                else:
                    assert args.repeat_model
                    repeat_model_tokenizer, repeat_model = model_util.load_repeat_model(args)

        if args.run_step[3] and not args.fast_run_step[3]:
            # Recover privacy entity 过程
            if args.recover_method == "direct":
                if args.dataset == "wmt14-de-en":
                    assert args.tool_model
                    tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
                pass  # 字符串替换
            elif args.recover_method == "prompt":  # 使用通用语言模型 + 提示工程恢复
                assert args.tool_model
                tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
            elif args.recover_method == "gen":  # 使用 repeat 模型在生成过程中恢复
                if args.dataset == "wmt14-de-en":
                    assert args.tool_model
                    tool_model_tokenizer, tool_model = model_util.load_tool_model(args)
                assert args.repeat_model
                repeat_model_tokenizer, repeat_model = model_util.load_repeat_model(args)

        # 加载 sentence transformer embedding 模型
        embedding_model = model_util.load_embedding_model(args)

    # ========== ↓ ========== 开始评估 ========== ↓ ========== #

    with torch.inference_mode():
        result_logger.info("Run mode: {}".format("base run" if args.base_run else "pseu run"))
        run()

    # ===== 性能分析统计 ===== #

    profiler.disable()  # 结束 profiler 监控
    stats = pstats.Stats(profiler)  # 获取统计信息
    output = io.StringIO()  # 创建一个 StringIO 对象捕获输出
    stats.stream = output  # 输出信息重定位

    # 打印所需的统计信息到 StringIO 对象
    # stats.print_stats()  # 打印全部信息
    stats.strip_dirs().sort_stats("cumtime").print_stats(30)  # 打印 cumtime 最多的 30 个函数

    output_str = output.getvalue()  # 获取捕获的输出
    result_logger.info("Performance Stats:\n%s", output_str)  # 将输出记录到日志文件中
    output.close()  # 关闭 StringIO 对象
