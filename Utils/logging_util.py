import json
import logging
import os
from datetime import datetime


def init_logger(logger_name, log_file_path, format_str):
    # 创建logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 创建日志文件handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8", mode="a")
    fh.setFormatter(logging.Formatter(format_str))
    fh.setLevel(logging.INFO)

    # 将处理器注册到loggers
    logger.addHandler(fh)

    return logger


def config_logger(args):
    date_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    base_path = os.path.join("Output", "Log")
    dataset_log_path = os.path.join(base_path, args.dataset, date_stamp + "_" + args.comment)

    os.makedirs(dataset_log_path, exist_ok=True)

    result_log_path = os.path.join(dataset_log_path, "result.log")
    result_logger = init_logger(args.dataset + "_result", result_log_path, "%(asctime)s - %(message)s")

    return result_logger


def add_result_log(result_logger, score_list):
    comment_list = [
        "全数据集    [原始] 数据得分 : ",
        "全数据集    [匿名化] 数据得分 : ",
        "全数据集    性能损失 : ",
        "匿名化数据集 [原始] 数据得分 : ",
        "匿名化数据集 [匿名化] 数据得分 : ",
        "匿名化数据集 性能损失 : ",
    ]

    def gen_log_format(prefix, score_dict):
        log_format_str = prefix
        for key in score_dict.keys():
            log_format_str += (
                    key + " = %.2f   " % (score_dict[key] if score_dict[key] > 1 else score_dict[key] * 100))
        return log_format_str

    for comment, score in zip(comment_list, score_list):
        result_logger.info(gen_log_format(comment, score))

    result_logger.info("\n")


def add_badcase_log(badcase_logger, ori_result, saniti_result, dataset_config, item, sanitized_item, entity_map,
                    ori_outputs, raw_output, sanitized_outputs):
    score_keys = dataset_config.score_keys
    badcase_score_thresholds = dataset_config.badcase_score_thresholds
    is_badcase = True
    for key, score_threshold in zip(score_keys, badcase_score_thresholds):
        if (
                saniti_result[key] > score_threshold or
                ori_result[key] - saniti_result[key] < score_threshold / 2.0
        ):
            is_badcase = False
        else:
            break

    def dict_dump(input_dict):
        return json.dumps(input_dict, indent=4, ensure_ascii=False)

    if is_badcase:
        badcase_logger.info(" ========== badcase item block ==========")
        badcase_logger.info("ori_item:\n{}".format(dict_dump(item)))  # 原数据
        badcase_logger.info("sanitized_item:\n{}".format(dict_dump(sanitized_item)))  # 净化后数据
        badcase_logger.info("entity_map:\n{}".format(dict_dump(entity_map)))  # 实体映射
        badcase_logger.info("ori_outputs:\n{}".format(ori_outputs))  # 模型对原数据推理得到的最终结果
        badcase_logger.info("sanitized_outputs:\n{}".format(sanitized_outputs))  # 模型对净化数据推理得到的最终结果
        badcase_logger.info("model_raw_output:\n{}".format(raw_output))  # LLM 模型的推理原始输出
        badcase_logger.info("ori_result:\n{}".format(dict_dump(ori_result)))  # 原始数据结果得分
        badcase_logger.info("saniti_result:\n{}".format(dict_dump(saniti_result)))  # 净化数据结果得分

    return


def add_item_result(ori_result, san_result, entity_map, sanitized_item, pseu_efficiency, dataset_log_dir, ori_outputs,
                    sanitized_outputs):
    ori_result_path = os.path.join(dataset_log_dir, "ori_result.txt")
    san_result_path = os.path.join(dataset_log_dir, "san_result.txt")
    entity_map_path = os.path.join(dataset_log_dir, "entity_map.txt")
    sanitized_item_path = os.path.join(dataset_log_dir, "sanitized_item.txt")
    pseu_efficiency_path = os.path.join(dataset_log_dir, "pseu_efficiency.txt")
    model_output_ori_path = os.path.join(dataset_log_dir, "model_output_ori")
    model_output_san_path = os.path.join(dataset_log_dir, "model_output_san")

    with open(
            ori_result_path, 'a', encoding='utf-8') as ori_result_file, open(
        san_result_path, 'a', encoding='utf-8') as san_result_file, open(
        entity_map_path, 'a', encoding='utf-8') as entity_map_file, open(
        sanitized_item_path, 'a', encoding='utf-8') as sanitized_item_file, open(
        pseu_efficiency_path, 'a', encoding='utf-8') as pseu_efficiency_file, open(
        model_output_ori_path, 'a', encoding='utf-8') as model_output_ori_file, open(
        model_output_san_path, 'a', encoding='utf-8') as model_output_san_file:
        ori_result_file.write(json.dumps(ori_result, ensure_ascii=False) + '\n')
        san_result_file.write(json.dumps(san_result, ensure_ascii=False) + '\n')
        entity_map_file.write(json.dumps(entity_map, ensure_ascii=False) + '\n')
        sanitized_item_file.write(json.dumps(sanitized_item, ensure_ascii=False) + '\n')
        pseu_efficiency_file.write(json.dumps(pseu_efficiency, ensure_ascii=False) + '\n')
        model_output_ori_file.write(ori_outputs + '\n')
        model_output_san_file.write(sanitized_outputs + '\n')


def load_item_cache(dataset_log_dir):
    sanitized_item_record, entity_map_record, pseu_efficiency_record = [], [], []
    with open(os.path.join(dataset_log_dir, "sanitized_item.txt"), "r", encoding="utf-8") as sanitized_item_file:
        for line in sanitized_item_file:
            sanitized_item_record.append(json.loads(line.strip()))
    with open(os.path.join(dataset_log_dir, "entity_map.txt"), "r", encoding="utf-8") as entity_map_file:
        for line in entity_map_file:
            entity_map_record.append(json.loads(line.strip()))
    with open(os.path.join(dataset_log_dir, "pseu_efficiency.txt"), "r", encoding="utf-8") as pseu_efficiency_file:
        for line in pseu_efficiency_file:
            pseu_efficiency_record.append(json.loads(line.strip()))
    return sanitized_item_record, entity_map_record, pseu_efficiency_record
