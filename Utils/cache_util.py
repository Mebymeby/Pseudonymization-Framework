import ast
import json
import os


# ========== Save ========== #

def save_ori_eval_result(args, ori_eval_result, cache_file_dir):
    cache_path = os.path.join(cache_file_dir, "OriEval")
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    cache_file = os.path.join(cache_path, args.eval_model + ".txt")

    if not args.fast_run_eval[0]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ori_eval": ori_eval_result}, ensure_ascii=False) + '\n')


def save_detect_result(args, entity_list, cache_file_dir):
    cache_path = os.path.join(cache_file_dir, "Detect")  # 路径: Output/Cache/[数据集名称]/Detect/
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    cache_file = os.path.join(cache_path, args.detect_method + ".txt")
    next_cache_file_prefix = args.detect_method + "-"

    if not args.fast_run_step[0]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entity_list, ensure_ascii=False) + '\n')

    return next_cache_file_prefix


def save_generate_result(args, entity_map, cache_file_dir, next_cache_file_prefix):
    cache_path = os.path.join(cache_file_dir, "Generate")  # 路径: Output/Cache/[数据集名称]/Generate/
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    if args.generate_method == "prompt":
        cache_file = os.path.join(cache_path, next_cache_file_prefix + args.generate_method + "_"
                                  + args.llm_generate_entity_prompt_input + ".txt")
        next_cache_file_prefix += args.generate_method + "_" + args.llm_generate_entity_prompt_input + "-"
    else:
        cache_file = os.path.join(cache_path, next_cache_file_prefix + args.generate_method + ".txt")
        next_cache_file_prefix += args.generate_method + "-"

    if not args.fast_run_step[1]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entity_map, ensure_ascii=False) + '\n')

    return next_cache_file_prefix


def save_replace_result(args, pseu_item, cache_file_dir, next_cache_file_prefix):
    cache_path = os.path.join(cache_file_dir, "Replace")  # 路径: Output/Cache/[数据集名称]/Replace/
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    cache_file = os.path.join(cache_path, next_cache_file_prefix + args.replace_method + ".txt")
    next_cache_file_prefix += args.replace_method + "-"

    if not args.fast_run_step[2]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(pseu_item, ensure_ascii=False) + '\n')

    return next_cache_file_prefix


def save_pseu_eval_result(args, pseu_eval_result, cache_file_dir, next_cache_file_prefix):
    cache_path = os.path.join(cache_file_dir, "PseuEval")
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    cache_file = os.path.join(cache_path, next_cache_file_prefix + "pseu_eval.txt")

    if not args.fast_run_eval[1]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pseu_eval": pseu_eval_result}, ensure_ascii=False) + '\n')


def save_recover_result(args, recover_output, cache_file_dir, next_cache_file_prefix):
    cache_path = os.path.join(cache_file_dir, "Recover")  # 路径: Output/Cache/[数据集名称]/Recover/
    if args.data_split == "train":
        cache_path = os.path.join(cache_path, "Train")
    os.makedirs(cache_path, exist_ok=True)

    cache_file = os.path.join(cache_path, next_cache_file_prefix + args.recover_method + ".txt")

    if not args.fast_run_step[3]:
        with open(cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"recover": recover_output}, ensure_ascii=False) + '\n')


# ========== Load ========== #

def load_ori_eval_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "OriEval")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    cache_file = os.path.join(cache_path, args.eval_model + ".txt")
    return literal_file(cache_file)


def load_detect_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "Detect")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    cache_file = os.path.join(cache_path, args.detect_method + ".txt")
    return literal_file(cache_file)


def load_generate_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "Generate")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    if args.generate_method == "prompt":
        cache_file = os.path.join(cache_path, args.detect_method + "-" + args.generate_method + "_"
                                  + args.llm_generate_entity_prompt_input + ".txt")
    else:
        cache_file = os.path.join(cache_path, args.detect_method + "-" + args.generate_method + ".txt")
    return literal_file(cache_file)


def load_replace_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "Replace")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    if args.generate_method == "prompt":
        cache_file = os.path.join(cache_path, args.detect_method + "-" + args.generate_method + "_"
                                  + args.llm_generate_entity_prompt_input + "-" + args.replace_method + ".txt")
    else:
        cache_file = os.path.join(cache_path,
                                  args.detect_method + "-" + args.generate_method + "-" + args.replace_method + ".txt")
    return literal_file(cache_file)


def load_pseu_eval_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "PseuEval")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    if args.generate_method == "prompt":
        cache_file = os.path.join(cache_path, args.detect_method + "-" + args.generate_method + "_"
                                  + args.llm_generate_entity_prompt_input + "-" + args.replace_method + "-pseu_eval.txt")
    else:
        cache_file = os.path.join(cache_path,
                                  args.detect_method + "-" + args.generate_method + "-" + args.replace_method + "-pseu_eval.txt")
    return literal_file(cache_file)


def load_recover_result(args):
    cache_path = os.path.join("Output", "Cache", args.dataset, "Recover")
    if args.end2end:
        cache_path = os.path.join(cache_path, "End2End")
    if args.generate_method == "prompt":
        cache_file = os.path.join(cache_path,
                                  args.detect_method + "-" + args.generate_method + "_" + args.llm_generate_entity_prompt_input
                                  + "-" + args.replace_method + "-" + args.recover_method + ".txt")
    else:
        cache_file = os.path.join(cache_path,
                                  args.detect_method + "-" + args.generate_method + "-" + args.replace_method + "-" + args.recover_method + ".txt")
    return literal_file(cache_file)


def literal_file(cache_file):
    with open(cache_file, "r", encoding="utf-8") as f:
        cache_result = [ast.literal_eval(line.strip()) for line in f]

    return cache_result
