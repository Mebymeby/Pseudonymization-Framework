import difflib
import json
import random
import re

import torch
from sentence_transformers import util

from Config import prompt
from Utils import infer_util, score_util

# 实体类型全称映射
entity_type_full = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
}


# ===== ↓ ===== 方案一 ===== ↓ ===== #

# 候选实体与原实体不可有包含关系
def dup_verify(ori_entity, candidate_entity):
    return (candidate_entity not in ori_entity) and (ori_entity not in candidate_entity)


def get_one_entity_by_ner_dict_select(ori_entity, entity_dict, entity_type, embedding_model):
    ori_entity_embedding = embedding_model.encode(ori_entity, convert_to_tensor=True)
    candidate_entity = random.choice(entity_dict[entity_type])
    candidate_entity_embedding = embedding_model.encode(candidate_entity, convert_to_tensor=True)
    ori_candidate_sim = util.cos_sim(ori_entity_embedding, candidate_entity_embedding).squeeze(0).item()

    best_sim = ori_candidate_sim  # 记录最佳相似度
    best_candidate_entity = candidate_entity  # 记录最佳候选实体
    select_count = 2  # 选择上限，超出时返回最佳（select_count 与 ori_candidate_sim 数值较大时非常影响速度）
    while select_count and ((not dup_verify(ori_entity, best_candidate_entity)) or ori_candidate_sim < 0.2):
        select_count -= 1
        candidate_entity = random.choice(entity_dict[entity_type])
        candidate_entity_embedding = embedding_model.encode(candidate_entity, convert_to_tensor=True)
        ori_candidate_sim = util.cos_sim(ori_entity_embedding, candidate_entity_embedding).squeeze(0).item()
        if ori_candidate_sim > best_sim:  # 更新最佳候选
            best_sim = ori_candidate_sim
            best_candidate_entity = candidate_entity

    return best_candidate_entity


# 获得候选实体映射字典
def get_entity_map_by_ner_dict_select(ner_result, entity_dict, embedding_model):
    entity_map = {}
    for entity in ner_result:
        ori_entity = entity["word"]
        ori_entity_embedding = embedding_model.encode(ori_entity, convert_to_tensor=True)  # 提前计算
        entity_type = entity["entity_group"]
        if ori_entity not in entity_map:
            candidate_entity = random.choice(entity_dict[entity_type])
            candidate_entity_embedding = embedding_model.encode(candidate_entity, convert_to_tensor=True)
            ori_candidate_sim = util.cos_sim(ori_entity_embedding, candidate_entity_embedding).squeeze(0).item()
            best_candidate_entity = candidate_entity  # 记录最佳候选实体
            best_sim = ori_candidate_sim  # 记录最佳相似度
            select_count = 2  # 选择上限，超出时返回最佳（select_count 与 ori_candidate_sim 数值较大时非常影响速度）
            while select_count and ((not dup_verify(ori_entity, best_candidate_entity)) or ori_candidate_sim < 0.2):
                select_count -= 1
                candidate_entity = random.choice(entity_dict[entity_type])
                candidate_entity_embedding = embedding_model.encode(candidate_entity, convert_to_tensor=True)
                ori_candidate_sim = util.cos_sim(ori_entity_embedding, candidate_entity_embedding).squeeze(0).item()
                if ori_candidate_sim > best_sim:  # 更新最佳候选
                    best_sim = ori_candidate_sim
                    best_candidate_entity = candidate_entity
            entity_map[ori_entity] = best_candidate_entity
    return entity_map


# 通过候选实体映射字典，使用字符串替换方式，替换或恢复原文本
def str_replace_with_entity_map(entity_map, seq, direction):
    replaced_seq = seq
    for key, value in entity_map.items():
        if direction == "kv":
            replaced_seq = replaced_seq.replace(key, value)  # seq: ｛key｝ --> {value}
        if direction == "vk":
            replaced_seq = replaced_seq.replace(value, key)  # seq: {value} --> ｛key｝
    return replaced_seq


# ===== ↓ =====  方案二，以及一些工具函数  ===== ↓ ===== #

# 通过 prompt 指导 llm 获取单个实体
def get_entity_by_llm(args, ori_entity, entity_type, tool_model, tool_tokenizer):
    if entity_type != "entity":
        entity_type = entity_type_full[entity_type]

    # 拼接 prompt
    if args.gen_entity_input == "entity_and_type_JSON":
        gen_entity_prompt = prompt.TASK["generate_with_entity_and_type_JSON"].format(ori_entity, entity_type)
    elif args.gen_entity_input == "entity_and_type":
        gen_entity_prompt = prompt.TASK["generate_with_entity_and_type"].format(ori_entity, entity_type)
    elif args.gen_entity_input == "entity":
        gen_entity_prompt = prompt.TASK["generate_with_entity"].format(ori_entity)
    elif args.gen_entity_input == "type":
        gen_entity_prompt = prompt.TASK["generate_with_type"].format(entity_type)
    else:
        exit(1)

    # 执行推理
    candidate_entity = infer_util.prompt_inference(tool_model, tool_tokenizer, gen_entity_prompt, 32).strip()

    # 解码输出
    if args.gen_entity_input == "entity_and_type_JSON":
        candidate_entity = infer_util.extract_llm_output(candidate_entity)

    # 前置空格拼接
    if ori_entity.startswith(" "):
        candidate_entity = " " + candidate_entity

    return candidate_entity


# 通过 prompt 指导 llm 获取实体映射字典
def get_entity_map_by_llm(args, ner_result, tool_model, tool_tokenizer):
    entity_map = {}
    for entity in ner_result:
        ori_entity = entity["word"]
        entity_type = entity["entity_group"]
        if ori_entity not in entity_map:
            entity_map[ori_entity] = get_entity_by_llm(args, ori_entity, entity_type, tool_model, tool_tokenizer)
    return entity_map


# 端到端实现 ner、replace、generate过程
def sanitize_end2end(input_text, model, tokenizer, ner_key):
    if ner_key == "entity":
        entity_type_full_str = "PER (person), LOC (location), ORG (organization)"
    else:
        entity_type_full_str = ner_key + " (" + entity_type_full[ner_key] + ")"

    # 获取prompt
    end2end_prompt = prompt.TASK["llm_end2end"].format(input_text, entity_type_full_str)

    # 执行推理
    output = infer_util.prompt_inference(model, tokenizer, end2end_prompt, 4096)

    # 获取实体字典
    entity_diff = find_replace_list(input_text, output)

    return output, entity_diff


def sanitize_end2end_step1_detect(input_text, model, tokenizer, entity_type_full_str):
    step1_prompt = prompt.TASK["llm_end2end_detect"].format(input_text, entity_type_full_str)

    output = infer_util.prompt_inference(model, tokenizer, step1_prompt, 512).strip()

    try:
        entity_list = json.loads(output)  # 实体列表的字符串形式
    except json.decoder.JSONDecodeError:
        print(f"Can't decode entity list by json")
        return []

    return entity_list


def sanitize_end2end_step3_replace(input_text, entity_map, model, tokenizer):
    step3_prompt = prompt.TASK["llm_end2end_replace"].format(input_text, entity_map)

    output = infer_util.prompt_inference(model, tokenizer, step3_prompt, 4096).strip()

    return output


def sanitize_end2end_step_prompt(args, input_text, model, tokenizer, ner_key):
    if ner_key == "entity":
        entity_type_full_str = "PER (person), LOC (location), ORG (organization)"
    else:
        entity_type_full_str = ner_key + " (" + entity_type_full[ner_key] + ")"

    # 1、Detection
    entity_list = sanitize_end2end_step1_detect(input_text, model, tokenizer, entity_type_full_str)

    # 2、Generate map
    entity_map = {}
    for ori_entity in entity_list:
        if ori_entity not in entity_map:
            entity_map[ori_entity] = get_entity_by_llm(args, ori_entity, ner_key, model, tokenizer)

    # 3、replace
    output_text = sanitize_end2end_step3_replace(input_text, entity_map, model, tokenizer)

    return output_text, entity_map


# ===== ↓ =====  方案三  ===== ↓ ===== #

# 初始化replace过程的中间信息
def init_replace_info(input_text, entity_info, rep_gen_tokenizer, action):
    # 当前样本 replace 信息记录结构体
    replace_info = {
        "entity_token": [],  # 待替换的实体的token形式
        "entity_word" : [],  # 待替换的实体本身
        "entity_type" : [],  # 待替换的实体的类型
        "replace_map" : {}  # 当前过程的替换映射记录
    }

    # 当前样本的所有待替换实体列表
    if action == "saniti":
        entity_list = [entity["word"] for entity in entity_info]  # entity_info 为 ner 模型得到的实体列表[{},...]
    else:
        entity_list = entity_info.keys()  # entity_info 为 recover_entity_map, key 为 被替换的 token 的字符串形式

    # 去掉 token 的前导空格符 Ġ ; Hello world! -> ["Hello","Ġworld","!"] -> ["hello","world","!"]
    tokenized_input_text = [x.replace("Ġ", " ") for x in rep_gen_tokenizer.tokenize(input_text)]
    for id_start, token in enumerate(tokenized_input_text):
        token_lower = token.strip().lower()
        for entity_idx, entity in enumerate(entity_list):  # 判断当前token是否需要被替换

            ## 方案一: 贪心匹配
            entity_lower = entity.strip().lower()
            if entity_lower.startswith(token_lower):
                # 双指针,匹配同一个实体分布在多个token中的情况
                for id_end in range(id_start, len(tokenized_input_text)):
                    token_temp = "".join(tokenized_input_text[id_start:id_end + 1]).lower().strip()
                    if not entity_lower.startswith(token_temp):
                        break
                matched_token = "".join(tokenized_input_text[id_start:id_end]).lower().strip()
                entity_token = tokenized_input_text[id_start:id_end]
                if matched_token == entity_lower and entity_token not in replace_info["entity_token"]:
                    replace_info["entity_token"].append(entity_token)
                    replace_info["entity_word"].append(entity)
                    if action == "saniti":
                        replace_info["entity_type"].append(entity_info[entity_idx]["entity_group"])

    for entity_idx, entity in enumerate(entity_list):
        ## 方案二: 直接对 entity 进行 token 化
        entity_token = [x.replace("Ġ", " ") for x in rep_gen_tokenizer.tokenize(entity)]
        if entity_token not in replace_info["entity_token"]:
            replace_info["entity_token"].append(entity_token)
            replace_info["entity_word"].append(entity)
            if action == "saniti":
                replace_info["entity_type"].append(entity_info[entity_idx]["entity_group"])

        ## 方案三: 直接用空格分词
        entity_token = [x.strip() for x in entity.split()]
        if entity_token not in replace_info["entity_token"]:
            replace_info["entity_token"].append(entity_token)
            replace_info["entity_word"].append(entity)
            if action == "saniti":
                replace_info["entity_type"].append(entity_info[entity_idx]["entity_group"])

    return replace_info


# 判断当前生成的next_token是否在输入的替换字典中，如果在，则做贪心匹配返回匹配信息
def need_replace(next_token, replace_info, repeat_model, repeat_tokenizer, repeat_input_prompt):
    for need_replace_entity_idx, one_entity_token_list in enumerate(replace_info["entity_token"]):
        matched = True
        for one_token in one_entity_token_list:  # 贪心匹配当前生成的next_token是否在已保存的需要替换的实体token列表中
            if one_token.strip().lower() == next_token.strip().lower():
                repeat_input_prompt += next_token
                next_token, _ = infer_util.next_token_inference(repeat_input_prompt, repeat_model, repeat_tokenizer, )
            else:
                matched = False
                break
        if matched:
            return need_replace_entity_idx
    return -1


# 从topk中选择合适的词汇作为next_token
def choose_from_top_k(args, next_token, tokenizer, next_token_logits):
    stopword_list = ["the", "", ",", "\"", "\'", ".", "...", "input"],

    # 获取topk的词
    next_token_logits_top_k, next_token_id_top_k = torch.topk(next_token_logits, k=args.top_k_range, largest=True,
                                                              sorted=True)
    # 合法性判断
    for temp_next_token_id in next_token_id_top_k:
        temp_next_token = tokenizer.decode(temp_next_token_id, skip_special_token=True).strip()
        if temp_next_token.lower() in stopword_list:
            continue  # 停用词，舍弃
        elif temp_next_token.lower() == next_token.lower():
            continue  # 原 token 的等价变形，舍弃
        elif temp_next_token.startswith("\"") or temp_next_token.startswith("\'") or temp_next_token.startswith(".") \
                or temp_next_token.startswith("_") or temp_next_token.startswith("<"):
            continue  # 无意义开头，舍弃
        elif len(temp_next_token) <= 1:
            continue  # 只有一个字母影响太大，也进行舍弃

        return temp_next_token

    return next_token  # 没有可用候选值时返回原值，即不进行替换


# 获得生成式替换过程中的下一个生成的token；先生成，再判断是否需要替换，再直接返回或执行替换
def get_next_token(args, replace_info, repeat_input_prompt, tool_model, tool_tokenizer, repeat_model, repeat_tokenizer,
                   action, entity_dict, embedding_model):
    # 获得 next_token（带空格） 以及对应的 logits 分布
    next_token, next_token_logits = infer_util.next_token_inference(repeat_input_prompt, repeat_model, repeat_tokenizer)

    # 判断当前生成的 next_token 是否需要被替换
    replace_idx = need_replace(next_token, replace_info, repeat_model, repeat_tokenizer, repeat_input_prompt)
    if replace_idx != -1:
        # 待替换的原始实体
        ori_entity = replace_info["entity_word"][replace_idx]

        if action == "saniti":
            entity_type = replace_info["entity_type"][replace_idx]  # 待替换的实体类型
        else:
            entity_type = ""

        # 正反一致性保证
        if ori_entity in replace_info["replace_map"]:  # 当前待替换token对应的实体已经被替换过，则替换为相同的token
            next_token = replace_info["replace_map"][ori_entity]
        elif action == "saniti":
            if args.gen_next_token_type == "top_k":
                next_token = choose_from_top_k(args, next_token, repeat_tokenizer, next_token_logits)
            elif args.gen_next_token_type == "prompt":
                next_token = get_entity_by_llm(args, ori_entity, entity_type, tool_model, tool_tokenizer)
            elif args.gen_next_token_type == "dict_select":
                next_token = get_one_entity_by_ner_dict_select(ori_entity, entity_dict, entity_type, embedding_model)
            replace_info["replace_map"].update({ori_entity: next_token})  # 更新映射信息
        else:
            return ori_entity

    return next_token


# 非 finetune llm 模型生成式替换；入口
def llm_rep_gen_with_entity_map(args, input_text, entity_info, tool_model, tool_tokenizer,
                                repeat_model, repeat_tokenizer, replace_map, action, entity_dict, embedding_model):
    repeat_input_prompt = prompt.TASK["repeat_prompt"].format(input_text)

    replace_info = init_replace_info(input_text, entity_info, repeat_tokenizer, action)

    # 一致性保证
    if action == "saniti":  # 正向记录：原始实体 -> 替换实体的token
        replace_info["replace_map"].update(replace_map)  # entity_info 为 ner 模型识别的实体列表 [{},...]
    elif action == "recover":  # 反向记录：替换实体的token -> 原始实体
        replace_info["replace_map"].update(entity_info)  # entity_info 为 recover_entity_map

    # 逐个生成token，并在生成过程中做处理
    output_text = ""
    output_length = len(repeat_tokenizer.tokenize(repeat_input_prompt))
    for _ in range(output_length):
        next_token = get_next_token(args, replace_info, repeat_input_prompt, tool_model, tool_tokenizer, repeat_model,
                                    repeat_tokenizer, action, entity_dict, embedding_model)

        # 提前结束
        if args.early_stop and need_early_stop(next_token):
            break

        # 准备下一次的生成处理
        repeat_input_prompt += next_token
        output_text += next_token

    # 去除生成时的提示语
    if args.output_text_head:
        output_text = output_text.replace(args.output_text_head, "").strip()

    return output_text, replace_info["replace_map"]


# 判断是否进行提前退出
def need_early_stop(next_token):
    for early_stop_flag in ["<|im_end|>", "<|endoftext|>", "Output", "<line><line>"]:
        if next_token.strip() == early_stop_flag:
            return True
    return False


# ===== ↓ =====  方案四  ===== ↓ ===== #

# 判断finetune ner_replace模型的输出是否需要替换
def need_replace_ft(output_text, input_text, tokenizer, args, final=False):
    replace_word = ""
    replace_part = ""

    # 为了加快判断速度，可能可以先加一个简短判断
    # 根据是否是最后一次进行略微不同的判断
    if final or args.search_suffix_thresold <= 0:
        output_text_temp = output_text
    else:
        output_text_temp = output_text[:-args.search_suffix_thresold]

    if args.ner_gen_model_type == "tag_wrap" and (output_text_temp.count("<") < 2 or output_text_temp.count(">") < 2):
        return output_text, replace_word, replace_part
    elif args.ner_gen_model_type == "only_rep" and (not "<" in output_text_temp or not ">" in output_text_temp):
        return output_text, replace_word, replace_part

    # 文本需要中包含占位符
    if not ("<" in output_text_temp and ">" in output_text_temp):
        return output_text, replace_word, replace_part
    # 文本中必须要求标记占位符不能是结尾
    if output_text_temp.endswith(">"):
        return output_text, replace_word, replace_part

    # 正式进行处理
    if args.ner_gen_model_type == "tag_wrap":
        # 用正则表达式来处理
        matches = re.finditer(r"<(.*?)>(.*?)</\1>", output_text)
        for match in matches:
            # 如果正确的话，应该只有一个匹配上
            replace_tag = match.group(1)
            replace_word = match.group(2)
            replace_part = "<{0}>{1}</{0}>".format(replace_tag, replace_word)
    elif args.ner_gen_model_type == "only_rep":
        # 为了保证后续能够获得被计算的实体词汇
        # 用正则表达式来处理
        matches = re.finditer(r"<(.*?)>", output_text)
        for match in matches:
            replace_tag = match.group(1)
            replace_part = "<{0}>".format(replace_tag)
        # 通过对比输入和输出的文本，计算此处被替换的实体词汇
        replace_word = search_replace_word(output_text, input_text, replace_part, tokenizer, args)
    else:
        exit(1)

    return output_text, replace_word, replace_part


# 对比输入和输出，获取被替换的词汇
def search_replace_word(output_text, input_text, replace_holder, tokenizer, args):
    if args.output_text_head != "":
        output_text = output_text.replace(args.output_text_head, "").strip()

    replace_word = ""

    # 先对比输入和输出，获取其中不相同的部分
    # 分词处理
    tokenized_input_text = [x for x in tokenizer.tokenize(input_text.replace("\n", " "))]
    tokenized_output_text = [x for x in tokenizer.tokenize(output_text.replace("\n", " "))]

    # 防止干扰，先调整一下input_text
    tokenized_input_text = tokenized_input_text[:int(len(tokenized_output_text) * (1 + args.search_range))]

    diff_dict = process_diff_map(tokenized_output_text, tokenized_input_text)

    # 找到其中对应上replace_holder的部分
    for diff_key in diff_dict.keys():
        if replace_holder in diff_key or replace_holder == diff_key:
            # 初步找到替换位置
            diff_value = diff_dict[diff_key]
            if replace_holder == diff_key:
                replace_word = diff_value
            else:
                # 进一步找到其中具体替换的部分
                # 根据长度进行不同的处理
                if len(diff_value) - len(diff_key) <= args.search_diff_length:
                    # 如果两者长度相差不大
                    if diff_key.startswith("<"):
                        left_pos = 0
                    else:
                        left_pos = diff_key.find("<")
                    if diff_key.endswith(">"):
                        replace_word = diff_value[left_pos:]
                    else:
                        diff_key_right_pos = diff_key.find(">")
                        diff_key_right = diff_key[(diff_key_right_pos + 1):]
                        diff_value_right_pos = diff_value.find(diff_key_right)
                        replace_word = diff_value[left_pos:diff_value_right_pos]
                else:
                    # 如果相差太大
                    diff_value = diff_value[:(len(diff_key) + args.search_diff_length)]
                    char_diff_dict = process_diff_map(diff_value, diff_key)

                    start_str, end_str = "", ""
                    for char_key in char_diff_dict.keys():
                        if char_key == replace_holder:
                            replace_word = char_diff_dict[char_key]
                            break
                        elif char_key.startswith("<"):
                            start_str = char_diff_dict[char_key].replace("+", " ")
                        elif char_key.endswith(">"):
                            end_str = char_diff_dict[char_key].replace("+", " ")
                        if start_str != "" and end_str != "":
                            start_pos = diff_value.find(start_str)
                            end_pos = diff_value.find(end_str)
                            replace_word = diff_value[start_pos:end_pos] + end_str
                            break

                    replace_word = replace_word.replace("+", " ")

            # 匹配上一个即可，
            break

    return replace_word


# 判断是否进行提前退出
def need_early_stop_ft(next_token, output_text):
    for early_stop_flag in ["<|im_end|>", "<|endoftext|>", "Output", "<line><line>"]:
        # 换行符采用<line>输入
        early_stop_flag = early_stop_flag.replace("<line>", "\n")
        # next_token本身是终止词
        if next_token.strip() == early_stop_flag:
            return output_text
        # 加上next_token之后文本包含了终止词
        output_text_temp = output_text + next_token
        if early_stop_flag in output_text_temp:
            output_text = output_text_temp.replace(early_stop_flag, "")
            return output_text
    return ""


# 找到前文部分
def find_prefix_output(ori_word, output_text):
    last_index = output_text.rfind(ori_word)
    return output_text[:last_index]


def ft_llm_rep_gen(args, input_text, tool_model, tool_tokenizer, ner_gen_model, ner_gen_tokenizer, ner_key):
    output_text = ""  # 最终文本
    entity_map = {}  # 记录实体映射

    ner_gen_prompt = ""
    if args.ner_gen_model_type == "only_rep":
        ner_gen_prompt = prompt.TASK["only_rep_prompt"].format(input_text)
    if args.ner_gen_model_type == "tag_wrap":
        ner_gen_prompt = prompt.TASK["tag_wrap_prompt"].format(input_text)

    # 逐个生成token，并在生成过程中做处理
    output_length = len(ner_gen_tokenizer.tokenize(ner_gen_prompt))

    for _ in range(min(4096, output_length)):
        next_token, _ = infer_util.next_token_inference(ner_gen_prompt, ner_gen_model, ner_gen_tokenizer)

        # 提前结束
        if args.early_stop:
            stop_flag = need_early_stop_ft(next_token, output_text)
            if stop_flag != "":
                output_text = stop_flag
                break

        # 准备下一次的生成处理
        ner_gen_prompt += next_token
        output_text += next_token

        # 判断是否需要进行替换
        output_text, ori_word, replace_part = need_replace_ft(output_text, input_text, ner_gen_tokenizer, args)
        if ori_word != "":  # 需要替换,生成替换值

            # 情形一: 当前词已经替换过
            if ori_word in entity_map and entity_map[ori_word] != "":
                generate_word = entity_map[ori_word]

            # 情形二
            elif args.ner_gen_model_type == "tag_wrap" and (
                    args.search_replace_constraint == "dict" and ori_word not in entity_map) or (
                    args.search_replace_constraint == "context" and ori_word in find_prefix_output(ori_word,
                                                                                                   output_text)):
                generate_word = ori_word

            # 情形三: 直接用 prompt 生成
            else:
                generate_word = get_entity_by_llm(args, ori_word, ner_key, tool_model, tool_tokenizer).strip()
                # 合法性校验，不通过则再次生成
                count = 3  # 避免死循环
                while count > 0 and (generate_word in input_text or generate_word in output_text):  # 拿生成词再生成一个
                    generate_word = get_entity_by_llm(args, generate_word, ner_key, tool_model, tool_tokenizer).strip()
                    count -= 1
                entity_map.update({ori_word: generate_word})

            # 执行替换
            output_text = output_text.replace(replace_part, generate_word)
            # 更新一下提示文本
            if args.ner_gen_model_type == "only_rep":
                ner_gen_prompt = prompt.TASK["only_rep_prompt"].format(input_text) + output_text
            if args.ner_gen_model_type == "tag_wrap":
                ner_gen_prompt = prompt.TASK["tag_wrap_prompt"].format(input_text) + output_text

    # 结束之前还需要进行一次替换判断
    output_text, ori_word, replace_part = need_replace_ft(output_text, input_text, ner_gen_tokenizer, args)
    if ori_word != "":  # 需要替换,生成替换值

        # 情形一: 当前词已经替换过
        if ori_word in entity_map and entity_map[ori_word] != "":
            generate_word = entity_map[ori_word]

        # 情形二
        elif args.ner_gen_model_type == "tag_wrap" and (
                args.search_replace_constraint == "dict" and ori_word not in entity_map) or (
                args.search_replace_constraint == "context" and ori_word in find_prefix_output(ori_word,
                                                                                               output_text)):
            generate_word = ori_word

        # 情形三: 直接用 prompt 生成
        else:
            generate_word = get_entity_by_llm(args, ori_word, ner_key, tool_model, tool_tokenizer).strip()
            # 合法性校验，不通过则再次生成
            count = 3  # 避免死循环
            while count > 0 and (generate_word in input_text or generate_word in output_text):  # 拿生成词再生成一个
                generate_word = get_entity_by_llm(args, generate_word, ner_key, tool_model, tool_tokenizer).strip()
                count -= 1
            entity_map.update({ori_word: generate_word})

        # 执行替换
        output_text = output_text.replace(replace_part, generate_word)

    # 整理输出结果
    if args.output_text_head != "":
        output_text = output_text.replace(args.output_text_head, "").strip()

    return output_text, entity_map


# ========== 映射字典生成函数 ========== #

# 生成翻译数据的德语映射
def gen_de_map(ori_sentence, san_sentence, tool_model, tool_tokenizer):
    ori_trans_prompt = prompt.DATASET["wmt14-de-en"].format(ori_sentence)
    san_trans_prompt = prompt.DATASET["wmt14-de-en"].format(san_sentence)

    ori_trans_res = infer_util.prompt_inference(tool_model, tool_tokenizer, ori_trans_prompt, len(ori_sentence) * 1.2)
    san_trans_res = infer_util.prompt_inference(tool_model, tool_tokenizer, san_trans_prompt, len(san_sentence) * 1.2)

    return find_replace_list(ori_trans_res, san_trans_res)


# 方案三在用
def find_replace_list(sentence_a, sentence_b):
    words_a = sentence_a.split()
    words_b = sentence_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)

    replace_map = {}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':  # 处理替换的部分
            key = ' '.join(words_a[i1:i2])  # 原句的部分
            value = ' '.join(words_b[j1:j2])  # 替换后的部分

            # 处理映射，支持多个词的替换
            if key in replace_map:
                # 避免重复，确保是列表
                if isinstance(replace_map[key], list):
                    replace_map[key].append(value)
                else:
                    replace_map[key] = [replace_map[key], value]
            else:
                replace_map[key] = value

    # 确保键值对均为字符串类型
    replace_map = {key: " ".join(value) if isinstance(value, list) else value for key, value in replace_map.items()}

    return replace_map


# 方案四在用
def process_diff_map(tokenized_ori_text, tokenized_sanitized_text):
    diff_map = {}

    # 比对
    d = difflib.Differ()
    diff = d.compare(tokenized_ori_text, tokenized_sanitized_text)
    diff_list = [x for x in diff]

    # 输出
    ori_token, replaced_token = "", ""
    for token in diff_list:
        if token.startswith("-"):
            replaced_token += "" + token.split()[-1]
        elif token.startswith("+"):
            ori_token += "" + token.split()[-1]
        elif token.startswith("?"):
            pass
        elif ori_token and replaced_token:
            diff_map[replaced_token.replace("Ġ", " ").strip()] = ori_token.replace("Ġ", " ").strip()
            ori_token, replaced_token = "", ""
        else:
            ori_token, replaced_token = "", ""

    if ori_token and replaced_token:
        diff_map[replaced_token.replace("Ġ", " ").strip()] = ori_token.replace("Ġ", " ").strip()

    return diff_map


# ===== ↓ =====  sanitization 主函数  ===== ↓ ===== #

# 主函数：输入源 item, 输出 sanitized_item 与实体映射字典
def sanitize_item(args, item, entity_dict, dataset_config, ner_model, ner_key, threshold, embedding_model,
                  tool_model, tool_tokenizer, repeat_model, repeat_tokenizer, ner_gen_model, ner_gen_tokenizer):
    sanitized_item = item.copy()

    # 获取item中所有需要净化的部分（LLM 输入部分）
    all_input = ""
    for key in dataset_config.input_keys:
        all_input += item[key] + " "

    if args.san_method == "end2end":
        key = dataset_config.input_keys[0]
        sanitized_item[key], entity_map = sanitize_end2end(sanitized_item[key], tool_model, tool_tokenizer, ner_key)
        if len(dataset_config.input_keys) > 1 and entity_map:  # 针对squad、squad_v2等问答数据的"question"部分
            for key in dataset_config.input_keys[1:]:  # 直接用已经生成的字典对"question"进行字符串替换
                sanitized_item[key] = str_replace_with_entity_map(entity_map, sanitized_item[key], "kv")
    elif args.san_method == "end2end_step_prompt":
        key = dataset_config.input_keys[0]
        sanitized_item[key], entity_map = sanitize_end2end_step_prompt(args, sanitized_item[key], tool_model,
                                                                       tool_tokenizer, ner_key)
        if len(dataset_config.input_keys) > 1 and entity_map:  # 针对squad、squad_v2等问答数据的"question"部分
            for key in dataset_config.input_keys[1:]:  # 直接用已经生成的字典对"question"进行字符串替换
                sanitized_item[key] = sanitize_end2end_step3_replace(sanitized_item[key], entity_map, tool_model,
                                                                     tool_tokenizer)
    else:
        # ===== ↓ ===== ner 过程 ===== ↓ ===== #
        if args.ner_model:  # 通过ner模型获取实体识别结果
            ner_result = ner_model(all_input)
            if ner_key == "entity":  # entity 代表实验不区分实体类型
                ner_result = [entity for entity in ner_result if
                              entity["entity_group"] in ["PER", "LOC", "ORG"] and entity["score"] >= threshold]
            else:
                ner_result = [entity for entity in ner_result if
                              entity["entity_group"] in ner_key and entity["score"] >= threshold]
        else:
            ner_result = []

        # ===== ↓ ===== 生成 entity_map 过程 ===== ↓ ===== #
        if args.entity_map_method == "ner_dict_select":  # 通过实体字典选取候选实体
            entity_map = get_entity_map_by_ner_dict_select(ner_result, entity_dict, embedding_model)
        elif args.entity_map_method == "llm_generate":  # 通过 prompt 指导工具模型生成候选实体
            entity_map = get_entity_map_by_llm(args, ner_result, tool_model, tool_tokenizer)
        else:
            entity_map = {}

        # ===== ↓ ===== replace/generate 过程 ===== ↓ ===== #
        if args.rep_gen_method == "str_replace":  # 通过字符串替换生成净化item
            for key in dataset_config.input_keys:
                sanitized_item[key] = str_replace_with_entity_map(entity_map, sanitized_item[key], "kv")
        elif args.rep_gen_method == "llm_generate":  # 方案三
            replace_map = {}  # 不同的item[key]做处理时的替换一致性保证，如squad中，context中的张三换成了李四，则question中要保持一致替换
            for key in dataset_config.input_keys:
                sanitized_item[key], replace_map = llm_rep_gen_with_entity_map(
                    args, sanitized_item[key], ner_result, tool_model, tool_tokenizer, repeat_model, repeat_tokenizer,
                    replace_map, "saniti", entity_dict, embedding_model)
                entity_map.update(replace_map)
        elif args.rep_gen_method == "ft_llm_generate":  # 方案四
            # 直接对待处理文本进行处理
            key = dataset_config.input_keys[0]
            sanitized_item[key], entity_map = ft_llm_rep_gen(args, sanitized_item[key], tool_model, tool_tokenizer,
                                                             ner_gen_model, ner_gen_tokenizer, ner_key)
            if len(dataset_config.input_keys) > 1 and entity_map:  # 针对squad、squad_v2等问答数据的"question"部分
                for key in dataset_config.input_keys[1:]:  # 直接用已经生成的字典对"question"进行字符串替换
                    sanitized_item[key] = str_replace_with_entity_map(entity_map, sanitized_item[key], "kv")

    all_sanitized_input = ""
    for key in dataset_config.input_keys:
        all_sanitized_input += sanitized_item[key] + " "

    # 计算原文本和净化文本之间的不相似性 -> 隐私保护效果
    san_effect = score_util.caculate_san_effect(all_input, all_sanitized_input, embedding_model)

    return sanitized_item, entity_map, san_effect


# ===== ↓ ===== recover 主函数 ===== ↓ ===== #

def recover_output(args, entity_map, sanitized_outputs, tool_model, tool_tokenizer, repeat_model, repeat_tokenizer):
    if args.recover_method == "str_replace":
        return str_replace_with_entity_map(entity_map, sanitized_outputs, "vk")
    elif args.recover_method == "llm_generate":
        recover_entity_map = {value: key for key, value in entity_map.items()}  # 把被替换的token作为key
        return llm_rep_gen_with_entity_map(
            args, sanitized_outputs, recover_entity_map, tool_model, tool_tokenizer,
            repeat_model, repeat_tokenizer, {}, "recover", None, None)[0]  # 反向执行生成式替换 -> recover
    elif args.recover_method == "llm_end2end_step3":
        recover_entity_map = {value: key for key, value in entity_map.items()}
        return sanitize_end2end_step3_replace(sanitized_outputs, recover_entity_map, tool_model, tool_tokenizer)
