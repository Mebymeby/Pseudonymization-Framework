import random

import torch
from sentence_transformers import util

from Config import prompt
from Utils import infer_util, diff_util


# 验证是否需要重新获取 candidate_entity
def overlap_verify(ori_entity, candidate_entity, ori_text, entity_map):
    verify_expr_list = [
        candidate_entity in ori_entity,  # candidate_entity 是 ori_entity 的一部分
        ori_entity in candidate_entity,  # ori_entity 是 candidate_entity 的一部分
        candidate_entity in ori_text,  # candidate_entity 是原文本的一部分
        any(candidate_entity in value for value in entity_map.values())  # candidate_entity 已经被使用过
    ]
    return any(verify_expr_list)  # 满足任一情况则需要重新获取 candidate_entity


# 通过数据集实体字典随机选择单个候选实体
def select_one_candidate_entity_from_dataset_entity_dict(
        ori_entity, ori_entity_type, dataset_entity_dict, item_input_content, candidate_entity_map, embedding_model):
    can_entity = random.choice(dataset_entity_dict[ori_entity_type])  # 此处选择是固定的
    can_entity_emb = embedding_model.encode(can_entity, convert_to_tensor=True)
    ori_entity_emb = embedding_model.encode(ori_entity, convert_to_tensor=True)
    ori_can_sim = util.cos_sim(ori_entity_emb, can_entity_emb).squeeze(0).item()

    best_can_entity = can_entity  # 记录最佳候选实体
    best_sim = ori_can_sim  # 记录最佳相似度
    re_generate_counter = 3  # 重新生成次数计数器，超出时返回当前结果
    while (re_generate_counter
           and (
                   overlap_verify(ori_entity, best_can_entity, item_input_content, candidate_entity_map)
                   or ori_can_sim < 0.3)
    ):
        random_state = random.getstate()  # 记录当前种子信息
        random.seed()  # 重置随机种子
        can_entity = random.choice(dataset_entity_dict[ori_entity_type])  # 此处选择是随机的
        random.setstate(random_state)  # 恢复随机种子信息
        can_entity_emb = embedding_model.encode(can_entity, convert_to_tensor=True)
        ori_can_sim = util.cos_sim(ori_entity_emb, can_entity_emb).squeeze(0).item()
        if ori_can_sim > best_sim:  # 更新最佳候选
            best_can_entity = can_entity
            best_sim = ori_can_sim
        re_generate_counter -= 1

    return best_can_entity


# 构造生成候选实体时使用的 llm_prompt
def format_llm_generate_entity_prompt(args, ori_entity, ori_entity_type):
    gen_entity_prompt = ""
    if args.llm_generate_entity_prompt_input == "entity_and_type":
        gen_entity_prompt = prompt.TASK["generate_with_entity_and_type"].format(ori_entity, ori_entity_type)
    elif args.llm_generate_entity_prompt_input == "entity":
        gen_entity_prompt = prompt.TASK["generate_with_entity"].format(ori_entity)
    elif args.llm_generate_entity_prompt_input == "type":
        gen_entity_prompt = prompt.TASK["generate_with_type"].format(ori_entity_type)

    return gen_entity_prompt


# 通过 llm_prompt 获取单个候选实体
def generate_one_candidate_entity_by_llm_prompt(args, ori_entity, ori_entity_type, item_input_content,
                                                candidate_entity_map, tool_model, tool_tokenizer):
    gen_entity_prompt = format_llm_generate_entity_prompt(args, ori_entity, ori_entity_type)  # 构造 prompt
    candidate_entity = infer_util.prompt_inference(tool_model, tool_tokenizer, gen_entity_prompt, 8)

    re_generate_counter = 3  # 重新生成次数计数器，超出时返回当前结果
    while (re_generate_counter
           and overlap_verify(ori_entity, candidate_entity, item_input_content, candidate_entity_map)):
        # 使用生成的 candidate_entity 替代 ori_entity 构造新 prompt
        gen_entity_prompt = format_llm_generate_entity_prompt(args, candidate_entity, ori_entity_type)
        candidate_entity = infer_util.prompt_inference(tool_model, tool_tokenizer, gen_entity_prompt, 8)
        re_generate_counter -= 1

    return candidate_entity


# 简单过滤
def is_valid(token_id, tool_model_tokenizer):
    token = tool_model_tokenizer.decode(token_id, skip_special_tokens=True).replace("Ġ", "").strip()
    return token.isalpha() and len(token) > 1


# top_k 算法入口
def select_one_candidate_entity_by_top_k(args, current_generated_text, ori_entity, next_target, tag_model,
                                         tag_model_tokenizer):
    inputs = tag_model_tokenizer(current_generated_text, return_tensors="pt").to(tag_model.device)

    outputs = tag_model.generate(
        **inputs,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tag_model_tokenizer.eos_token_id
    )

    target_position = len(inputs.input_ids[0])
    scores = outputs.scores[0]
    top_k = torch.topk(scores, k=args.top_k_range, dim=-1)

    ori_token_id = outputs.sequences[0, target_position]

    candidates_token_id = [token_id.item() for token_id in top_k.indices[0]
                           if token_id != ori_token_id and is_valid(token_id, tag_model_tokenizer)]

    # 打印候选token及其解码结果
    # print("valid topK tokens: ")
    # for token_id in candidates_token_id:
    #     token = tag_model_tokenizer.decode(token_id, skip_special_tokens=True)
    #     print(f"ID: {token_id} → Text: '{token}'")

    if not candidates_token_id:
        print(f"[ERROR]:top_k range {args.top_k_range} found no candidate entity!")
        return ori_entity

    # 验证候选 token 合理性
    best_candidate = None
    best_score = -float('inf')

    for candidate in candidates_token_id:
        # 拼接新输入
        new_text = current_generated_text + tag_model_tokenizer.decode(candidate)
        new_inputs = tag_model_tokenizer(new_text, return_tensors="pt").to(tag_model.device)

        # 计算后续目标 token 的概率
        outputs = tag_model(**new_inputs)
        next_token_logits = outputs.logits[0, -1, :]
        key_token_id = tag_model_tokenizer.encode(next_target, add_special_tokens=False)[0]
        key_token_prob = torch.softmax(next_token_logits, dim=-1)[key_token_id].item()

        if key_token_prob > best_score:
            best_score = key_token_prob
            best_candidate = candidate

    return tag_model_tokenizer.decode(best_candidate, skip_special_tokens=True)


# 参数路由
def generate_one_candidate_entity(args, ori_entity, ori_entity_type, dataset_entity_dict, item_input_content,
                                  tag_model_repeat_prompt, candidate_entity_map, tool_model, tool_model_tokenizer,
                                  tag_model, tag_model_tokenizer, embedding_model):
    candidate_entity = ori_entity

    if args.generate_method == "rand":
        candidate_entity = select_one_candidate_entity_from_dataset_entity_dict(
            ori_entity, ori_entity_type, dataset_entity_dict, item_input_content, candidate_entity_map, embedding_model)
    elif args.generate_method == "prompt":
        candidate_entity = generate_one_candidate_entity_by_llm_prompt(
            args, ori_entity, ori_entity_type, item_input_content, candidate_entity_map, tool_model,
            tool_model_tokenizer)
    elif args.generate_method == "top_k":
        if args.tag_model_type == "tag_mask":
            prefix, suffix = reformat_prompt_get_prefix_suffix(args, tag_model_repeat_prompt, "<ENTITY>")
        elif args.tag_model_type == "tag_mark":
            prefix, suffix = reformat_prompt_get_prefix_suffix(args, tag_model_repeat_prompt, "<Entity>")
        else:
            print("[ERROR]:tag_model_type must be 'tag_mask' or 'tag_mark'")
            exit(1)

        candidate_entity = select_one_candidate_entity_by_top_k(
            args, tag_model_repeat_prompt, ori_entity, suffix, tag_model, tag_model_tokenizer)

    return candidate_entity


# [Generate] candidate entities 主函数
def generate_candidate_entity_map(args, entity_list_map, dataset_entity_dict, item_input_content,
                                  tool_model, tool_model_tokenizer, tag_model, tag_model_tokenizer, embedding_model):
    candidate_entity_map = {}

    for ori_entity, ori_entity_type in entity_list_map.items():
        candidate_entity = generate_one_candidate_entity(
            args, ori_entity, ori_entity_type, dataset_entity_dict, item_input_content, "", candidate_entity_map,
            tool_model, tool_model_tokenizer, tag_model, tag_model_tokenizer, embedding_model)

        candidate_entity_map.update({ori_entity: candidate_entity})

    return candidate_entity_map


# 生成翻译数据目标语言的实体映射
def gen_entity_map_trans(dataset_config, ori_sentence, pseu_sentence, tool_model, tool_tokenizer):
    ori_trans_prompt = prompt.DATASET["wmt14-de-en"].format(ori_sentence)
    san_trans_prompt = prompt.DATASET["wmt14-de-en"].format(pseu_sentence)

    ori_trans_res = infer_util.prompt_inference(tool_model, tool_tokenizer, ori_trans_prompt,
                                                dataset_config.infer_max_new_tokens)
    san_trans_res = infer_util.prompt_inference(tool_model, tool_tokenizer, san_trans_prompt,
                                                dataset_config.infer_max_new_tokens)

    ori_trans_token = tool_tokenizer.tokenize(ori_trans_res)
    san_trans_token = tool_tokenizer.tokenize(san_trans_res)

    entity_map_trans = diff_util.process_token_diff_map(ori_trans_token, san_trans_token)

    return entity_map_trans


# 找到最后一个匹配项并替换
def reformat_prompt_get_prefix_suffix(args, repeat_prompt, replace_part):
    # 查找最后一个替换标记的下标
    last_pos = repeat_prompt.rfind(replace_part)
    if last_pos == -1:
        return repeat_prompt  # 未找到替换标记，返回原prompt

    # 获得 prompt 前缀
    prefix = repeat_prompt[:last_pos]

    suffix = ""
    if args.tag_model_type == "tag_mask":
        suffix = repeat_prompt[last_pos:].replace(replace_part, "")
    elif args.tag_model_type == "tag_mark":
        suffix_with_tag = repeat_prompt[last_pos:]
        end_tag_idx = suffix_with_tag.rfind("<\Entity>")
        suffix = suffix_with_tag[end_tag_idx:].replace("<\Entity>", "")

    return prefix, suffix
