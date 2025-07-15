import re

from Config import prompt
from Utils import infer_util, generate_process, diff_util


# 判断生成过程是否已经终止
def generation_early_stop_sig(next_token, current_generated_text):
    for early_stop_token in ["<|im_end|>", "<|endoftext|>", "Output", "\n\n"]:

        # next_token 本身是终止词时，输出当前生成的文本作为最终文本
        if next_token.strip() == early_stop_token:
            return current_generated_text

        # 如果当前输出加上 next_token 之后包含了终止词，则去除终止词
        output_text_temp = current_generated_text + next_token
        if early_stop_token in output_text_temp:
            current_generated_text = output_text_temp.replace(early_stop_token, "")
            return current_generated_text

    return ""


# 获取 tag_mask 模型被替换的词汇
def get_tag_mask_entity(args, current_generated_text, item_input_text, replace_part, tag_model_tokenizer):
    masked_token = ""

    ori_text_tokens = [x for x in tag_model_tokenizer.tokenize(item_input_text.replace("\n", " "))]
    tag_mask_tokens = [x for x in tag_model_tokenizer.tokenize(current_generated_text.replace("\n", " "))]

    # 由于此时 repeat 过程很可能只进行了一部分，tag_mask_tokens 较少，因此截取 ori_text_tokens 的子集做计算，减少错误且提升速度
    ori_text_tokens = ori_text_tokens[:int(len(tag_mask_tokens) * 1.2)]

    # 获取 token 级别的映射，其中包含 tag_mask -> ori_token 的映射
    token_diff_map = diff_util.process_token_diff_map(tag_mask_tokens, ori_text_tokens)

    # 找到 key 为 replace_part 的部分
    token_matching_pair = [{key: token_diff_map[key]} for key in token_diff_map if replace_part in key]
    if token_matching_pair:  # 找到占位符
        if len(token_matching_pair) > 1:
            print(f"[ERROR]: unexpected redundant replace_part in tag_mask token_diff_map: {token_matching_pair}")

        token_diff_key, token_diff_value = next(
            iter(token_matching_pair[0].items()))  # {'<ENTITY>,': 'University of Southern California,'}

        if replace_part == token_diff_key:  # 占位符为 key 本身时，则 masked_token 为 token_diff_value 本身
            masked_token = token_diff_value
        else:

            # token_diff_key 和 token_diff_value 长度相差不大的情况
            if len(token_diff_value) - len(token_diff_key) <= args.search_diff_length:
                tag_left_pos = token_diff_key.find("<")  # 找到 tag 的起始位置
                val_left_pos = tag_left_pos  # val 的起始位置和 tag 相同
                if token_diff_key.endswith(">"):  # 如果 token_diff_key 以 tag 结尾
                    masked_token = token_diff_value[val_left_pos:]  # 直接获取 masked_token
                else:
                    tag_right_pos = token_diff_key.find(">")  # 找到 tag 的结束位置
                    tag_right_part = token_diff_key[tag_right_pos + 1:]  # 获取 token_diff_key 中，tag 后面的部分
                    val_right_pos = token_diff_value.find(tag_right_part)  # 获取 token_diff_value 中 tag_right_part 的起始位置
                    masked_token = token_diff_value[val_left_pos:val_right_pos]  # 则中间部分为 masked_token

            # token_diff_key 和 token_diff_value 长度相差较大的情况
            else:
                # 获取 char 级别的映射
                token_diff_value = token_diff_value[:(len(token_diff_key) + args.search_diff_length)]
                char_diff_map = diff_util.process_token_diff_map(token_diff_key, token_diff_value)

                start_str, end_str = "", ""
                for char_diff_key, char_diff_val in char_diff_map.items():
                    if replace_part == char_diff_key:
                        masked_token = char_diff_val
                        break
                    elif char_diff_key.startswith("<"):
                        start_str = char_diff_val.replace("+", " ")
                    elif char_diff_key.endswith(">"):
                        end_str = char_diff_val.replace("+", " ")
                    if start_str != "" and end_str != "":
                        start_pos = token_diff_value.find(start_str)
                        end_pos = token_diff_value.find(end_str)
                        masked_token = token_diff_value[start_pos:end_pos] + end_str
                        break

                masked_token = masked_token.replace("+", " ")

        return masked_token  # 返回找到的 masked_token

    else:  # 未找到占位符
        print("[WARNING]: tag_mask not found in token_diff_map, return empty string instead")
        return ""


# 从 tag_model 的当前输出寻找需要替换的部分
def find_need_replace_part(args, current_generated_text, item_input_text, tag_model_tokenizer, final=False):
    ori_entity = ""
    tag_entity_part = ""

    # 获取占位符搜索字符串 TODO 需要研究下这块的作用,末端少几个字符的影响？
    if final or args.search_suffix_threshold <= 0:  # repeat 完毕 或 没有设置前向占位符搜索字符数，则搜索范围为当前生成的全文本
        output_text_temp = current_generated_text
    else:  # repeat 过程中 且 设置了前向占位符搜索字符数，则搜索范围为当前生成文本的[:-前向占位符搜索字符数]
        output_text_temp = current_generated_text[:-args.search_suffix_threshold]

    no_need_cond_expr_list = [
        not "<" in output_text_temp or not ">" in output_text_temp,  # 搜索内容中不包含完整占位符
        output_text_temp.endswith(">"),  # 搜索内容以占位符结束符结尾时，不进行替换（使repeat过程稳定）
        # tag_mark 模型需要两对完整占位符
        args.tag_model_type == "tag_mark" and (output_text_temp.count("<") < 2 or output_text_temp.count(">") < 2)
    ]

    # 满足以上任一条件，则说明当前无需替换内容
    if any(no_need_cond_expr_list):
        # return current_generated_text, ori_entity, tag_entity_part
        return ori_entity, tag_entity_part

    # 正式进行处理
    if args.tag_model_type == "tag_mark":
        matches = re.finditer(r"<(.*?)>(.*?)</\1>", current_generated_text)  # 用正则表达式获取 tag_mark 整块内容, 含实体本身
        for match in matches:  # 正确情况应该只有一个匹配项
            tag = match.group(1)
            ori_entity = match.group(2)
            tag_entity_part = "<{0}>{1}</{0}>".format(tag, ori_entity)
    elif args.tag_model_type == "tag_mask":
        matches = re.finditer(r"<(.*?)>", current_generated_text)  # 用正则表达式获取 tag_mask 标记, 不含实体本身
        for match in matches:
            tag = match.group(1)
            tag_entity_part = "<{0}>".format(tag)
        # 获取被 tag_mask 掉的实体本身
        ori_entity = get_tag_mask_entity(args, current_generated_text, item_input_text, tag_entity_part,
                                         tag_model_tokenizer)
    else:
        print("[ERROR]: param undefined: args.tag_model_type")
        exit(1)

    return ori_entity.strip(), tag_entity_part


def no_need_generate_candidate_entity_verify(ori_entity, entity_map, current_generated_text):
    # 情形一: 当前实体已被 entity_map 记录，直接返回记录值
    if ori_entity in entity_map and entity_map[ori_entity] != "":
        return entity_map[ori_entity]

    # 情形二: 当前实体未被 entity_map 记录，但是在前文中出现过(即:未知原因未被替换)，则不进行替换，防止语义不一致
    elif ori_entity in current_generated_text[:current_generated_text.rfind(ori_entity)]:
        return ori_entity

    # 情形三: 需要生成新实体
    else:
        return ""


# 根据输入文本，通过 tag_model 执行 detect、replace 过程
def detect_replace_generative(args, item_input_content, dataset_entity_dict, entity_list_map, candidate_entity_map,
                              tool_model, tool_model_tokenizer, repeat_model, repeat_model_tokenizer,
                              tag_model, tag_model_tokenizer, ner_model, embedding_model):
    current_generated_text = ""  # 当前生成的替换文本, 逐渐拼接为最终输出文本

    tag_model_repeat_prompt = ""
    if args.tag_model_type == "tag_mask":
        tag_model_repeat_prompt = prompt.TASK["tag_mask_repeat"].format(item_input_content)
    elif args.tag_model_type == "tag_mark":
        tag_model_repeat_prompt = prompt.TASK["tag_mark_repeat"].format(item_input_content)
    else:
        print("[ERROR]: param undefined: args.tag_model_type")
        exit(1)

    # 前向传播 / 生成 next_token 的次数
    output_length = len(tag_model_tokenizer.tokenize(tag_model_repeat_prompt))
    generate_count = min(2048, output_length + 1)

    # 逐个生成 token，并在生成过程中做处理
    for idx in range(generate_count):

        if idx != generate_count - 1:  # token 生成未完成时，需要获取 next_token, 判断结束条件, 记录结果等

            # 获取 next_token
            next_token, next_token_logits = infer_util.next_token_inference(
                tag_model_repeat_prompt, tag_model, tag_model_tokenizer)

            # 判断是否满足提前结束条件
            if args.generate_early_stop:
                cur_rep_text_when_early_stop = generation_early_stop_sig(next_token, current_generated_text)
                if cur_rep_text_when_early_stop != "":
                    current_generated_text = cur_rep_text_when_early_stop
                    break

            tag_model_repeat_prompt += next_token  # 准备生成下一个 token 的 prompt
            current_generated_text += next_token  # 记录当前结果

        # 框架循环

        ## ========== [Detect] entities ========== ##
        ori_entity, tag_entity_part = find_need_replace_part(
            args, current_generated_text, item_input_content, tag_model_tokenizer)
        if ori_entity != "":  # 找到了需要替换的实体
            candidate_entity = no_need_generate_candidate_entity_verify(ori_entity, candidate_entity_map,
                                                                        current_generated_text)
            ori_entity_type = "entity"
            if not args.fast_run_step[0]:
                ori_entity_type = infer_util.determine_entity_type(ori_entity, ner_model)
                entity_list_map.update({ori_entity: ori_entity_type})

            ## ========== [Generate] candidate entities ========== ##
            if args.run_step[1]:
                if candidate_entity == "":  # 需要生成新的候选实体
                    # 生成新的候选实体
                    candidate_entity = generate_process.generate_one_candidate_entity(
                        args, ori_entity, ori_entity_type, dataset_entity_dict, item_input_content,
                        tag_model_repeat_prompt, candidate_entity_map, tool_model, tool_model_tokenizer, tag_model,
                        tag_model_tokenizer, embedding_model)
                    # 记录新的映射关系
                    candidate_entity_map.update({ori_entity: candidate_entity})
            else:
                candidate_entity = ori_entity

            ## ========== [Replace] privacy entity in text ========== ##
            current_generated_text = current_generated_text.replace(tag_entity_part, candidate_entity)

            # 更新提示文本
            if args.tag_model_type == "tag_mask":
                tag_model_repeat_prompt = prompt.TASK["tag_mask_repeat"].format(
                    item_input_content) + current_generated_text
            if args.tag_model_type == "tag_mark":
                tag_model_repeat_prompt = prompt.TASK["tag_mark_repeat"].format(
                    item_input_content) + current_generated_text

    return current_generated_text, candidate_entity_map
