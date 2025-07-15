# 初始化 generate_process_replace 过程的中间信息
from Config import prompt
from Utils import infer_util


# 判断生成过程是否已经终止
def generation_early_stop(next_token, current_generated_text):
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


# 根据输入的 candidate_entity_map, 获取 token <-> entity 映射关系
def init_token_entity_map(item_input_content, candidate_entity_map, rep_gen_tokenizer):
    # key 为 entity 对应的 token 子列表的元组， val 为 entity 本身
    ## 举例: {("Donald", "ĠTrump") : "Donald Trump"}
    token_entity_map = {}

    # 获取输入文本的 token 列表
    ## 去掉 token 的前导空格符 Ġ ; Hello world! -> ["Hello","Ġworld","!"] -> ["hello"," world","!"]
    item_input_content_tokens = [x.replace("Ġ", " ") for x in rep_gen_tokenizer.tokenize(item_input_content)]

    # item_input_content 中的所有待替换实体列表, 即 candidate_entity_map 的 key
    entity_list = candidate_entity_map.keys()

    # 遍历 token, 找到哪些 token 是需要被替换的
    ## O(m*n)时间复杂度;   m 为 token 数, n 为 entity 数
    for token_start_idx, token in enumerate(item_input_content_tokens):
        token_lower = token.strip().lower()
        for entity_idx, entity in enumerate(entity_list):  # 判断当前token是否需要被替换

            ## key 形式一 : 贪心匹配
            entity_lower = entity.lower()
            if entity_lower.startswith(token_lower):
                # 双指针,匹配同一个实体分布在多个token中的情况
                token_end_idx = token_start_idx + 1
                for token_end_idx in range(token_start_idx + 1, len(item_input_content_tokens)):
                    token_temp = "".join(item_input_content_tokens[token_start_idx:token_end_idx + 1]).lower().strip()
                    if not entity_lower.startswith(token_temp):
                        break
                matched_token_list = item_input_content_tokens[token_start_idx:token_end_idx]  # entity 对应的 token 子列表
                matched_token = "".join(matched_token_list).lower().strip()
                if matched_token == entity_lower and tuple(matched_token_list) not in token_entity_map:
                    token_entity_map.update({tuple(matched_token_list): entity})

    for entity_idx, entity in enumerate(entity_list):
        ## key 形式二 : 直接对 entity 进行 token 化
        matched_token_list = [x.replace("Ġ", " ") for x in rep_gen_tokenizer.tokenize(entity)]
        if tuple(matched_token_list) not in token_entity_map:
            token_entity_map.update({tuple(matched_token_list): entity})

        ## key 形式三 : 直接用空格分词
        matched_token_list = [x.strip() for x in entity.split()]
        if tuple(matched_token_list) not in token_entity_map:
            token_entity_map.update({tuple(matched_token_list): entity})

    return token_entity_map


# 判断当前生成的 next_token 是否在输入的替换字典中，如果在，则做贪心匹配返回匹配到的待替换实体
def find_need_replace_entity(next_token, token_entity_map, repeat_model, repeat_tokenizer, repeat_input_prompt):
    for idx, token_tuple_key in enumerate(token_entity_map.keys()):
        matched = True
        for one_token in token_tuple_key:  # 贪心匹配当前生成的 next_token 是否在已保存的需要替换的实体 token 列表中
            if one_token.strip().lower() == next_token.strip().lower():
                repeat_input_prompt += next_token  # 不替换的情况下，继续执行前向传播，获取待替换 entity 的全部 token
                next_token, next_token_logits = infer_util.next_token_inference(
                    repeat_input_prompt, repeat_model, repeat_tokenizer, )
            else:  # 当前 token_tuple_key 未匹配，继续遍历外层
                matched = False
                break
        if matched:
            return token_entity_map[token_tuple_key]
    return ""


# 生成式替换 主函数
def replace_generative(args, item_input_content, candidate_entity_map, repeat_model, repeat_model_tokenizer):
    current_generated_text = ""

    repeat_prompt = prompt.TASK["ft_repeat"].format(item_input_content)

    token_entity_map = init_token_entity_map(item_input_content, candidate_entity_map, repeat_model_tokenizer)

    # 前向传播 / 生成 next_token 的次数
    output_length = len(repeat_model_tokenizer.tokenize(repeat_prompt))
    generate_count = min(2048, output_length)

    # 记录一下上一个 token，避免重复输出
    pre_token = ""

    # 逐个生成 token，并在生成过程中做处理
    for idx in range(generate_count):

        # 获得 next_token（带格式的） 以及对应的 logits 分布
        next_token, next_token_logits = infer_util.next_token_inference(
            repeat_prompt, repeat_model, repeat_model_tokenizer)
        ori_next_token = next_token

        # 去重
        if pre_token == next_token and len(next_token) > 2:
            # 重试三次，找到二者不相等的下一个token
            count = 2
            while pre_token == next_token and count:
                next_token, next_token_logits = infer_util.next_token_inference(
                    repeat_prompt + next_token, repeat_model, repeat_model_tokenizer)
                count -= 1
        else:
            pre_token = next_token

        # 判断生成的 next_token 是否是待替换的，并返回替换后的实体作为 next_token
        need_replace_entity = find_need_replace_entity(
            next_token, token_entity_map, repeat_model, repeat_model_tokenizer, repeat_prompt)
        if need_replace_entity != "":  # 找到了待替换实体
            next_token = candidate_entity_map[need_replace_entity]
            if current_generated_text[-1:] != " " and not next_token.startswith(" "):
                next_token = " " + next_token

        # 判断是否满足提前结束条件
        if args.generate_early_stop:
            cur_rep_text_when_early_stop = generation_early_stop(next_token, current_generated_text)
            if cur_rep_text_when_early_stop != "":
                current_generated_text = cur_rep_text_when_early_stop
                break

        # 准备下一次的生成处理
        ## 1、拼接生成内容
        current_generated_text += next_token
        ## 2、如果当前token发生了替换，先不改变上文
        if need_replace_entity != "":
            repeat_prompt += ori_next_token
        ## 3、当前token未发生替换，统一替换上文
        else:
            repeat_prompt = prompt.TASK["ft_repeat"].format(item_input_content) + current_generated_text

    return current_generated_text
