import argparse
import difflib
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# 参数解析
def parse_args(args):
    parser = argparse.ArgumentParser(description='run LLM with prompt')

    # IO基础参数
    parser.add_argument('--ner_gen_model_path', type=str,
                        default="/data5/longzi/Workspace/Pytorch/Qwen/Qwen2.5-1.5B-Instruct",
                        required=False)
    parser.add_argument('--ent_gen_model_path', type=str,
                        default="/data5/longzi/Workspace/Pytorch/Qwen/Qwen2.5-1.5B-Instruct",
                        required=False)
    parser.add_argument('--input_text_path', type=str, default="input/output_for_raw/xsum_train_document.raw_data"
                        , required=False)
    parser.add_argument('--output_file_prefix', default="output/test.output", required=False)
    parser.add_argument('--ner_gen_prompt_path', type=str, default="input/prompt/ner_gen.20250125.prompt",
                        required=False)
    parser.add_argument('--ner_gen_type', type=str, required=False, default="only_rep")
    parser.add_argument('--ent_gen_prompt_path', type=str, default="input/prompt/generate.20250120.prompt",
                        required=False)
    parser.add_argument("--max_line", type=int, default=0, required=False)
    parser.add_argument("--max_length", type=int, default=4096, required=False)

    # 程序逻辑处理参数
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--early_stop_flags", type=str, required=False, default="<|im_end|>,<|endoftext|>")
    parser.add_argument("--replace_topk", type=int, default=20, required=False)
    parser.add_argument("--generate_type", type=str, default="prompt", required=False)
    # prompt：通过提示文本生成，或是topk：通过LM的topk逻辑生成
    parser.add_argument("--generate_context", type=str, default="seperate", required=False)
    # 选择topk时只能默认使用prefix内容
    # 当选择prompt进行替换式，使用前文（prefix）还是使用全局（global)信息，还是不适用语境（seperate, seperate_with_type）进行替换
    parser.add_argument("--generate_error_thresold", type=float, default=0.1, required=False)
    parser.add_argument("--output_text_head", type=str, required=False, default="")
    parser.add_argument("--search_mark_timeout", type=int, required=False, default=100)
    # search相关参数
    parser.add_argument("--search_range", type=float, required=False, default=0.5)
    parser.add_argument("--search_suffix_thresold", type=int, required=False, default=1)
    parser.add_argument("--search_diff_length", type=int, required=False, default=20)
    parser.add_argument("--search_replace_constraint", type=str, required=False, default="dict")

    # debug相关参数
    parser.add_argument('--is_debug', action="store_true")
    parser.add_argument('--debug_line', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42, required=False)
    flags = parser.parse_args()

    # 部分参数调整
    if flags.max_line > 0:
        flags.debug_line = min(flags.debug_line, flags.max_line)
    if flags.early_stop:
        flags.early_stop_flag_list = flags.early_stop_flags.split(",")

    # 输出参数信息
    print("[INFO] flags: ", flags, file=sys.stderr, flush=True)

    # 部分参数合法性校验
    assert os.path.exists(flags.input_text_path), "input_text_path {0} not exist".format(flags.input_text_path)
    assert flags.ner_gen_type in ["only_rep", "tag_wrap"], "ner_gen_type must be a value in only_rep, tag_warp"
    assert flags.generate_type in ["topk", "prompt"], "generate_type must be a value in topk, or prompt"
    assert flags.generate_context in ["prefix", "global", "seperate", "seperate_with_type", "only_type"], \
        "generate_context must be a value in prefix, global, or seperate, seperate_with_type, only_type."

    # 读取repeat_prompt，加入到flags的空间中
    flags.ner_gen_prompt = read_prompt(flags.ner_gen_prompt_path)

    # 如果generate_type是prompt，则额外读取一个prompt
    if flags.generate_type == "prompt":
        flags.ent_gen_prompt = read_prompt(flags.ent_gen_prompt_path)

    return flags


# 读取预训练好的模型
def from_pretrained(model_path):
    print("[INFO] loading pretrained model {0}".format(model_path), file=sys.stderr, flush=True)
    # 载入模型部分
    start_time = time.time()
    # 载入模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="sequential"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[INFO] cost {0} seconds for init".format(time.time() - start_time), file=sys.stderr, flush=True)

    return model, tokenizer


# 设定种子值
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(seed)


# 读取提示文本文件
def read_prompt(prompt_path):
    with open(prompt_path, "r") as input:
        content = "".join(input.readlines())
    prompt = content
    print("[INFO] prompt: \n{0}".format(prompt), file=sys.stderr, flush=True)

    return prompt


# 读取数据
def read_data(flags):
    input_text = list()
    # 只需要读取文本数据
    with open(flags.input_text_path, "r", encoding="utf8") as input:
        print("[INFO] reading input text file: {0}".format(flags.input_text_path), file=sys.stderr, flush=True)
        count = 0
        for line in input:
            line = line.strip()
            input_text.append(json.loads(line))
            count += 1
            if flags.max_line > 0 and count >= flags.max_line:
                break

    if flags.is_debug:
        for idx, input_text_line in enumerate(input_text[:flags.debug_line]):
            print("[INFO] read_data debug. ")
            print(input_text_line)
            print()

    return input_text


# 执行生成式替换
def replace_generate(input_data, ner_gen_model, ner_gen_tokenizer, ent_gen_model, ent_gen_tokenizer, flags=None):
    if flags.generate_type == "prompt":
        assert ent_gen_model is not None and ent_gen_tokenizer is not None

    print("[INFO] processing {0} lines of input data".format(len(input_data)), file=sys.stderr, flush=True)

    # 将整体结果保存出来
    result_list = list()
    entity_dict_list = list()

    # 对数据进行处理
    for input_text in tqdm(input_data):
        # 初始化部分值
        prompt_text = flags.ner_gen_prompt.format(input_text)
        output_text = ""
        # 对replace_dict进行处理
        replace_dict = preprocess_replace_dict(input_text, ner_gen_model, ner_gen_tokenizer, flags)
        # replace_dict = dict()

        # 根据输入长度判断循环次数
        output_length = min(flags.max_length, len(ner_gen_tokenizer.tokenize(prompt_text)))

        # 开始处理
        for _ in range(output_length):
            # 获得下一次token，以及对应的logits分布
            next_token, next_token_logits = generate_next_token(prompt_text, ner_gen_model, ner_gen_tokenizer, flags)
            # 判断是否提前结束
            if flags.early_stop:
                stop_flag = need_early_stop(next_token, output_text, flags)
                if stop_flag != "":
                    output_text = stop_flag
                    if flags.is_debug:
                        print("[INFO] early stop at {0}".format(next_token))
                    break
            # 准备下一次的生成处理
            prompt_text += next_token
            output_text += next_token
            # 判断是否需要进行替换
            output_text, replace_word, replace_part = need_replace(output_text, input_text, ner_gen_tokenizer, flags)
            if replace_word != "":
                # 需要替换
                # 生成替换值
                generate_word, replace_dict = generate_replace_word(output_text, input_text, replace_word, replace_dict,
                                                                    ent_gen_model, ent_gen_tokenizer, flags)
                # 执行替换
                output_text = output_text.replace(replace_part, generate_word)
                # 更新一下提示问嗯
                prompt_text = flags.ner_gen_prompt.format(input_text) + output_text

        # 结束之前还需要进行一次替换判断
        output_text, replace_word, replace_part = need_replace(output_text, input_text, ner_gen_tokenizer, flags,
                                                               final=True)
        if replace_word != "":
            # 需要替换
            # 生成替换值
            generate_word, replace_dict = generate_replace_word(output_text, input_text, replace_word, replace_dict,
                                                                ent_gen_model, ent_gen_tokenizer, flags)
            # 执行替换
            output_text = output_text.replace(replace_part, generate_word)

        if flags.is_debug:
            print("[INFO] replace_generate debug. ")
            print("###input:\n", input_text)
            print("###output:\n", output_text)
            print("###replace_dict:\n", replace_dict)
            print()

        # 整理输出结果
        if flags.output_text_head != "":
            output_text = output_text.replace(flags.output_text_head, "").strip()
        result_list.append(output_text)
        entity_dict_list.append(replace_dict)

    return result_list, entity_dict_list


# 预先获取需要替换的词汇
def preprocess_replace_dict(input_text, ner_gen_model, ner_gen_tokenizer, flags):
    replace_dict = dict()
    if flags.search_replace_constraint == "dict" and flags.ner_gen_type == "tag_wrap":
        # tag_wrap进行额外处理，预先获取那些词汇需要处理
        # 即只需要重新执行一下，获取标注的词就好
        prompt_text = flags.ner_gen_prompt.format(input_text)
        generate_text = generate_response(prompt_text, ner_gen_model, ner_gen_tokenizer, flags)
        matches = re.finditer(r"<(.*?)>(.*?)</\1>", generate_text)
        for match in matches:
            replace_word = match.group(2)
            replace_dict.update({replace_word: ""})
    if flags.is_debug:
        print("replace dict: ", replace_dict)

    return replace_dict


# 判断是否进行提前退出
def need_early_stop(next_token, output_text, flags):
    # print("next_token:", json.dumps(next_token))
    for early_stop_flag in flags.early_stop_flag_list:
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
def find_prefix_output(replace_word, output_text):
    last_index = output_text.rfind(replace_word)
    return output_text[:last_index]


# 生成式替换的每一步
def generate_replace_word(output_text, input_text, replace_word, replace_dict, ent_gen_model, ent_gen_tokenizer, flags):
    # 需要被替换，则进行替换生成
    if flags.generate_type == "topk":
        # 选择使用下k个单词
        # next_token = choose_from_topk(next_token, tokenizer, next_token_logits, flags)
        # 暂时不使用
        next_token = replace_word
        pass
    elif flags.generate_type == "prompt":
        # 选择通过一次LLM提示工程的生成来执行替换
        if replace_word in replace_dict and replace_dict[replace_word] != "":
            # 防止重复计算以及随机性带来的替换不一致，replace_dict已经有值时则直接进行替换
            next_token = replace_dict[replace_word]
        elif flags.ner_gen_type == "tag_wrap" and flags.search_replace_constraint == "dict" and replace_word not in replace_dict:
            # tag_wrap且dict限制条件时，不存在于replace_dict中的词不做生成，直接返回replace_wrod值就可以
            next_token = replace_word
        elif flags.ner_gen_type == "tag_wrap" and flags.search_replace_constraint == "context" and replace_word in find_prefix_output(
                replace_word, output_text):
            # tag_wrap且context限制条件时，存在于output_text前文中的replace_dict不做生成，直接返回replace_wrod值就可以
            # 后文中保证了，当replace_word存在于生成的output_text前文中时，replace_word必然是不同于原文的新内容
            # 这种方法的问题在于，replace不存在于output_text的前文中时也有可能新内容，这种时候只能期待前一种方法
            next_token = replace_word
        else:
            # 根据情况拼接提示文本
            if flags.generate_context == "prefix":
                # 使用前文进行生成
                exit(1)  # 效果不佳，暂时舍弃这种方案
                generate_prompt_text = flags.ent_gen_prompt.format(prefix_input_text.strip() + " " + replace_word,
                                                                   replace_word)
            elif flags.generate_context == "global":
                # 使用全文进行替换生成
                exit(1)  # 效果不佳，暂时舍弃这种方案
                generate_prompt_text = flags.ent_gen_prompt.format(
                    raw_input_text.replace(replace_word, flags.replace_place_holder))
            elif flags.generate_context == "seperate":
                generate_prompt_text = flags.ent_gen_prompt.format(replace_word)
            else:
                exit(1)
            # 执行生成
            next_token = generate_response(generate_prompt_text, ent_gen_model, ent_gen_tokenizer, flags)
            next_token = next_token.strip()
            # 判断一下生成单词是否已经出现在替换的列表中
            replace_word_temp = replace_word
            while next_token in input_text or next_token in output_text:
                # 防止不同的词被替换为同一个词，重新生成一下
                # 直接拿生成词再去生成一下
                generate_prompt_text = generate_prompt_text.replace(replace_word_temp, next_token)
                replace_word_temp = next_token
                next_token = generate_response(generate_prompt_text, ent_gen_model, ent_gen_tokenizer, flags)

            replace_dict.update({replace_word: next_token})

    if (flags.is_debug): print("generate next token: ***", next_token, "***\n")

    return next_token, replace_dict


# 生成next token
def generate_next_token(input_text, model, tokenizer, flags):
    # 执行输入文本的编码
    model_inputs = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    # 执行forward处理
    outputs = model(model_inputs)
    logits = outputs.logits
    # 得到当前情况下下一个单词的概率分布，并获取最大值作为next token
    next_token_logits = logits[:, -1, :].squeeze()  # size: [vocab]
    # 通过argmax取得logits最大值为next_token
    next_token_id = torch.argmax(next_token_logits)
    next_token = tokenizer.decode(next_token_id, skip_special_token=True)

    return next_token, next_token_logits


# 返回整个生成结果
def generate_response(input_text, model, tokenizer, flags):
    # 执行输入文本的编码
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    # 执行生成处理
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=flags.max_length,
    )
    # 处理一下生成结果并返回
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 判断是否需要替换
# 使用和另一边代码不同的逻辑
def need_replace(output_text, input_text, tokenizer, flags, final=False):
    replace_word = ""
    replace_part = ""

    # 为了加快判断速度，可能可以先加一个简短判断
    # 根据是否是最后一次进行略微不同的判断
    if final or flags.search_suffix_thresold <= 0:
        output_text_temp = output_text
    else:
        output_text_temp = output_text[:-flags.search_suffix_thresold]
    # 文本需要中包含足够占位符
    if flags.ner_gen_type == "tag_wrap" and (output_text_temp.count("<") < 2 or output_text_temp.count(">") < 2):
        return output_text, replace_word, replace_part
    elif flags.ner_gen_type == "only_rep" and (not "<" in output_text_temp or not ">" in output_text_temp):
        return output_text, replace_word, replace_part
    # 文本中必须要求标记占位符不能是结尾
    if output_text_temp.endswith(">"):
        return output_text, replace_word, replace_part

    if flags.is_debug:
        print("output_text: {0}".format(output_text))

    # 正式进行处理
    if flags.ner_gen_type == "tag_wrap":
        # 用正则表达式来处理
        matches = re.finditer(r"<(.*?)>(.*?)</\1>", output_text)
        for match in matches:
            # 如果正确的话，应该只有一个匹配上
            replace_tag = match.group(1)
            replace_word = match.group(2)
            replace_part = "<{0}>{1}</{0}>".format(replace_tag, replace_word)
            start_pos = match.start()
    elif flags.ner_gen_type == "only_rep":
        # 为了保证后续能够获得被计算的实体词汇
        # 用正则表达式来处理
        matches = re.finditer(r"<(.*?)>", output_text)
        for match in matches:
            replace_tag = match.group(1)
            replace_part = "<{0}>".format(replace_tag)
            start_pos = match.start()
        # 通过对比输入和输出的文本，计算此处被替换的实体词汇
        replace_word = search_replace_word(output_text, input_text, replace_part, tokenizer, flags)
    else:
        exit(1)

    if flags.is_debug:
        print("replace_part: {0}, replace_word: {1}".format(replace_part, replace_word))

    return output_text, replace_word, replace_part


# 对比输入和输出，获取被替换的词汇
def search_replace_word(output_text, input_text, replace_holder, tokenizer, flags):
    if flags.output_text_head != "":
        output_text = output_text.replace(flags.output_text_head, "").strip()

    # if flags.is_debug:
    #     print("search_replace_word fucntion")
    #     print("output_text: ", output_text)
    # print("input_text: ", input_text)

    replace_word = ""

    # 先对比输入和输出，获取其中不相同的部分
    # 分词处理
    tokenized_input_text = [x for x in tokenizer.tokenize(input_text.replace("\n", " "))]
    tokenized_output_text = [x for x in tokenizer.tokenize(output_text.replace("\n", " "))]
    # 防止干扰，先调整一下input_text
    tokenized_input_text = tokenized_input_text[:int(len(tokenized_output_text) * (1 + flags.search_range))]
    if flags.is_debug:
        print("input_text: ", "".join(tokenized_input_text).replace("Ġ", " "))
    diff_dict = process_diff_dict(tokenized_input_text, tokenized_output_text, flags)
    if flags.is_debug: print("diff_dict: ", diff_dict)

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
                if len(diff_value) - len(diff_key) <= flags.search_diff_length:
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

                        if flags.is_debug: print(
                            "left_pos, diff_key_right_pos, diff_key_right, diff_value, diff_value_right_pos",
                            left_pos, diff_key_right_pos, diff_key_right, diff_value, diff_value_right_pos)
                else:
                    # 如果相差太大
                    diff_value = diff_value[:(len(diff_key) + flags.search_diff_length)]
                    char_diff_dict = process_diff_dict(diff_value, diff_key, flags)
                    if flags.is_debug: print("char_diff_dict: ", char_diff_dict)

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


# 输出处理
def output_results(result_list, entity_dict_list, flags):
    # 输出替换结果
    output_result_path = flags.output_file_prefix + ".result"
    print("[INFO] writing results in {0}".format(output_result_path), file=sys.stderr, flush=True)
    with open(output_result_path, "w", encoding="utf8") as output:
        for result in result_list:
            print(json.dumps(result), file=output, flush=True)
    # 输出映射字典吧
    if flags.generate_type != "dict":
        output_dict_path = flags.output_file_prefix + ".entity_dict"
        print("[INFO] writing entity replace dict in {0}".format(output_dict_path), file=sys.stderr, flush=True)
        with open(output_dict_path, "w", encoding="utf8") as output:
            for entity_dict in entity_dict_list:
                print(json.dumps(entity_dict), file=output, flush=True)


# 处理替换映射表
def process_diff_dict(tokenized_result_text, tokenized_input_text, flags):
    # if flags.is_debug:
    #     print("input_text: \n", tokenized_input_text)
    #     print("output_text: \n", tokenized_result_text)

    entity_dict = dict()

    # 比对
    d = difflib.Differ()
    diff = d.compare(tokenized_input_text, tokenized_result_text)
    diff_list = [x for x in diff]

    if flags.is_debug: print(diff_list)

    # 输出
    replace_tok = ""
    replaced_tok = ""
    for tok in diff_list:
        if tok.startswith("-"):
            replaced_tok += "" + tok.split()[-1]
        elif tok.startswith("+"):
            replace_tok += "" + tok.split()[-1]
        elif tok.startswith("?"):
            pass
        elif replace_tok != "" and replaced_tok != "":
            entity_dict[replaced_tok.replace("Ġ", " ").strip()] = replace_tok.replace("Ġ", " ").strip()
            replace_tok = ""
            replaced_tok = ""
        else:
            replace_tok = ""
            replaced_tok = ""
    if replace_tok != "" and replaced_tok != "":
        entity_dict[replaced_tok.replace("Ġ", " ").strip()] = replace_tok.replace("Ġ", " ").strip()

    return entity_dict


def main():
    # 解析参数
    flags = parse_args(sys.argv)
    # 固定种子值，保证实验结果的稳定
    set_random_seed(flags.seed)
    # 读取数据
    input_data = read_data(flags)
    # 读取模型和分词器
    ner_gen_model, ner_gen_tokenizer = from_pretrained(flags.ner_gen_model_path)
    if flags.generate_type == "prompt":
        ent_gen_model, ent_gen_tokenizer = from_pretrained(flags.ent_gen_model_path)
    else:
        ent_gen_model, ent_gen_tokenizer = None, None
    # 执行生成式替换
    result_list, entity_dict_list = replace_generate(input_data, ner_gen_model, ner_gen_tokenizer, ent_gen_model,
                                                     ent_gen_tokenizer, flags)
    # 执行输出处理
    output_results(result_list, entity_dict_list, flags)


if __name__ == '__main__':
    main()
