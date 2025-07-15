import argparse
import difflib
import json
import os
import random
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
    parser.add_argument('--model_path', type=str,
                        default="/data5/longzi/Workspace/Pytorch/Qwen/Qwen2.5-1.5B-Instruct",
                        required=False)
    parser.add_argument('--input_text_path', type=str, default="input/output_for_raw/xsum_train_document.raw_data"
                        , required=False)
    parser.add_argument('--input_entity_path', type=str,
                        default="input/data/modified_dataset/xsum10000/item_entity_dict.txt",
                        required=False)
    parser.add_argument('--input_dict_path', type=str, default="output/test.output.entity_dict", required=False)
    parser.add_argument('--output_file_prefix', default="output/test.output", required=False)
    parser.add_argument('--repeat_prompt_path', type=str, default="input/prompt/repeat.20250120.prompt", required=False)
    parser.add_argument('--generate_prompt_path', type=str, default="input/prompt/generate.20250120.prompt",
                        required=False)
    parser.add_argument("--max_line", type=int, default=0, required=False)
    parser.add_argument("--max_length", type=int, default=4096, required=False)

    # 程序逻辑处理参数
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--replace_topk", type=int, default=20, required=False)
    parser.add_argument("--generate_type", type=str, default="prompt", required=False)
    # prompt：通过提示文本生成，或是topk：通过LM的topk逻辑生成, dict:通过映射词典进行替换
    parser.add_argument("--generate_context", type=str, default="seperate", required=False)
    # 选择topk时只能默认使用prefix内容
    # 当选择prompt进行替换式，使用前文（prefix）还是使用全局（global)信息，还是不适用语境（seperate, seperate_with_type）进行替换
    parser.add_argument("--generate_error_thresold", type=float, default=0.2, required=False)
    parser.add_argument("--output_text_head", type=str, required=False, default="")
    parser.add_argument("--replace_place_holder", type=str, required=False, default="<place_holder>")

    # debug相关参数
    parser.add_argument('--is_debug', action="store_true")
    parser.add_argument('--debug_line', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42, required=False)
    flags = parser.parse_args()

    # 部分参数调整
    if flags.max_line > 0:
        flags.debug_line = min(flags.debug_line, flags.max_line)

    # 输出参数信息
    print("[INFO] flags: ", flags, file=sys.stderr, flush=True)

    # 部分参数合法性校验
    assert os.path.exists(flags.model_path), "model_path {0} not exist".format(flags.model_path)
    assert os.path.exists(flags.input_text_path), "input_text_path {0} not exist".format(flags.input_text_path)
    assert os.path.exists(flags.input_entity_path), "input_entity_path {0} not exist".format(flags.input_entity_path)
    assert flags.generate_type in ["topk", "prompt", "dict"], "generate_type must be a value in topk, prompt, or dict"
    assert flags.generate_context in ["prefix", "global", "seperate", "seperate_with_type", "only_type"], \
        "generate_context must be a value in prefix, global, or seperate, seperate_with_type, only_type."

    # 读取repeat_prompt，加入到flags的空间中
    flags.repeat_prompt = read_prompt(flags.repeat_prompt_path)

    # 如果generate_type是prompt，则额外读取一个prompt
    if flags.generate_type == "prompt":
        flags.generate_prompt = read_prompt(flags.generate_prompt_path)

    return flags


# 读取预训练好的模型
def from_pretrained(flags):
    # 载入模型部分
    start_time = time.time()
    # 模型路径
    model_name = flags.model_path
    # 载入模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="sequential"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    input_entity = list()
    # 读取文本数据
    with open(flags.input_text_path, "r", encoding="utf8") as input:
        print("[INFO] reading input text file: {0}".format(flags.input_text_path), file=sys.stderr, flush=True)
        count = 0
        for line in input:
            line = line.strip()
            input_text.append(json.loads(line))
            count += 1
            if flags.max_line > 0 and count >= flags.max_line:
                break

    # 读取entity数据
    if flags.generate_type != "dict":
        # entity的种类替换采用hard coding
        entity_type_proj = {"LOC": "location", "PER": "person", "ORG": "organization"}
        with open(flags.input_entity_path, "r", encoding="utf8") as input:
            print("[INFO] reading input entity file: {0}".format(flags.input_entity_path), file=sys.stderr, flush=True)
            count = 0
            for line in input:
                line = line.strip()
                data_dict = json.loads(line)
                data_line = list()
                for key in data_dict.keys():
                    data_line.extend([(x, entity_type_proj[key]) for x in data_dict[key]])
                input_entity.append(data_line)
                count += 1
                if flags.max_line > 0 and count >= flags.max_line:
                    break
    else:
        with open(flags.input_dict_path, "r", encoding="utf8") as input:
            print("[INFO] reading input entity dict file: {0}".format(flags.input_dict_path), file=sys.stderr,
                  flush=True)
            count = 0
            for line in input:
                line = line.strip()
                data_dict = json.loads(line)
                input_entity.append(data_dict)
                count += 1
                if flags.max_line > 0 and count >= flags.max_line:
                    break

    # 验证并输出部分数据进行参考
    assert len(input_text) == len(input_entity), "input_text line {0} not matach input_entity line {1}".format(
        len(input_text), len(input_entity))
    if flags.is_debug:
        for idx, input_text_line in enumerate(input_text[:flags.debug_line]):
            print("[INFO] read_data debug. ")
            print(input_text_line)
            print(input_entity[idx])
            print()

    # 将输入数据合并成tuple
    input_data = [(input_text[idx], input_entity[idx]) for idx in range(len(input_text))]
    return input_data


# 执行生成式替换
def replace_generate(input_data, model, tokenizer, flags):
    print("[INFO] processing {0} lines of input data".format(len(input_data)), file=sys.stderr, flush=True)

    # 将整体结果保存出来
    result_list = list()
    # 先仅采用repeat_prompt，后续根据情况使用
    prompt = flags.repeat_prompt

    # 对数据进行处理
    for input_text, input_entity in tqdm(input_data):
        # 初始化部分值
        prompt_text = prompt.format(input_text)
        replace_list = preprocess_replace_list(input_text, input_entity, tokenizer, flags)  # 这里的逻辑还需要进一步处理
        # 执行处理
        output_list = list()
        output_flag = 0
        if flags.is_debug:
            print("###prompt: \n{0}".format(prompt_text))
            print("###replace_list:\n", replace_list)
        # 根据输入长度判断循环次数
        output_length = min(flags.max_length, len(tokenizer.tokenize(prompt_text)))
        # 开始输出
        for _ in range(output_length):
            prefix_input_text = "".join(output_list[-10:])
            # 获得next_token
            next_token = replace_generate_per_step(prompt_text, input_text, prefix_input_text, replace_list, model,
                                                   tokenizer, flags)
            # 判断是否提前结束
            if flags.early_stop and next_token.strip() == "<|im_end|>":
                if flags.is_debug:
                    print("[INFO] early stop at {0}".format(next_token))
                break
            # 准备下一次的生成处理
            prompt_text += next_token
            output_list.append(next_token)

        if flags.is_debug:
            print("[INFO] replace_generate debug. ")
            print("###input:\n", input_text)
            print("###output:\n", "".join(output_list))
            print()

        # 整理输出结果
        output_text = "".join(output_list)
        if flags.output_text_head != "":
            output_text = output_text.replace(flags.output_text_head, "").strip()
        result_list.append(output_text)

    return result_list


# 准备好需要替换的entity
def preprocess_replace_list(input_text, input_entity, tokenizer, flags):
    # 对input_entity进行可追溯的处理
    # 进行一定的初始化
    replace_list = {"tokenized": list(),
                    "original": list(),
                    "type": list(),
                    "dict": dict()}

    if flags.generate_type == "dict":
        entity_list = input_entity.keys()
    else:
        entity_list = [x[0] for x in input_entity]

    # 从整句的分割结果中找出词汇的分割结果
    # 保证词汇的分割粒度与句子的分割粒度的一致性
    # 先对text进行分割
    tokenized_input_text = [x.replace("Ġ", " ") for x in tokenizer.tokenize(input_text)]
    for idx, tok in enumerate(tokenized_input_text):
        tok = tok.strip().lower()
        for entity_idx, entity in enumerate(entity_list):
            if flags.generate_type == "dict":
                # 保存一下映射结果
                replace_list["dict"].update({entity: input_entity[entity]})
            # 开始匹配
            entity_lower = entity.lower()
            if entity_lower.startswith(tok):
                # 匹配成功，继续进一步匹配
                for idy in range(idx, len(tokenized_input_text)):
                    tok_temp = "".join(tokenized_input_text[idx:idy + 1]).lower().strip()
                    if not entity_lower.startswith(tok_temp):
                        break
                tok_temp = "".join(tokenized_input_text[idx:idy]).lower().strip()
                if tok_temp == entity_lower:
                    replace_list["tokenized"].append(tokenized_input_text[idx:idy])
                    replace_list["original"].append(entity)
                    if flags.generate_type != "dict":
                        replace_list["type"].append(input_entity[entity_idx][1])

    # 为了保证匹配的正确性
    # 增加分割的方案
    for entity_idx, entity in enumerate(entity_list):
        # 同时也单独对单词进行分割
        replace_list["tokenized"].append([x.replace("Ġ", " ") for x in tokenizer.tokenize(entity)])
        replace_list["original"].append(entity)
        if flags.generate_type != "dict":
            replace_list["type"].append(input_entity[entity_idx][1])

        # 直接用空格隔开
        replace_list["tokenized"].append([x.strip() for x in entity.split()])
        replace_list["original"].append(entity)
        if flags.generate_type != "dict":
            replace_list["type"].append(input_entity[entity_idx][1])

    return replace_list


# 生成式替换的每一步
def replace_generate_per_step(prompt_text, raw_input_text, prefix_input_text, replace_list, model, tokenizer, flags):
    # 获得下一次token，以及对应的logits分布
    next_token, next_token_logits = generate_next_token(prompt_text, model, tokenizer, flags)
    # 判断next_token是否是要被替换掉的词汇
    replace_flag = need_replace(next_token, replace_list, model, tokenizer, prompt_text, flags)
    if replace_flag != -1:
        replace_word = replace_list["original"][replace_flag]
        if flags.generate_type != "dict":
            replace_word_type = replace_list["type"][replace_flag]
        else:
            replace_word_type = ""
        if flags.is_debug: print("replace words \"{0}\", type: {1}".format(replace_word, replace_word_type))

        # 需要被替换，则进行替换生成
        if flags.generate_type == "topk":
            # 选择使用下k个单词
            next_token = choose_from_topk(next_token, tokenizer, next_token_logits, flags)
        elif flags.generate_type == "prompt":
            # 选择通过一次LLM提示工程的生成来执行替换
            # 防止重复计算以及随机性带来的替换不一致，加入替换
            if replace_word in replace_list["dict"]:
                response = replace_list["dict"][replace_word]
            else:
                # 根据情况拼接提示文本
                if flags.generate_context == "prefix":
                    # 使用前文进行生成
                    exit(1)  # 效果不佳，暂时舍弃这种方案
                    generate_prompt_text = flags.generate_prompt.format(prefix_input_text.strip() + " " + replace_word,
                                                                        replace_word)
                elif flags.generate_context == "global":
                    # 使用全文进行替换生成
                    exit(1)  # 效果不佳，暂时舍弃这种方案
                    generate_prompt_text = flags.generate_prompt.format(
                        raw_input_text.replace(replace_word, flags.replace_place_holder))
                elif flags.generate_context == "seperate_with_type":
                    generate_prompt_text = flags.generate_prompt.format(replace_word, replace_word_type)
                elif flags.generate_context == "seperate":
                    generate_prompt_text = flags.generate_prompt.format(replace_word)
                elif flags.generate_context == "only_type":
                    generate_prompt_text = flags.generate_prompt.format(replace_word_type)
                else:
                    exit(1)
                # 执行生成
                response = generate_response(generate_prompt_text, model, tokenizer, flags)
                response = response.strip()
                replace_list["dict"].update({replace_word: response})

            next_token = " " + response
        elif flags.generate_type == "dict":
            # 在恢复流程中，选择dict中的词汇作为替代单词
            if replace_word in replace_list["dict"]:
                response = replace_list["dict"][replace_word]
            else:
                response = replace_word

            next_token = " " + response

    if (flags.is_debug): print("next token: ***", next_token, "***")

    return next_token


# 从topk中选择合适的词汇作为next_token
def choose_from_topk(next_token, tokenizer, next_token_logits, flags):
    # hard coding定义停用词
    stopword_list = ["the", "The", ",", "\"", "\'", ".", "...", "input"],
    # 获取topk的词
    next_token_logits_top_k, next_token_id_top_k = torch.topk(next_token_logits, k=flags.replace_topk, largest=True,
                                                              sorted=True)
    if (flags.is_debug): print("top {0} token: ".format(flags.replace_topk),
                               [tokenizer.decode(x, skip_special_token=True) for x in next_token_id_top_k])
    # 进行可信性判断
    for temp_next_token_id in next_token_id_top_k:
        temp_next_token = tokenizer.decode(temp_next_token_id, skip_special_token=True)
        temp_next_token = temp_next_token.strip()
        if temp_next_token.strip().lower() == "the":
            # the词汇，停用
            continue
        elif temp_next_token.strip().lower() in stopword_list:
            # 停用词，舍弃
            continue
        elif temp_next_token.strip() == "":
            # 回车，空格等空行，舍弃
            continue
        elif temp_next_token.strip().lower() == next_token.strip().lower():
            # 原token的等价变形，舍弃
            continue
        elif temp_next_token.startswith("\"") or temp_next_token.startswith("\'") or temp_next_token.startswith(".") \
                or temp_next_token.startswith("_") or temp_next_token.startswith("<"):
            # 无意义开头，舍弃
            continue
        elif len(temp_next_token.strip()) <= 1:
            # 只有一个字母影响太大，也进行舍弃
            continue
        return temp_next_token

    # 没有可用候选值时则返回原值
    return next_token


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
def need_replace(next_token, replace_list, model, tokenizer, prompt_text, flags):
    for idx, replace_target_list in enumerate(replace_list["tokenized"]):
        matched = True
        for replace_target in replace_target_list:
            if replace_target.strip().lower() == next_token.strip().lower():
                # if flags.is_debug: print("***{0}*** matches ***{1}***".format(next_token, replace_target))
                # 继续匹配下一个
                prompt_text += next_token
                # 这次采用假设执行
                # 假设next token不做替换式，下一次的输出是什么
                next_token, _ = generate_next_token(prompt_text, model, tokenizer, flags)
            else:
                # 匹配失败
                matched = False
                break
        if matched:
            return idx
    return -1


# 输出处理
def output_results(results, input_data, tokenizer, flags):
    # 输出替换结果
    output_result_path = flags.output_file_prefix + ".result"
    print("[INFO] writing results in {0}".format(output_result_path), file=sys.stderr, flush=True)
    with open(output_result_path, "w", encoding="utf8") as output:
        for idx, result in enumerate(results):
            print(json.dumps(result), file=output, flush=True)
    # 如果不是dict处理逻辑，则额外输出dict映射信息
    if flags.generate_type != "dict":
        output_dict_path = flags.output_file_prefix + ".entity_dict"
        print("[INFO] writing entity replace dict in {0}".format(output_dict_path), file=sys.stderr, flush=True)
        with open(output_dict_path, "w", encoding="utf8") as output:
            for idx, result in enumerate(results):
                input_text, input_entity = input_data[idx]
                entity_dict = process_entity_dict(result, input_text, input_entity, tokenizer, flags)
                print(json.dumps(entity_dict), file=output, flush=True)


# 处理替换映射表
def process_entity_dict(result_text, input_text, input_entity, tokenizer, flags):
    entity_dict = dict()

    # 分词处理
    tokenized_input_text = [x for x in tokenizer.tokenize(input_text.replace("\n", " "))]
    tokenized_result_text = [x for x in tokenizer.tokenize(result_text.replace("\n", " "))]

    if len(tokenized_input_text) > (1 + flags.generate_error_thresold) * len(tokenized_result_text) \
            or len(tokenized_input_text) < (1 - flags.generate_error_thresold) * len(tokenized_result_text):
        return entity_dict

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
        elif replace_tok != "" and replaced_tok != "":
            entity_dict[replaced_tok.replace("Ġ", " ").strip()] = replace_tok.replace("Ġ", " ").strip()
            replace_tok = ""
            replaced_tok = ""
    if replace_tok != "" and replaced_tok != "":
        entity_dict[replaced_tok.replace("Ġ", " ").strip()] = replace_tok.replace("Ġ", " ").strip()

    return entity_dict


def main():
    # 解析参数
    flags = parse_args(sys.argv)
    # 读取模型和分词器
    model, tokenizer = from_pretrained(flags)
    # 固定种子值，保证实验结果的稳定
    set_random_seed(flags.seed)
    # 读取数据
    input_data = read_data(flags)
    # 执行生成式替换
    results = replace_generate(input_data, model, tokenizer, flags)
    # 执行输出处理
    output_results(results, input_data, tokenizer, flags)


if __name__ == '__main__':
    main()
