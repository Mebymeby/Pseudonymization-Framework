import re

import torch

from Utils import dataset_util
from Config import prompt


# 解析JSON
def extract_llm_output(decoded_output):
    pattern = r'\{\s*"([^"]+)"\s*:\s*"([^"]+)"\s*\}'  # 直接将所有JSON字符串识别并抽取为二元组列表
    matches = re.findall(pattern, decoded_output)  # 找出所有匹配项
    return matches[0][1] if matches else ""


# 统一推理函数：根据模型、数据、任务类型适配推理逻辑
def eval_inference(item, eval_model, eval_tokenizer, eval_model_config, dataset_config):
    # LLM 模型需要端到端推理
    if eval_model_config.load_type == "CausalLM":
        # 附加提示工程，构造统一输入序列
        if dataset_config.name == "wikitext-103":
            input_prompt = prompt.DATASET[dataset_config.name].format(" ".join(item["text"].split()[:50]))  # 输入前50个词
        if dataset_config.name == "wmt14-de-en":
            input_prompt = prompt.DATASET[dataset_config.name].format(item["en"])
        else:
            input_prompt = dataset_util.generate_dataset_prompt(item, dataset_config)

        # 生成token
        input_token = eval_tokenizer(input_prompt, return_tensors="pt").to(eval_model.device)

        # 执行推理，获得原始输出
        raw_output = eval_model.generate(
            **input_token,
            max_new_tokens=dataset_config.LLM_max_new_tokens,
            do_sample=False,
            num_beams=1
        )

        # 解码原始输出
        ## 方式一
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(input_token.input_ids, raw_output)]
        decoded_output = eval_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        ## 方式二（prompt 可能会因 tokenizer 的合并规则而发生变化，可能导致截取错位）
        # decoded_output = eval_tokenizer.decode(raw_output[0], skip_special_tokens=True)
        # decoded_output = decoded_output[len(input_prompt):].strip()

        # 获得有效输出
        if dataset_config.name == "wikitext-103" or dataset_config.name == "wmt14-de-en":
            final_output = decoded_output
        else:
            final_output = extract_llm_output(decoded_output)

        return final_output, decoded_output

    # 微调模型单独适配
    if eval_model_config.load_type == "QuestionAnswering":
        return eval_model(question=item["question"], context=item["context"])["answer"], ""


def prompt_inference(tool_model, tool_tokenizer, prompt, max_new_tokens=32):
    # 生成token
    input_token = tool_tokenizer(prompt, return_tensors="pt").to(tool_model.device)

    # 执行推理，获得原始输出
    raw_output = tool_model.generate(
        **input_token,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1
    )

    # 获得并解码模型生成的新内容
    ## 方式一
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_token.input_ids, raw_output)]
    decoded_output = tool_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    ## 方式二（prompt 可能会因 tokenizer 的合并规则而发生变化，可能导致截取错位）
    # decoded_output = tokenizer.decode(raw_output[0], skip_special_tokens=True)
    # decoded_output = decoded_output[len(prompt):].strip()

    return decoded_output


# 生成 next token
def next_token_inference(input_text, model, tokenizer):
    # 获取当前输入的 token
    input_text_token = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    # 执行一次前向传播
    outputs = model(input_text_token)

    # 获取所有输出的概率分布
    logits = outputs.logits

    # 得到当前输入的下一个单词的概率分布
    next_token_logits = logits[:, -1, :].squeeze()  # size: [vocab]

    # 通过 argmax 获取概率分布的最大值作为 next token
    next_token_id = torch.argmax(next_token_logits)

    # 解码 token
    next_token = tokenizer.decode(next_token_id, skip_special_token=True)

    return next_token, next_token_logits
