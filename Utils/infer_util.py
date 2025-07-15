import re

import torch

from Utils import dataset_util


# 解析JSON
def extract_llm_output(decoded_output):
    pattern = r'\{\s*"([^"]+)"\s*:\s*"([^"]+)"\s*\}'  # 直接将所有JSON字符串识别并抽取为二元组列表
    matches = re.findall(pattern, decoded_output)  # 找出所有匹配项
    return matches[0][1] if matches else ""


# 统一推理函数：根据模型、数据、任务类型适配推理逻辑
def eval_inference(item, eval_model, eval_tokenizer, dataset_config):
    # 获取 prompt
    infer_prompt = dataset_util.generate_dataset_infer_prompt(item, dataset_config)

    # 生成token
    input_token = eval_tokenizer(infer_prompt, return_tensors="pt").to(eval_model.device)

    # 执行推理，获得原始输出
    output_token = eval_model.generate(
        **input_token,
        max_new_tokens=dataset_config.infer_max_new_tokens,
        do_sample=False,
        num_beams=1
    )

    # 解码输出
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_token.input_ids, output_token)]
    decoded_output = eval_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return decoded_output


def prompt_inference(tool_model, tool_tokenizer, prompt, max_new_tokens=32):
    input_token = tool_tokenizer(prompt, return_tensors="pt").to(tool_model.device)
    raw_output = tool_model.generate(
        **input_token,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(input_token.input_ids, raw_output)]
    decoded_output = tool_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return decoded_output


# 根据当前输入内容生成 next_token 通过调用一次模型的前向传播实现
def next_token_inference(input_text, model, tokenizer):
    input_text_token = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    outputs = model(input_text_token)  # 执行一次前向传播

    logits = outputs.logits  # 获取所有输出的概率分布

    next_token_logits = logits[:, -1, :].squeeze()  # 得到当前输入的下一个单词的概率分布 size: [vocab]

    next_token_id = torch.argmax(next_token_logits)  # 通过 argmax 获取概率分布的最大值作为 next token

    next_token = tokenizer.decode(next_token_id, skip_special_token=True)  # 解码 token

    return next_token, next_token_logits


# 根据输入的实体文本，判断实体类型
def determine_entity_type(ori_entity, ner_model):
    ori_entity_info = ner_model(ori_entity.strip())

    if not ori_entity_info:
        print(f"[ERROR]: Can't determine entity type, entity: {ori_entity}")
    elif len(ori_entity_info) == 1:
        if ori_entity_info[0]["entity_group"] in ["PER", "LOC", "ORG"]:
            return ori_entity_info[0]["entity_group"]
    else:
        print(
            f"[WARRING]: Multiple entities were detected in the input: {ori_entity} | Multiple entity list: {ori_entity_info}")
        if ori_entity_info[0]["entity_group"] in ["PER", "LOC", "ORG"]:
            return ori_entity_info[0]["entity_group"]

    return "entity"
