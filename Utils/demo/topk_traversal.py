import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_topk_model(model_path, torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    model.eval()
    return tokenizer, model


# 简单过滤
def is_valid(token_id):
    token = tokenizer.decode(token_id, skip_special_tokens=True).replace("Ġ", "").strip()
    return token.isalpha() and len(token) > 1


def generate_with_topk_replacement(prefix, next_target, topK, model, tokenizer):
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id
    )

    target_position = len(inputs.input_ids[0])
    scores = outputs.scores[0]
    top_k = torch.topk(scores, k=topK, dim=-1)

    ori_token_id = outputs.sequences[0, target_position]

    candidates_token_id = [token_id.item() for token_id in top_k.indices[0]
                           if token_id != ori_token_id and is_valid(token_id)]

    # 打印候选token及其解码结果
    print("valid topK tokens: ")
    for token_id in candidates_token_id:
        token = tokenizer.decode(token_id, skip_special_tokens=True)
        print(f"ID: {token_id} → Text: '{token}'")

    if not candidates_token_id:
        print("No valid candidates_token_id found.")
        return

    # 验证候选 token 合理性
    best_candidate = None
    best_score = -float('inf')

    for candidate in candidates_token_id:
        # 拼接新输入
        new_text = prefix + tokenizer.decode(candidate)
        new_inputs = tokenizer(new_text, return_tensors="pt").to(model.device)

        # 计算后续目标 token 的概率
        outputs = model(**new_inputs)
        next_token_logits = outputs.logits[0, -1, :]
        key_token_id = tokenizer.encode(next_target, add_special_tokens=False)[0]
        key_token_prob = torch.softmax(next_token_logits, dim=-1)[key_token_id].item()

        if key_token_prob > best_score:
            best_score = key_token_prob
            best_candidate = candidate

    # 输出最佳 token
    print(f"best_candidate: '{tokenizer.decode(best_candidate, skip_special_tokens=True)}'")


if __name__ == '__main__':
    # 加载模型
    model_path = os.path.join("Model", "Qwen2.5-1.5B-Instruct")
    tokenizer, model = load_topk_model(model_path)

    # 模拟输入
    text = "Zhangsan works for Tencent in ShenZhen"
    entity_list_map = {"Zhangsan": "PER", "Tencent": "ORG", "ShenZhen": "LOC"}

    input_prompt = "Zhangsan works for"

    next_target = " in"

    topK_range = 30

    with torch.inference_mode():
        generate_with_topk_replacement(input_prompt, next_target, topK_range, model, tokenizer)
