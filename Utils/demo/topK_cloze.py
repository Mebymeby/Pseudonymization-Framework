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


def generate_with_topk_replacement(prefix, next_target):
    input_string = prefix + "[MASK]" + next_target
    input_prompt = """<|im_start|>system
Fill the [MASK] tag in the text given by the user with the appropriate word.
Output this word.

The user input format is as follows:
Text: 
The assistant output format is as follows:
Word: 

Here is an example:
Text: Li Si is the boss of[MASK] in
Word:  ByteDance
<|im_end|>

<|im_start|>user
Text: {0}
<|im_end|>

<|im_start|>assistant
Word:
""".format(input_string)

    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        num_beams=1
    )

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"best_candidate: '{decoded_output}'")


if __name__ == '__main__':
    # 加载模型
    model_path = os.path.join("Model", "Qwen2.5-1.5B-Instruct")
    tokenizer, model = load_topk_model(model_path)

    # 模拟输入
    text = "Zhangsan works for Tencent in ShenZhen"
    entity_list_map = {"Zhangsan": "PER", "Tencent": "ORG", "ShenZhen": "LOC"}

    prefix_input = "Zhangsan works for"

    suffix_input = " in"

    with torch.inference_mode():
        generate_with_topk_replacement(prefix_input, suffix_input)
