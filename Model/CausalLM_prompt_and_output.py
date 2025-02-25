import json
import os.path
import re
import datasets
import torch
from Config import prompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm_model(model_path, torch_dtype=torch.bfloat16, device_map="auto"):
    _eval_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _eval_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    return _eval_model, _eval_tokenizer


def load_data_item(dataset_name, split, index):
    item = datasets.load_from_disk(os.path.join("Dataset", dataset_name))[split][index]
    return item, dataset_name


def get_dataset_input_keys(dataset_name):
    task_input_keys = {
        "squad": ["context", "question"],
        "xsum": ["document"]
    }
    return task_input_keys[dataset_name]


def generate_task_input(item, input_keys):
    task_input = ""
    for key in input_keys:
        task_input += ("\n" + key + ": " + item[key])
    return task_input


def generate_task_prompt(task, item, dataset_name):
    return f"""
{prompt.TASK(task)}

{generate_task_input(item, get_dataset_input_keys(dataset_name))}
"""


def task_with_llm(task_input):
    inputs = llm_tokenizer(task_input, return_tensors="pt").to(llm_model.device)
    raw_outputs = llm_model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    output = llm_tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
    # print(output)

    # 解析输出部分
    matches = re.findall(r'{.*?}', output)
    if matches:
        last_match_str = matches[-1]
        json_data = json.loads(last_match_str)
        for _, value in json_data.items():
            print(value)


if __name__ == "__main__":
    llm_model_path = "Model/Qwen2.5-1.5B-Instruct"
    llm_model, llm_tokenizer = load_llm_model(llm_model_path)


    # QA
    task_with_llm(generate_task_prompt("QuestionAnswering", load_data_item("squad", "validation", 123)))
    # TS
    task_with_llm(generate_task_prompt("TextSummarization", load_data_item("xsum", "validation", 321)))
