import argparse
import ast
import json
import os

import datasets
from tqdm import tqdm


def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [ast.literal_eval(line.strip()) for line in file]


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--data_size", type=int, default=5000)
    parser.add_argument("--detect_method", type=str, default="ner")
    parser.add_argument("--generate_method", type=str, default="rand")
    parser.add_argument("--replace_method", type=str, default="direct")
    args = parser.parse_args()

    # 加载数据集
    dataset_path = os.path.join("/data4/houshilong/Dataset", args.dataset)
    dataset = datasets.load_from_disk(dataset_path)[args.data_split].select(range(0, args.data_size))

    # 读取中间结果文件
    cache_base_path = os.path.join("Output", "Cache", args.dataset)
    entity_list = load_cache_file(
        os.path.join(
            cache_base_path, "Detect", "Train",
            f"{args.detect_method}.txt"
        )
    )
    entity_map = load_cache_file(
        os.path.join(
            cache_base_path, "Generate", "Train",
            f"{args.detect_method}-{args.generate_method}.txt"
        )
    )
    pseu_text = load_cache_file(
        os.path.join(
            cache_base_path, "Replace", "Train",
            f"{args.detect_method}-{args.generate_method}-{args.replace_method}.txt"
        )
    )

    # 组织数据
    alpaca = []
    for i in tqdm(range(args.data_size)):
        alpaca.append({
            "instruction": """You are a privacy-focused text editor. Perform named entity anonymization by replacing all `PER` (Person), `LOC` (Location), and `ORG` (Organization) entities while preserving:
1. Original text semantics
2. Entity type consistency
3. Grammatical correctness

Execution Steps:
1. EXTRACT all `PER/LOC/ORG` entities from input text and REPLACE each entity with a plausible same-type substitute that is contextually appropriate.
2. GENERATE anonymized text using replacements.

Output Steps:
1. Entity Replacement Map. Strictly formatted as: <EMAP>{{"Extracted Entity1":"Replacement Entity1",...}}</EMAP>
2. Anonymized Text. Strictly formatted as: <ANYT>Anonymized Text</ANYT>

Example:
Input: Tom met Jerry at Microsoft in NewYork.
Output: <EMAP>{{"Tom":"Jhon","Jerry":"Alice","Microsoft":"Google","NewYork":"Paris"}}</EMAP>\n<ANYT>Jhon met Alice at Google in Paris.</ANYT>
""",
            "input": dataset[i]["document"],
            "output": "<EMAP>" + json.dumps(entity_map[i]) + "</EMAP>\n<ANYT>" + pseu_text[i]["document"] + "</ANYT>",
        })

    # 保存数据
    save_path = os.path.join("Train", "ft_dataset")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(
        save_path,
        f"{args.dataset}_{args.data_split}_{args.data_size}_{args.detect_method}_{args.generate_method}_{args.replace_method}.json"
    )

    with open(save_file, 'a', encoding='utf-8') as file:
        json.dump(alpaca, file, indent=4, ensure_ascii=False)
