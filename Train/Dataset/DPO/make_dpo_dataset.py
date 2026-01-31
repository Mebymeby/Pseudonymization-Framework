import argparse
import json
import os

import datasets
from tqdm import tqdm

from Config import prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--dataset_key", type=str, default="document")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--data_start", type=int, default=0)
    parser.add_argument("--data_size", type=int, default=5000)
    parser.add_argument(
        "--target_path", type=str, default=os.path.join("Train", "Dataset")
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 参数
    args = parse_args()
    args.resource_path = "" if os.name == "nt" else "/longzi05/houshilong/"

    # 加载数据集
    dataset_path = os.path.join(args.resource_path, "Dataset", args.dataset)
    dataset = datasets.load_from_disk(dataset_path)[args.data_split].select(
        range(args.data_start, args.data_size)
    )

    # 预加载数据集内容到内存，提高访问效率
    dataset_content_cache = {item["id"]: item[args.dataset_key] for item in dataset}

    # 读取 rank 文件
    rank_file = os.path.join(
        args.target_path, f"{args.dataset}_{args.data_split}_rank.json"
    )
    with open(rank_file, "r") as f:
        rank_data = json.load(f)

    # dpo 数据
    dpo_data = []

    # 遍历 rank_data 中的每个样本
    for item_id, item in tqdm(rank_data.items(), desc="Processing Dpo Data"):
        if len(item) < 2:
            continue

        # 根据 rank 文件，为每个 item 选取 dpo 所需的 chosen 和 rejected 数据
        chosen = item[0]
        rejected = item[1]

        # 使用缓存的数据集内容
        prompt_text = prompt.TASK["llm_end2end_long"].format(
            dataset_content_cache[item_id]
        )

        # 将数据写入 dpo 训练数据
        dpo_data.append(
            {
                "prompt": prompt_text,
                "chosen": f"<NED>{json.dumps(chosen['NED'])}</NED>\n<ERM>{json.dumps(chosen['ERM'])}</ERM>\n<ATG>{chosen['ATG']}</ATG>",
                "rejected": f"<NED>{json.dumps(rejected['NED'])}</NED>\n<ERM>{json.dumps(rejected['ERM'])}</ERM>\n<ATG>{rejected['ATG']}</ATG>",
            }
        )

    # 保存 dpo 训练数据
    dpo_data_save_path = os.path.join(
        args.target_path, f"{args.dataset}_{args.data_split}_dpo.json"
    )
    os.makedirs(os.path.dirname(dpo_data_save_path), exist_ok=True)
    with open(dpo_data_save_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=4)

    print(f"DPO数据生成完成，共生成 {len(dpo_data)} 条数据")
