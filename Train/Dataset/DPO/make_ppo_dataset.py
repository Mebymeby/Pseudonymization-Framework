import json
import os

from tqdm import tqdm


def make_ppo_dataset():
    # 读取源数据文件
    source_file = os.path.join("Train", "RL","Dataset", "xsum_train_5000_ner_rand_direct_dpo.json")
    with open(source_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备DPO数据集
    ppo_data = []

    for item in tqdm(data):
        prompt = item["prompt"]
 
        # 添加到DPO数据集中
        ppo_data.append({
            "prompt": prompt
        })

    # 保存DPO数据集
    save_path = os.path.join("Train", "RL","Dataset")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "xsum_train_5000_ner_rand_direct_ppo.json")

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(ppo_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    make_ppo_dataset()
