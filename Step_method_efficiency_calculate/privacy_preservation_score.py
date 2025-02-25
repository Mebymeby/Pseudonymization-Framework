import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def calculate_privacy_preservation(ent1, ent2):
    eb1 = embedding_model.encode(ent1, convert_to_tensor=True)
    eb2 = embedding_model.encode(ent2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(eb1, eb2).squeeze(0).item()
    return sim


if __name__ == '__main__':
    # 加载 embedding 模型
    embedding_model_path = os.path.join("Model", "all-mpnet-base-v2")
    embedding_model = SentenceTransformer(embedding_model_path).to("cuda")

    # 读取entity map
    rand_entity_map = []
    prompt_entity_map = []

    with open(os.path.join("Step_method_efficiency_calculate", "entity_map", "rand_entity_map.txt"), "r",
              encoding="utf-8") as f:
        for entity_line in f:
            rand_entity_map.append(json.loads(entity_line.strip()))
    with open(os.path.join("Step_method_efficiency_calculate", "entity_map", "prompt_entity_map.txt"), "r",
              encoding="utf-8") as f:
        for entity_line in f:
            prompt_entity_map.append(json.loads(entity_line.strip()))

    privacy_preservation_score_list_map = {
        "rand"  : [],
        "prompt": [],
    }

    for idx in tqdm(range(len(rand_entity_map))):
        cur_item_rand_entity_map = rand_entity_map[idx]
        cur_item_prompt_entity_map = prompt_entity_map[idx]

        for key, value in cur_item_rand_entity_map.items():
            sim = calculate_privacy_preservation(key, value)
            privacy_preservation_score_list_map["rand"].append(sim)
        for key, value in cur_item_prompt_entity_map.items():
            sim = calculate_privacy_preservation(key, value)
            privacy_preservation_score_list_map["prompt"].append(sim)

    privacy_preservation_score = {key: np.mean(val) for key, val in privacy_preservation_score_list_map.items()}

    print(privacy_preservation_score)
