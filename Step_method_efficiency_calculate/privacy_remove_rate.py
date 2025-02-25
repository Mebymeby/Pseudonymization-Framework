import json
import os

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    # 加载entity map文件，获取同一个数据集每一个方案的entity keys
    ner_entity_map = []
    prompt_entity_map = []
    tag_mask_entity_map = []
    tag_rep_entity_map = []

    # 读取每一行NER的实体结果
    with open(os.path.join("entity_list", "ner_entity_list.txt"), "r", encoding="utf-8") as f:
        for entity_line in f:
            ner_entity_map.append([key for key in json.loads(entity_line.strip()).keys()])
    with open(os.path.join("entity_list", "prompt_entity_list.txt"), "r", encoding="utf-8") as f:
        for entity_line in f:
            prompt_entity_map.append([key for key in json.loads(entity_line.strip()).keys()])
    with open(os.path.join("entity_list", "tag_mask_entity_list.txt"), "r", encoding="utf-8") as f:
        for entity_line in f:
            tag_mask_entity_map.append(json.loads(entity_line.strip()).keys())
    with open(os.path.join("entity_list", "tag_rep_entity_list.txt"), "r", encoding="utf-8") as f:
        for entity_line in f:
            tag_rep_entity_map.append([key for key in json.loads(entity_line.strip()).keys()])

    # 记录数据
    detection_method_key = ["ner", "prompt", "tag_mask", "tag_rep"]

    all_detection_entities_list = []
    for idx in range(len(ner_entity_map)):
        current_idx_entities = list(set(ner_entity_map[idx] + prompt_entity_map[idx] + tag_rep_entity_map[idx]))
        all_detection_entities_list.append(current_idx_entities)

    privacy_remove_rate_list = {key: [] for key in detection_method_key}
    for idx, one_item_ner_entities in enumerate(tqdm(all_detection_entities_list)):
        privacy_remove_hit_count = {key: 0 for key in detection_method_key}
        ner_entity_num = len(one_item_ner_entities)
        for ner_entity in ner_entity_map[idx]:
            if ner_entity in one_item_ner_entities:
                privacy_remove_hit_count["ner"] += 1
        for prompt_entity in prompt_entity_map[idx]:
            if prompt_entity in one_item_ner_entities:
                privacy_remove_hit_count["prompt"] += 1
        for tag_mask_entity in tag_mask_entity_map[idx]:
            if tag_mask_entity in one_item_ner_entities:
                privacy_remove_hit_count["tag_mask"] += 1
        for tag_rep_entity in tag_rep_entity_map[idx]:
            if tag_rep_entity in one_item_ner_entities:
                privacy_remove_hit_count["tag_rep"] += 1

        for key in detection_method_key:
            privacy_remove_rate_list[key].append(
                privacy_remove_hit_count[key] / ner_entity_num if ner_entity_num > 0 else 1)

    privacy_remove_rate = {key: np.mean(val) for key, val in privacy_remove_rate_list.items()}

    print(privacy_remove_rate)
