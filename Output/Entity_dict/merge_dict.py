import json
import os


# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 写入 JSON 文件
def save_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def is_valid_item(item):
    start_with_list = ["#", "@", "&", "'", '"', "-", "+", ",", ".", "<", ">", "/"]

    if item[0] in start_with_list:
        return False
    if len(item) < 3:
        return False
    return True


# 合并多个 JSON 数据
def merge_multiple_json():
    merged = {}

    for file in json_files:
        json_data = load_json(file)  # 读取当前 JSON 文件
        for key, values in json_data.items():  # 当前 values 为某一实体列表
            if key not in merged:
                merged[key] = set()  # 使用集合去重
            merged[key].update(values)  # 添加数据到集合，自动去重

    for key, values in merged.items():
        merged[key] = [value for value in values if is_valid_item(value)]  # 做额外处理

    return {key: sorted(values) for key, values in merged.items()}  # 对每个 key 的列表进行排序（默认按字母顺序）


if __name__ == '__main__':
    # 指定存放 JSON 文件的目录，获取目录中所有 JSON 文件文件描述符列表
    json_folder = os.path.join("Output", "Entity_dict")
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]

    # 处理多个 JSON 文件
    merged_data = merge_multiple_json()

    # 保存处理后的 JSON 文件
    save_json(merged_data, os.path.join(json_folder, "merged.json"))
