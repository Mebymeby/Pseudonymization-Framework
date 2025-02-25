import json
import os

import datasets

from Utils import config_util


def parse_dataset_path():
    dataset_path = os.path.join("Dataset", dataset_config.path)
    if os.name != 'nt':
        dataset_path = os.path.join("Dataset", dataset_path)
    return dataset_path


if __name__ == "__main__":

    dataset_name = "xsum"

    # 加载数据
    dataset_config = config_util.Config(os.path.join("Config", "dataset_config", dataset_name))
    # 加载数据
    data = datasets.load_from_disk(parse_dataset_path())
    print(data.shape)

    data = data["test"]
    # print(data.shape)
    data = data.select(range(796, 798))
    for idx, item in enumerate(data):
        print(json.dumps(item, indent=4, ensure_ascii=False))
        input_len = len(item["document"])
        print(f"idx:{idx} length:{input_len}")
