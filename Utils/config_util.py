import json
import os


# 旧脚本用法，先保留
def load_config(sub_config_name=""):
    config_path = os.path.join(os.getcwd(), "Config", "config.json")
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    if sub_config_name != "":
        config = config[sub_config_name]
    return config


# 配置参数对象化
class Config:
    def __init__(self, filepath):
        with open(filepath + ".json", 'r', encoding='utf-8') as f:
            self._config = json.load(f)

    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"当前配置文件中不存在属性: {name}")

    def print_config(self):
        return json.dumps(self._config, indent=4, ensure_ascii=False)
