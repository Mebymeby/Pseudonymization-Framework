import ast
import glob
import json
import os


# 需要去除模型解释部分的 推理 cache 文件，将该脚本文件放到同级目录运行

def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [ast.literal_eval(line.strip()) for line in file]


def remove_text_after_double_newline(text):
    return text.replace('\n', '').replace('\r', '')


if __name__ == '__main__':
    # 读取当前文件夹的所有结果文件
    cache_files = glob.glob('*.txt')

    # 遍历结果文件
    for cache_file in cache_files:
        # 加载结果字典列表
        content_list = load_cache_file(cache_file)

        # 创建一个新文件去写入
        new_cache_path = os.path.join("RMLB-" + cache_file)
        with open(new_cache_path, 'a', encoding='utf-8') as file:
            for line in content_list:
                line["document"] = remove_text_after_double_newline(line["document"])
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
