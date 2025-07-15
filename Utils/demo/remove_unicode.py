import ast
import glob
import json
import os


def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [ast.literal_eval(line.strip()) for line in file]


# 重新编码文件
if __name__ == '__main__':
    # 读取当前文件夹的所有结果文件
    cache_files = glob.glob('*.txt')

    # 遍历结果文件
    for cache_file in cache_files:
        # 加载结果字典列表
        content_list = load_cache_file(cache_file)

        # 创建一个新文件去写入
        new_cache_path = os.path.join("RM_UNICODE-" + cache_file)
        with open(new_cache_path, 'a', encoding='utf-8') as file:
            for line in content_list:
                for key in line.keys():
                    line[key] = line[key].encode('utf-8', errors='replace').decode('utf-8')
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
