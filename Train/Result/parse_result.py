import json
import os
import re
import argparse


def merge_result(data_path: str, output_filename: str = "merged_results.txt") -> str:
    """
    合并指定目录下的所有.txt文件到一个输出文件中。

    Args:
        data_path: 包含.txt文件的目录路径
        output_filename: 合并后的输出文件名

    Returns:
        str: 合并后的输出文件路径
    """
    # 获取data_path文件夹下所有.txt文件
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]

    # 按文件名排序
    files.sort()

    # 合并文件内容
    merged_content = ""
    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            merged_content += f.read()

    # 写入合并后的文件
    output_file = os.path.join(data_path, output_filename)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(merged_content)

    print(f"已成功合并{len(files)}个文件到{output_file}")
    return output_file


def validate_and_extract(content: str) -> tuple:
    """
    验证并提取内容中的NED、ERM和ATG标签。

    Args:
        id_: 数据的ID
        content: 包含标签的内容

    Returns:
        tuple: (提取结果字典或None, 失败类型或None)
        失败类型包括: "JSON", "ATG", "ERM", "NED"
    """
    # 检查是否包含且仅包含一对NED、ERM、ATG标签
    ned_matches = re.findall(r"<NED>(.*?)</NED>", content, re.DOTALL)
    erm_matches = re.findall(r"<ERM>(.*?)</ERM>", content, re.DOTALL)
    atg_matches = re.findall(r"<ATG>(.*?)</ATG>", content, re.DOTALL)

    # 格式解析
    if len(ned_matches) == 1:
        if len(erm_matches) == 1:
            if len(atg_matches) == 1:
                try:
                    # 解析NED和ERM内容
                    ned_content = json.loads(ned_matches[0])
                    erm_content = json.loads(erm_matches[0])
                    atg_content = atg_matches[0].strip()

                    return {
                        "NED": ned_content,
                        "ERM": erm_content,
                        "ATG": atg_content,
                    }, None
                except json.JSONDecodeError:
                    return None, "JSON"
            else:
                return None, "ATG"
        else:
            return None, "ERM"
    else:
        return None, "NED"


def parse_result(input_file: str) -> tuple:
    """
    解析输入文件中的每行数据，提取有效内容。

    Args:
        input_file: 输入文件路径

    Returns:
        tuple: (成功解析的数据列表, 各类失败数据的字典)
    """
    # 用于存储结果
    result_data = []
    failed_data = {
        "JSON": [],
        "NED": [],
        "ERM": [],
        "ATG": [],
    }

    # 读取源文件并处理每一行
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            try:
                # 解析JSON行
                data = json.loads(line)
                # 获取key和value
                key = list(data.keys())[0]
                value = data[key]

                # 验证并提取内容
                extracted, fail_type = validate_and_extract(value)
                if extracted:
                    # 构建目标格式的数据
                    target_data = {
                        "key": key,
                        "NED": extracted["NED"],
                        "ERM": extracted["ERM"],
                        "ATG": extracted["ATG"],
                    }
                    result_data.append(target_data)
                elif fail_type:
                    failed_data[fail_type].append({key: value})
            except json.JSONDecodeError:
                failed_data["JSON"].append({key: value})

    return result_data, failed_data


def write_json_file(file_path: str, data: list):
    """
    将数据写入JSON文件。

    Args:
        file_path: 输出文件路径
        data: 要写入的数据
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    # 增加参数解析目标文件夹路径
    parser = argparse.ArgumentParser(description="Parse result files")
    parser.add_argument("--data_path", type=str, default="./")
    args = parser.parse_args()

    print(f"目标路径: {args.data_path}")

    # 合并文件（如果需要）
    merged_file_name = "merged_results.txt"
    merged_file_path = os.path.join(args.data_path, merged_file_name)
    if not os.path.exists(merged_file_path):
        merged_file_path = merge_result(args.data_path, merged_file_name)

    # 解析结果
    result_data, failed_data = parse_result(merged_file_path)
    success_count = len(result_data)

    # 写入成功数据
    target_file = os.path.join(args.data_path, f"converted_{success_count}.json")
    write_json_file(target_file, result_data)

    # 写入失败数据
    for fail_type, data in failed_data.items():
        if data:  # 只有当有失败数据时才写入文件
            failed_file = os.path.join(
                args.data_path, f"failed_{fail_type}_{len(data)}.json"
            )
            write_json_file(failed_file, data)

    print(f"处理完成！成功解析{success_count}条数据，失败数据已分类保存。")
