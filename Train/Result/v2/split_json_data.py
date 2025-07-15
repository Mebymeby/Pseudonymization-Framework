#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON数据拆分脚本
将包含key、NED、ERM、ATG字段的JSON文件拆分为四个独立文件
"""

import json
import argparse
import os
from typing import List, Dict, Any


def split_json_data(input_file: str, output_dir: str = "split_output"):
    """
    拆分JSON文件中的数据
    
    Args:
        input_file: 输入的JSON文件路径
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 初始化四个列表
    keys = []
    ned_values = []
    erm_values = []
    atg_values = []
    
    # 处理每个数据项
    for item in data:
        # 提取key
        if 'key' in item:
            keys.append(item['key'])
        
        # 提取NED值（字典类型）
        if 'NED' in item:
            ned_json_str = json.dumps(item['NED'], ensure_ascii=False)
            ned_values.append(ned_json_str)
        
        # 提取ERM值（字典类型）
        if 'ERM' in item:
            erm_json_str = json.dumps(item['ERM'], ensure_ascii=False)
            erm_values.append(erm_json_str)
        
        # 提取ATG值（转换为以"document"为key的字典）
        if 'ATG' in item:
            atg_dict = {"document": item['ATG']}
            atg_json_str = json.dumps(atg_dict, ensure_ascii=False)
            atg_values.append(atg_json_str)
    
    # 保存key文件（列表格式）
    keys_file = os.path.join(output_dir, "keys.json")
    with open(keys_file, 'w', encoding='utf-8') as f:
        json.dump(keys, f, ensure_ascii=False, indent=2)
    
    # 保存NED文件（每行一个JSON字符串）
    ned_file = os.path.join(output_dir, "ned_values.txt")
    with open(ned_file, 'w', encoding='utf-8') as f:
        for ned_str in ned_values:
            f.write(ned_str + '\n')
    
    # 保存ERM文件（每行一个JSON字符串）
    erm_file = os.path.join(output_dir, "erm_values.txt")
    with open(erm_file, 'w', encoding='utf-8') as f:
        for erm_str in erm_values:
            f.write(erm_str + '\n')
    
    # 保存ATG文件（每行一个JSON字符串）
    atg_file = os.path.join(output_dir, "atg_values.txt")
    with open(atg_file, 'w', encoding='utf-8') as f:
        for atg_str in atg_values:
            f.write(atg_str + '\n')
    
    print(f"数据拆分完成！")
    print(f"输出目录: {output_dir}")
    print(f"生成的文件:")
    print(f"  - keys.json: {len(keys)} 个key")
    print(f"  - ned_values.txt: {len(ned_values)} 个NED值")
    print(f"  - erm_values.txt: {len(erm_values)} 个ERM值")
    print(f"  - atg_values.txt: {len(atg_values)} 个ATG值")


def main():
    parser = argparse.ArgumentParser(description='拆分JSON文件中的数据')
    parser.add_argument('input_file', help='输入的JSON文件路径')
    parser.add_argument('-o', '--output_dir', default='split_output', 
                       help='输出目录 (默认: split_output)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 '{args.input_file}' 不存在")
        return
    
    try:
        split_json_data(args.input_file, args.output_dir)
    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    main() 