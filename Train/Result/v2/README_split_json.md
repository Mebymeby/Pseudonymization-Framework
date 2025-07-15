# JSON数据拆分脚本使用说明

## 功能描述

这个脚本用于将包含特定结构的JSON文件拆分为四个独立文件。每个数据项包含以下字段：
- `key`: 字符串标识符
- `NED`: 字典类型（实体识别结果）
- `ERM`: 字典类型（实体替换映射）
- `ATG`: 字符串（长文本内容）

## 输出文件

脚本会生成四个文件：

1. **keys.json** - 包含所有key的列表
2. **ned_values.txt** - 每行一个NED字典的JSON字符串
3. **erm_values.txt** - 每行一个ERM字典的JSON字符串
4. **atg_values.txt** - 每行一个以"document"为key的字典的JSON字符串

## 使用方法

### 基本用法
```bash
python split_json_data.py input_file.json
```

### 指定输出目录
```bash
python split_json_data.py input_file.json -o output_directory
```

### 示例
```bash
# 使用默认输出目录 "split_output"
python split_json_data.py "Train/Result/v2/Qwen2.5-1.5B-Instruct-stf-merged-v2_xsum(0,5000).json"

# 指定自定义输出目录
python split_json_data.py "input.json" -o "my_split_output"
```

## 参数说明

- `input_file`: 输入的JSON文件路径（必需）
- `-o, --output_dir`: 输出目录（可选，默认为 "split_output"）

## 文件格式说明

### 输入文件格式
```json
[
  {
    "key": "38264402",
    "NED": {
      "Prison Link Cymru": "ORG",
      "Wales": "LOC",
      "Andrew Stevens": "PER"
    },
    "ERM": {
      "Prison Link Cymru": "Cumbria Police",
      "Wales": "Bathurst Road",
      "Andrew Stevens": "John Mccafferty"
    },
    "ATG": "Cumbria Police had 1,099 referrals in 2015-16..."
  }
]
```

### 输出文件格式

**keys.json:**
```json
[
  "38264402",
  "34227252",
  "38537698"
]
```

**ned_values.txt:**
```
{"Prison Link Cymru": "ORG", "Wales": "LOC", "Andrew Stevens": "PER"}
{"Waterfront Park": "LOC", "Colonsay View": "LOC"}
{"Jordan Hill": "PER", "Brittany Covington": "PER"}
```

**erm_values.txt:**
```
{"Prison Link Cymru": "Cumbria Police", "Wales": "Bathurst Road", "Andrew Stevens": "John Mccafferty"}
{"Waterfront Park": "Bathurst Street", "Colonsay View": "Cape St George"}
{"Jordan Hill": "Karlheinz Stockhausen", "Brittany Covington": "Bobby Jones"}
```

**atg_values.txt:**
```
{"document": "Cumbria Police had 1,099 referrals in 2015-16..."}
{"document": "Waterfront Park is a popular destination..."}
{"document": "Jordan Hill was arrested in Chicago..."}
```

## 注意事项

1. 脚本会自动创建输出目录（如果不存在）
2. 所有文件都使用UTF-8编码
3. JSON字符串使用`ensure_ascii=False`确保正确处理中文字符
4. 如果输入文件不存在，脚本会显示错误信息并退出
5. 处理过程中如果出现异常，会显示详细的错误信息

## 依赖要求

- Python 3.6+
- 标准库模块：`json`, `argparse`, `os`

## 错误处理

脚本包含完善的错误处理机制：
- 检查输入文件是否存在
- 验证JSON格式是否正确
- 处理文件读写异常
- 提供详细的错误信息 