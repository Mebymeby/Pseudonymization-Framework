# 面向线上开放LLM的数据净化隐私保护方案研究

## 快速开始

1. 克隆仓库
2. 下载所需模型文件置于 `{pre_path}/Model/`
3. 通过 `Dataset/download_datasets.py` 脚本下载数据集或自行准备数据集至 `{pre_path}/Dataset/`
4. 下载 `evaluate` 库的所需评估函数源码置于 `{pre_path}/Metrics/`
5. 安装 python 依赖库
6. 终端运行推理脚本
    1. 推理脚本可参考 `Utils/argparse_util.py` ,根据参数描述按需生成
    2. 注：所有脚本工作目录默认为 `Paper/`

## 介绍

### 项目结构

```text
Paper

|--Config                               配置文件
    |--dataset_config                       数据集配置文件
    |--model_config                         模型配置文件
    |--prompt.py                            prompt常量文件
    |--config.py                            旧脚本配置文件

|--Dataset                              数据集文件
    |--squad                                保存的数据集文件
    |--... ...
    |--download_datasets.py                 数据集下载脚本
    |--show_dataset.py                      数据集查看脚本

|--Metrics                              evaluate库的评价指标源码
    |--accuracy                             具体的评价指标源码
    |--... ...

|--Model                                模型文件
    |--bart-large-cnn
    |--Qwen2.5-1.5B-Instruct
    |--... ...
    |--CausalLM_prompt_and_output.py        通用LM基于不同任务的prompt与输出验证

|--Output                               实验输出文件
    |--Entity_dict                          各数据集抽取的实体字典
        |--squad.json
        |--... ...
    |--Log                              日志文件
        |--squad                            具体数据集的日志文件
            |--result                           结果日志
            |--badcase                          badcase日志
        |--... ...
    |--Log_old                          旧脚本日志文件
        |--... ...

|--Utils                                工具脚本
    |--demo                                 demo测试脚本
        |--... ...
    |--entity_dict_util                     实体字典抽取脚本
        |--... ...
    |--argparse_util.py                     参数解析器
    |--config_util.py                       配置相关
    |--dataset_util.py                      数据、评价指标相关
    |--infer_util.py                        模型推理相关
    |--logging_util.py                      日志相关
    |--model_util.py                        模型加载相关
    |--saniti_func.py                       data sanitization 相关方法
    |--score_util.py                        评估分数计算与汇总
    
|--squad.py                             具体数据集的旧脚本
|--cnn-dm.py                            具体数据集的旧脚本
|--xsum.py                              具体数据集的旧脚本

|--main.py                              主脚本
```

### Config/ 配置文件结构

#### dataset_config/ 数据集配置文件结构

```json
{
  "name": "数据集名称",
  "path": "数据集在 Dataset/ 下的路径",
  "eval_split": "指定作为评估数据的数据集split",
  "task": "数据集对应的NLP任务名（使用时注意统一名称）",
  "metric_path": "数据集对应任务的评价指标在 Metric/ 下的路径",
  "input_keys": "数据集需要输入到评估模型中的内容的keys",
  "entity_dict_path": "数据集抽取的实体字典的存放路径",
  "LLM_max_new_tokens": "LLM推理当前数据集时，额外输出的最大长度",
  "badcase_xxx": "badcase相关配置，待优化"
}
```

#### model_config/ 模型配置文件结构

```json
{
  "model_name": "模型名称",
  "model_path": "模型在 Model/ 下的路径",
  "load_type": "模型加载时的类型（用于控制加载方式）",
  "model_pipe": "是否使用pipeline简化当前模型的推理",
  "pipe_task": "使用pipeline时的任务描述字符串"
}
```

#### prompt.py 模型提示语常量文件

```python
dict_name = {
    # 任务名 : 具体的 prompt 语句
    "key": "value"
}
```

#### config.json 旧脚本配置文件

## 如何新增验证配置

### 新增数据集

1. 准备数据集 【Dataset/{数据集文件}】
2. 新增数据集配置文件 【Config/dataset/{dataset_name}.json】
3. 适配数据集prompt 【Config/prompt.py】
4. 适配评估函数 【Utils/score_util.py -> init_score() load_metric() compute_score()】
5. 抽取实体字典（可选）【Utils/entity_dict_util/ner_dict_pipe.py】

### 新增验证模型

1. 准备模型
2. 新增模型配置文件
3. 适配模型输入输出格式