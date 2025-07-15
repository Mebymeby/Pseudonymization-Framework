import os

# 使用HF镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import datasets


def parse_path(dataset_name, arg):
    if os.name == 'nt':
        return os.path.join("Dataset", dataset_name + (("-" + arg) if arg else ""))
    return os.path.join("/data4/houshilong/Dataset", dataset_name + ("-" + arg) if arg else "")


def download_dataset(dataset_name, arg=""):
    if not arg:
        dataset = datasets.load_dataset(dataset_name)
    else:
        dataset = datasets.load_dataset(dataset_name, arg)
    save_path = parse_path(dataset_name, arg)
    dataset.save_to_disk(save_path)
    print(f"Dataset {dataset_name} has been saved to {save_path}\n")


# 下载数据集，并保存到本地（可以连接到HF的环境或使用镜像站）
if __name__ == "__main__":
    # 问答
    # download_dataset("squad")
    # download_dataset("squad_v2")
    # download_dataset("hotpot_qa", "distractor")  # hotpot_qa 包含两个配置：distractor 和 fullwiki

    # 摘要
    # download_dataset("xsum")
    # download_dataset("samsum")
    # download_dataset("cnn_dailymail")  # 默认最新版本 3.0.0

    # 分类
    # download_dataset("glue", "mnli")  # mnli 数据集 是 glue 基准中的一个子集

    # 翻译
    # download_dataset("wmt14", "de-en")

    # 续写 Text Continuation
    # download_dataset("EleutherAI/the_pile")
    # download_dataset("wikitext", "wikitext-103-v1")
    download_dataset("lambada")
