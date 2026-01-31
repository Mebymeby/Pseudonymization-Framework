import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分布式运行端到端假名化脚本")

    # 数据参数
    parser.add_argument("--dataset", type=str, default="xsum", help="数据集名称")
    parser.add_argument("--data_split", type=str, default="train", help="数据集分割")
    parser.add_argument("--data_start", type=int, default=0, help="数据起始索引")
    parser.add_argument("--data_size", type=int, default=5000, help="数据总量")

    # 模型参数
    parser.add_argument("--end2end_model", type=str, default="Qwen3-4B", help="端到端模型名称")
    parser.add_argument("--decode_mode", type=str, default="multinomial_sampling", help="解码模式")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--model_gpu", type=int, default=-1, help="模型使用的GPU索引")

    # 分布式参数
    parser.add_argument("--num_gpus", type=int, default=8, help="使用的GPU数量")

    return parser.parse_args()


def calculate_data_split(total_size: int, num_splits: int, start_idx: int) -> List[Tuple[int, int]]:
    """
    计算数据分割方案

    Args:
        total_size: 数据总量
        num_splits: 分割数量
        start_idx: 起始索引

    Returns:
        每个分割的 (start, size) 元组列表
    """
    # 计算基本分割大小和余数
    base_size = total_size // num_splits
    remainder = total_size % num_splits

    splits = []
    current_start = start_idx

    for i in range(num_splits):
        # 为前 remainder 个分割分配额外的一个样本
        split_size = base_size + (1 if i < remainder else 0)
        splits.append((current_start, split_size))
        current_start += split_size

    return splits


def build_command(
        script_path: str,
        args_dict: dict,
        gpu_idx: int,
        data_start: int,
        data_size: int
) -> List[str]:
    """
    构建子进程命令

    Args:
        script_path: 脚本路径
        args_dict: 参数字典
        gpu_idx: GPU索引
        data_start: 数据起始索引
        data_size: 数据大小

    Returns:
        命令行参数列表
    """
    cmd = [sys.executable, script_path]

    # 复制参数字典并更新数据分割参数
    cmd_args = args_dict.copy()
    cmd_args["data_start"] = data_start
    cmd_args["data_size"] = data_size

    # 构建命令行参数
    for key, value in cmd_args.items():
        cmd.append(f"--{key}={value}")

    return cmd


def main() -> None:
    """主函数"""
    args = parse_args()

    # 创建结果目录
    result_dir = os.path.join(
        "Train", "Result",
        f"{args.end2end_model}_{args.dataset}_{args.data_split}_{args.decode_mode}_{args.seed}"
    )
    os.makedirs(result_dir, exist_ok=True)

    # 计算数据分割
    data_splits = calculate_data_split(args.data_size, args.num_gpus, args.data_start)

    # 准备参数字典（排除不需要传递的参数）
    args_dict = {
        "dataset": args.dataset,
        "data_split": args.data_split,
        "end2end_model": args.end2end_model,
        "decode_mode": args.decode_mode,
        "seed": args.seed,
        "model_gpu": args.model_gpu
    }

    # 启动子进程
    processes = []
    script_path = "Train/end2end_replace.py"

    print(f"启动 {args.num_gpus} 个进程处理数据...")

    for gpu_idx, (data_start, data_size) in enumerate(data_splits):
        # 构建命令
        cmd = build_command(script_path, args_dict, gpu_idx, data_start, data_size)

        # 准备环境变量
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        # 打印启动信息
        print(f"启动 GPU {gpu_idx}: PYTHONPATH=. CUDA_VISIBLE_DEVICES={gpu_idx} {' '.join(cmd[1:])}")

        # 启动子进程
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append((gpu_idx, process))

    # 等待所有进程完成
    print("\n等待所有进程完成...")

    for gpu_idx, process in processes:
        print(f"\n=== GPU {gpu_idx} 输出 ===")
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"GPU {gpu_idx} 成功完成！")
        else:
            print(f"GPU {gpu_idx} 失败，返回码: {process.returncode}")

        if stdout:
            print("STDOUT:")
            print(stdout)

        if stderr:
            print("STDERR:")
            print(stderr)

    print("\n所有 GPU 处理完成！")


if __name__ == "__main__":
    main()