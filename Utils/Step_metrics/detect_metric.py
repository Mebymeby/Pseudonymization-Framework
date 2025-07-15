import argparse
import json
import os.path


def load_detect_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()).keys() for line in file]


def calculate_metrics():
    detect_metric_scores = {detect_method: {
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'common': 0,
        'unique': 0,
        'jaccard': 0,
        'length': 0
    } for detect_method in args.detect_method}

    method_num = len(args.detect_method)

    # 遍历每句话的结果
    for sentence_idx in range(args.sentence_num):
        # 获取不同方法针对同一句话得到的结果列表
        one_sentence_results = [detect_cache_files[i][sentence_idx] for i in range(method_num)]
        # 获取同一句话的所有实体去重列表
        all_entities = set([entity for result in one_sentence_results for entity in result])

        # 计算共识实体集合
        majority = set()
        for entity in all_entities:
            # 计算当前实体在所有方法中，被识别出了几次
            count = sum([entity in result for result in one_sentence_results])
            if count >= args.consensus_threshold:
                majority.add(entity)

        # 基于共识实体集合，计算各方法的各项指标
        for cur_method_idx in range(method_num):
            cur_method_result = set(one_sentence_results[cur_method_idx])

            # 计算精准率
            precision = len(cur_method_result.intersection(majority)) / len(cur_method_result) if len(
                cur_method_result) > 0 else 0
            detect_metric_scores[args.detect_method[cur_method_idx]]['precision'] += precision

            # 计算召回率
            recall = len(cur_method_result.intersection(majority)) / len(majority) if len(majority) > 0 else 0
            detect_metric_scores[args.detect_method[cur_method_idx]]['recall'] += recall

            # 计算 F1 分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            detect_metric_scores[args.detect_method[cur_method_idx]]['f1'] += f1

            # 获取其他方法的并集
            other_method_result_union = set()
            for other_method_idx in range(method_num):
                if other_method_idx != cur_method_idx:
                    other_method_result_union.update(one_sentence_results[other_method_idx])

            # 计算普遍性指标
            common = len(cur_method_result.intersection(other_method_result_union)) / len(cur_method_result) if len(
                cur_method_result) > 0 else 0
            detect_metric_scores[args.detect_method[cur_method_idx]]['common'] += common

            # 计算独特性指标
            unique = len(cur_method_result - other_method_result_union) / len(cur_method_result) if len(
                cur_method_result) > 0 else 0
            detect_metric_scores[args.detect_method[cur_method_idx]]['unique'] += unique

            # 计算 Jaccard 相似度
            jaccard_sum = 0
            for other_method_idx in range(method_num):
                if other_method_idx != cur_method_idx:
                    other_result = set(one_sentence_results[other_method_idx])
                    jaccard = len(cur_method_result.intersection(other_result)) / len(
                        cur_method_result.union(other_result)) if len(cur_method_result.union(other_result)) > 0 else 0
                    jaccard_sum += jaccard
            detect_metric_scores[args.detect_method[cur_method_idx]]['jaccard'] += jaccard_sum / (method_num - 1)

            # 实体数量
            detect_metric_scores[args.detect_method[cur_method_idx]]['length'] += len(cur_method_result)

    # 平均每个句子的指标
    for cur_method_idx in range(method_num):
        for metric in detect_metric_scores[args.detect_method[cur_method_idx]]:
            current_metric = detect_metric_scores[args.detect_method[cur_method_idx]][metric]
            current_metric = current_metric / args.sentence_num
            if metric != 'length':
                current_metric *= 100
            current_metric = round(current_metric, 2)
            detect_metric_scores[args.detect_method[cur_method_idx]][metric] = '%.2f' % current_metric

    return detect_metric_scores


if __name__ == "__main__":

    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--detect_method", type=str, nargs="+", default=["ner", "prompt", "tag_mask", "tag_mark"])
    parser.add_argument("--sentence_num", type=int, default=5000)
    parser.add_argument("--consensus_threshold", type=int, default=2)
    args = parser.parse_args()

    # 读取文件
    detect_cache_file_dir = os.path.join("Output", "Cache", args.dataset, "Detect")
    detect_cache_file_paths = [os.path.join(detect_cache_file_dir, method + ".txt") for method in args.detect_method]

    detect_cache_files = [load_detect_cache_file(cache_file_path) for cache_file_path in detect_cache_file_paths]

    detect_metric_scores = calculate_metrics()

    # 输出指标值
    for method, metric in detect_metric_scores.items():
        print(f"{method:^{10}}: {metric}")
