import os.path

import evaluate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def init_score(dataset_config):
    def create_score_dict():
        return {key: [] for key in dataset_config.score_keys}

    return {
        "full_ori_scores": create_score_dict(),  # 全数据集 : 原始分数
        "full_pseu_scores": create_score_dict(),  # 全数据集 : 假名化分数
        "pseu_ori_scores": create_score_dict(),  # 假名化数据集 : 原始分数
        "pseu_pseu_scores": create_score_dict(),  # 假名化数据集 : 假名化分数
    }


def load_metric(args, dataset_config):
    # 使用 evaluate 库（问答、分类）
    if args.dataset in ["squad", "squad_v2", "glue-mnli"]:
        return evaluate.load(str(os.path.join(args.metric_base_path, dataset_config.metric_path)))
    # 使用 rouge_scorer 库（摘要）
    elif args.dataset in ["xsum", "samsum"]:
        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    else:
        return None


def calculate_bleu(reference_text, inference_text):
    reference_list = [reference_text]
    inference_list = inference_text.split()

    bleu_weight_map = {
        "bleu_1": [1, 0, 0, 0],
        "bleu_2": [0.5, 0.5, 0, 0],
        "bleu_3": [0.33, 0.33, 0.33, 0],
        "bleu_4": [0.25, 0.25, 0.25, 0.25],
    }

    smoother = SmoothingFunction().method1

    result = [sentence_bleu(reference_list, inference_list, weights=value, smoothing_function=smoother)
              for _, value in bleu_weight_map.items()]

    return result


# 针对不同数据集，适配不同compute逻辑
def compute_score(args, item, outputs, dataset_config):
    # 获取评估指标
    metric = load_metric(args, dataset_config)

    if args.dataset == "squad":
        return metric.compute(
            predictions=[{"id": item["id"], "prediction_text": outputs}],
            references=[{"id": item["id"], "answers": item["answers"]}],
        )

    if args.dataset == "squad_v2":
        return metric.compute(
            predictions=[
                {"id": item["id"], "prediction_text": outputs, "no_answer_probability": 1.0 if not outputs else 0.0}],
            references=[{"id": item["id"], "answers": item["answers"]}],
        )

    if args.dataset in ["xsum", "samsum"]:
        score_dict = metric.score(item["summary"], outputs)
        return {key: score_dict[key].fmeasure for key in score_dict.keys()}

    if args.dataset == "cnn_dailymail":
        input_len = int(len(item["article"].split()) * 0.3)
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(item["article"].split()[input_len:], outputs)
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }

    if args.dataset == "glue-mnli":
        mnli_label_map = {
            "Entailment": 0,
            "Neutral": 1,
            "Contradiction": 2,
        }

        output_label = -1
        for key, value in mnli_label_map.items():
            if key in outputs:
                output_label = value
                break
        if output_label != -1:
            return metric.compute(references=[output_label], predictions=[item["label"]])
        else:
            return {"accuracy": 0.0}

    if args.dataset == "wikitext-103":
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(item["text"].split()[50:], outputs)
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }

    if args.dataset == "wmt14-de-en":
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(item["de"].split(), outputs)
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }


def compute_avg_score(scores_dict):
    avg_scores = []

    def np_mean(one_score_dict, key):
        if not one_score_dict[key]:
            return 0.0
        return np.mean(one_score_dict[key])

    def score_append(one_score_dict):
        avg_scores.append({key: np_mean(one_score_dict, key) for key in one_score_dict.keys()})

    def last_two_sub_score_append():
        index = len(avg_scores)
        avg_scores.append(
            {key: avg_scores[index - 1][key] - avg_scores[index - 2][key] for key in avg_scores[index - 1]})

    count = 0
    for _, value in scores_dict.items():
        if isinstance(value, dict) and value:
            score_append(value)
            count += 1
            if count & 1 == 0:
                last_two_sub_score_append()

    return avg_scores
