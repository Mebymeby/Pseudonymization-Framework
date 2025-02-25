import os.path

import evaluate
import numpy as np
import torch
from sentence_transformers import util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def init_score(dataset_config):
    def create_score_dict():
        score_dict = {key: [] for key in dataset_config.score_keys}
        score_dict.update({"san_effect": []})
        return score_dict

    return {
        "ori_scores": create_score_dict(),  # 全数据集:原始得分
        "san_scores": create_score_dict(),  # 全数据集:数据净化后得分
        "sanitized_ori_scores": create_score_dict(),  # 被净化数据子集:原始得分
        "sanitized_san_scores": create_score_dict(),  # 被净化数据子集:数据净化后得分
        "sanitized_count": 0,  # 数据净化样本数量
        "transparent_sanitized_count": 0  # 透明净化样本数量
    }


def load_metric(args, dataset_config):
    # 使用 evaluate 库（问答、分类）
    if dataset_config.name in ["squad", "squad_v2", "glue-mnli"]:
        return evaluate.load(str(os.path.join(args.metric_base_path, dataset_config.metric_path)))
    # 使用 rouge_scorer 库（摘要）
    elif dataset_config.name in ["xsum", "cnn_dailymail", "samsum"]:
        return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    else:
        return None


def calculate_ppl(input_text, eval_model, eval_tokenizer):
    inputs = eval_tokenizer(input_text, return_tensors="pt").to(eval_model.device)
    outputs = eval_model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


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
def compute_score(item, outputs, metric, dataset_config, eval_model, eval_tokenizer):
    if dataset_config.name == "squad":
        return metric.compute(
            predictions=[{"id": item["id"], "prediction_text": outputs}],
            references=[{"id": item["id"], "answers": item["answers"]}],
        )

    if dataset_config.name == "squad_v2":
        return metric.compute(
            predictions=[
                {"id": item["id"], "prediction_text": outputs, "no_answer_probability": 1.0 if not outputs else 0.0}],
            references=[{"id": item["id"], "answers": item["answers"]}],
        )

    if dataset_config.name in ["xsum", "samsum"]:
        score_dict = metric.score(item["summary"], outputs)
        return {key: score_dict[key].fmeasure for key in score_dict.keys()}

    if dataset_config.name == "cnn_dailymail":
        score_dict = metric.score(item["highlights"], outputs)
        return {key: score_dict[key].fmeasure for key in score_dict.keys()}

    if dataset_config.name == "glue-mnli":
        mnli_label_map = {
            "Entailment": 0,
            "Neutral": 1,
            "Contradiction": 2,
        }
        if outputs in mnli_label_map:
            label = mnli_label_map[outputs]
            return metric.compute(references=[label], predictions=[item["label"]])
        else:
            return {"accuracy": 0.0}

    if dataset_config.name == "wikitext-103":
        perplexity = calculate_ppl(" ".join(item["text"].split()[:50]), eval_model, eval_tokenizer)
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(item["text"].split()[50:], outputs)
        return {
            "perplexity": perplexity,
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }

    if dataset_config.name == "wmt14-de-en":
        bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu(item["de"].split(), outputs)
        return {
            "bleu_1": bleu_1,
            "bleu_2": bleu_2,
            "bleu_3": bleu_3,
            "bleu_4": bleu_4,
        }


# 计算两个文本的"不相似度"
def caculate_san_effect(seq1, seq2, embedding_model):
    eb1 = embedding_model.encode(seq1, convert_to_tensor=True)
    eb2 = embedding_model.encode(seq2, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(eb1, eb2).squeeze(0).item()
    return {"san_effect": 1.0 - sim}


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
