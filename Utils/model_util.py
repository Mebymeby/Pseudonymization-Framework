import os

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 全局变量模拟单例模式
_eval_tokenizer, _eval_model = None, None
_tool_tokenizer, _tool_model = None, None
_ner_pipeline = None
_embedding_model = None
_repeat_model_tokenizer, _repeat_model = None, None
_tag_model_tokenizer, _tag_model = None, None


def load_causal_lm(model_path, model_device, device_map="auto", torch_dtype=torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map={"": model_device} if model_device >= 0 else device_map,
    )
    model.eval()
    return tokenizer, model


def load_eval_model(args):
    eval_model_path = os.path.join(args.model_base_path, args.eval_model)

    global _eval_tokenizer, _eval_model
    if _eval_tokenizer is None or _eval_model is None:
        _eval_tokenizer, _eval_model = load_causal_lm(eval_model_path, args.eval_model_gpu)

    return _eval_tokenizer, _eval_model


def load_tool_model(args):
    tool_model_path = os.path.join(args.model_base_path, args.tool_model)

    global _tool_tokenizer, _tool_model
    if _tool_tokenizer is None or _tool_model is None:
        _tool_tokenizer, _tool_model = load_causal_lm(tool_model_path, args.tool_model_gpu)

    return _tool_tokenizer, _tool_model


def load_repeat_model(args):
    repeat_model_path = os.path.join(args.model_base_path, args.repeat_model)

    global _repeat_model_tokenizer, _repeat_model
    if _repeat_model_tokenizer is None or _repeat_model is None:
        _repeat_model_tokenizer, _repeat_model = load_causal_lm(repeat_model_path, args.repeat_model_gpu)

    return _repeat_model_tokenizer, _repeat_model


def load_tag_model(args):
    ner_repeat_mode_path = os.path.join(args.model_base_path, args.tag_model)

    global _tag_model_tokenizer, _tag_model
    if _tag_model_tokenizer is None or _tag_model is None:
        _tag_model_tokenizer, _tag_model = load_causal_lm(ner_repeat_mode_path, args.tag_model_gpu)

    return _tag_model_tokenizer, _tag_model


def load_pipeline_model(model_path, model_device, pipeline_task="ner"):
    pipeline_model = pipeline(
        pipeline_task,
        model=model_path,
        tokenizer=model_path,
        device=model_device,
        # grouped_entities=True if pipeline_task == "ner" else None,
        aggregation_strategy="simple" if pipeline_task == "ner" else None,
    )
    return pipeline_model


def load_ner_model(args):
    ner_model_path = os.path.join(args.model_base_path, args.ner_model)

    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = load_pipeline_model(ner_model_path, args.ner_model_gpu)

    return _ner_pipeline


def load_embedding_model(args):
    embedding_model_path = os.path.join(args.model_base_path, args.embedding_model)

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(embedding_model_path)

    return _embedding_model
