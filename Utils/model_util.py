import os

import torch
from sentence_transformers import SentenceTransformer

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForQuestionAnswering, \
    AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# 全局变量模拟单例模式
_eval_tokenizer = None
_eval_model = None
_eval_pipe = None

_ner_tokenizer = None
_ner_model = None
_ner_pipe = None

_embedding_model = None

_tool_tokenizer = None
_tool_model = None

_repeat_model_tokenizer = None
_repeat_model = None

_ner_gen_model_tokenizer = None
_ner_gen_model = None


def load_eval_model(args, model_config, torch_dtype=torch.bfloat16, device_map="auto"):
    model_path = os.path.join(args.model_base_path, model_config.model_path)

    global _eval_tokenizer
    if _eval_tokenizer is None:
        _eval_tokenizer = AutoTokenizer.from_pretrained(model_path)

    global _eval_model
    if _eval_model is None:
        if model_config.load_type == "CausalLM":
            _eval_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map={"": args.eval_model_gpu} if args.eval_model_gpu > 0 else device_map,
            )
        if model_config.load_type == "QuestionAnswering":
            _eval_model = AutoModelForQuestionAnswering.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
        if model_config.load_type == "Seq2SeqLM":
            _eval_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map
            )

    _eval_model.eval()

    global _eval_pipe
    if _eval_pipe is None:
        if model_config.model_pipe == "True":
            _eval_pipe = pipeline(
                task=model_config.pipe_task,
                model=_eval_model,
                tokenizer=_eval_tokenizer,
                device=args.eval_model_gpu
            )
            return _eval_pipe, None

    return _eval_model, _eval_tokenizer


def load_ner_model(args, model_config, torch_dtype=torch.bfloat16, device_map="auto"):
    model_path = os.path.join(args.model_base_path, model_config.model_path)

    global _ner_tokenizer
    if _ner_tokenizer is None:
        _ner_tokenizer = AutoTokenizer.from_pretrained(model_path)

    global _ner_model
    if _ner_model is None:
        if model_config.load_type == "CausalLM":
            _ner_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map={"": args.ner_model_gpu}
            )
        if model_config.load_type == "TokenClassification":
            _ner_model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )

    _ner_model.eval()

    global _ner_pipe
    if _ner_pipe is None:
        if model_config.model_pipe == "True":
            if model_config.pipe_task == "ner":
                _ner_pipe = pipeline(
                    task=model_config.pipe_task,
                    model=_ner_model,
                    tokenizer=_ner_tokenizer,
                    aggregation_strategy="simple",  # 旧参数: grouped_entities=True,
                    device=args.ner_model_gpu
                )
            return _ner_pipe, None

    return _ner_model, _ner_tokenizer


def load_embedding_model(args):
    embedding_model_path = os.path.join(args.model_base_path, args.embedding_model)

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(embedding_model_path)
    return _embedding_model


def load_tool_model(args, model_config, torch_dtype=torch.bfloat16, device_map="auto"):
    tool_model_path = os.path.join(args.model_base_path, model_config.model_path)

    global _tool_tokenizer
    if _tool_tokenizer is None:
        _tool_tokenizer = AutoTokenizer.from_pretrained(tool_model_path)

    global _tool_model
    if _tool_model is None:
        _tool_model = AutoModelForCausalLM.from_pretrained(
            tool_model_path,
            torch_dtype=torch_dtype,
            device_map={"": args.tool_model_gpu}
        )

    _tool_model.eval()

    return _tool_model, _tool_tokenizer


def load_repeat_model(args, torch_dtype=torch.bfloat16, device_map="auto"):
    repeat_model_path = os.path.join(args.model_base_path, args.repeat_model)

    global _repeat_model_tokenizer
    if _repeat_model_tokenizer is None:
        _repeat_model_tokenizer = AutoTokenizer.from_pretrained(repeat_model_path)

    global _repeat_model
    if _repeat_model is None:
        _repeat_model = AutoModelForCausalLM.from_pretrained(
            repeat_model_path,
            torch_dtype=torch_dtype,
            device_map={"": args.repeat_model_gpu}
        )

    _repeat_model.eval()

    return _repeat_model, _repeat_model_tokenizer


def load_ner_gen_model(args, torch_dtype=torch.bfloat16, device_map="auto"):
    ner_repeat_mode_path = os.path.join(args.model_base_path, args.ner_gen_model)

    global _ner_gen_model_tokenizer
    if _ner_gen_model_tokenizer is None:
        _ner_gen_model_tokenizer = AutoTokenizer.from_pretrained(ner_repeat_mode_path)

    global _ner_gen_model
    if _ner_gen_model is None:
        _ner_gen_model = AutoModelForCausalLM.from_pretrained(
            ner_repeat_mode_path,
            torch_dtype=torch_dtype,
            device_map={"": args.repeat_model_gpu}
        )

    _ner_gen_model.eval()

    return _ner_gen_model, _ner_gen_model_tokenizer
