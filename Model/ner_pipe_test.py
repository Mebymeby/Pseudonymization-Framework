import os.path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from Utils.config_util import Config

ner_pipe = None
ner_model = None
ner_tokenizer = None


def load_ner_model(torch_dtype=torch.bfloat16, device_map="auto"):
    global ner_tokenizer
    ner_tokenizer = AutoTokenizer.from_pretrained(model_path)

    global ner_model
    if model_config.load_type == "TokenClassification":
        ner_model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            # device_map=device_map
            # bert-large-cased-finetuned-conll03-english 模型类 (BertForTokenClassification) 不支持 device_map='auto' 参数
        )

    global ner_pipe
    if model_config.model_pipe == "True":
        if model_config.pipe_task == "ner":
            ner_pipe = pipeline(
                task=model_config.pipe_task,
                model=ner_model,
                tokenizer=ner_tokenizer,
                aggregation_strategy="simple",
                # grouped_entities=True,
            )
        return ner_pipe, None

    return ner_model, ner_tokenizer


if __name__ == "__main__":
    model_name = "bert-large-cased-finetuned-conll03-english"
    model_config = Config(os.path.join("Config", "model_config", model_name))
    model_path = os.path.join("Model", model_name)

    ner_input = """Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50. Which NFL team represented the AFC at Super Bowl 50? """

    ner_model, ner_tokenizer = load_ner_model()
    if ner_tokenizer:
        print("NER Tokenizer loaded")
    else:
        ner_result = ner_model(ner_input)
        print(ner_result)
