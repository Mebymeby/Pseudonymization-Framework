import json

from Config import prompt
from Utils import infer_util


def detect_by_ner_model(args, item_input_content, ner_model):
    entity_list_map = {}

    ner_result = ner_model(item_input_content)
    for entity_item in ner_result:
        if entity_item["entity_group"] in args.entity_types and entity_item["score"] >= args.entity_score_threshold:
            entity_list_map.update({entity_item["word"]: entity_item["entity_group"]})

    return entity_list_map


def detect_by_tool_model_prompt(args, item_input_content, tool_model, tool_model_tokenizer):
    entity_list_map = {}
    input_len = len(item_input_content.split())
    if input_len == 0:
        input_len = 1

    detect_prompt = prompt.TASK["detect"].format(item_input_content)
    output = infer_util.prompt_inference(
        tool_model, tool_model_tokenizer, detect_prompt, 10 * input_len)

    try:
        entity_list_map = json.loads(output)  # 实体列表字典的字符串形式

        # 只留下需要识别的实体类型
        entity_list_map = {key: value for key, value in entity_list_map.items() if value in args.entity_types}

    except json.decoder.JSONDecodeError:
        print(f"[ERROR]: Can't decode entity list map, prompt inference output: {output}")

    return entity_list_map


# [Detect] entities 主函数
def detect_entity_list(args, dataset_config, item_input_content, ner_model, tool_model, tool_model_tokenizer):
    entity_list_map = {}  # 格式: {"entity1":"type1", "entity2":"type2", ..., "entityN":"typeN"}

    if args.detect_method == "ner":
        entity_list_map = detect_by_ner_model(args, item_input_content, ner_model)
    elif args.detect_method == "prompt":
        entity_list_map = detect_by_tool_model_prompt(
            args, item_input_content, tool_model, tool_model_tokenizer)
    elif args.detect_method == "tag_mask" or args.detect_method == "tag_mark":
        entity_list_map = {}  # 这里先返回，后续过程调用 detect_replace_process_generative_tag_model 实现多阶段

    return entity_list_map
