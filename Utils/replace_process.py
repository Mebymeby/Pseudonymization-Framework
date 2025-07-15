from Config import prompt
from Utils import infer_util, replace_process_generative, \
    detect_replace_process_generative_tag_model as tag_model_script


def direct_replace(need_replace_text, candidate_entity_map):
    replace_text = need_replace_text
    for key, value in candidate_entity_map.items():
        replace_text = replace_text.replace(key, value)
    return replace_text


def prompt_replace(input_text, candidate_entity_map, tool_model, tool_model_tokenizer):
    replace_prompt = prompt.TASK["replace"].format(input_text, candidate_entity_map)
    output = infer_util.prompt_inference(tool_model, tool_model_tokenizer, replace_prompt,
                                         2 * len(input_text.split())).strip()
    return output


# [Replace] privacy entity in text 主函数
def replace_privacy_entity(args, dataset_config, item, entity_list_map, candidate_entity_map, dataset_entity_dict,
                           tool_model, tool_model_tokenizer, repeat_model, repeat_model_tokenizer,
                           tag_model, tag_model_tokenizer, ner_model, embedding_model):
    pseu_item = item.copy()

    for key in dataset_config.input_keys:
        if dataset_config.name == "cnn_dailymail":
            input_len = int(len(pseu_item[key].split()) * 0.3)
            value_split = pseu_item[key].split()[:input_len]
            pseu_item[key] = " ".join(value_split)

        if args.replace_method == "direct":
            pseu_item[key] = direct_replace(pseu_item[key], candidate_entity_map)
        elif args.replace_method == "prompt":
            pseu_item[key] = prompt_replace(
                pseu_item[key], candidate_entity_map, tool_model, tool_model_tokenizer)
        elif args.replace_method == "gen" or args.detect_method.startswith("tag"):
            if args.detect_method == "tag_mask" or args.detect_method == "tag_mark":
                pseu_item[key], candidate_entity_map = tag_model_script.detect_replace_generative(
                    args, pseu_item[key], dataset_entity_dict, entity_list_map, candidate_entity_map, tool_model,
                    tool_model_tokenizer, repeat_model, repeat_model_tokenizer, tag_model, tag_model_tokenizer,
                    ner_model, embedding_model)
            else:
                pseu_item[key] = replace_process_generative.replace_generative(
                    args, pseu_item[key], candidate_entity_map, repeat_model, repeat_model_tokenizer)

    return pseu_item
