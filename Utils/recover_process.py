from Config import prompt
from Utils import infer_util, replace_process_generative


def direct_recover_with_recover_entity_map(need_recover_text, recover_entity_map):
    replace_text = need_recover_text
    for key, value in recover_entity_map.items():
        replace_text = replace_text.replace(key, value)
    return replace_text


def prompt_recover_with_recover_entity_map(input_text, recover_entity_map, dataset_config, tool_model,
                                           tool_model_tokenizer):
    replace_prompt = prompt.TASK["replace"].format(input_text, recover_entity_map)
    output = infer_util.prompt_inference(tool_model, tool_model_tokenizer,
                                         replace_prompt, dataset_config.infer_max_new_tokens).strip()
    return output


def recover_privacy_entity(args, dataset_config, pseu_output, recover_entity_candidate_map,
                           tool_model, tool_model_tokenizer, repeat_model, repeat_model_tokenizer):
    recover_output = pseu_output

    if args.dataset == "glue-mnli":
        return recover_output

    # 置换映射方向
    recover_entity_candidate_map = {value: key for key, value in recover_entity_candidate_map.items()}

    if args.recover_method == "direct":
        recover_output = direct_recover_with_recover_entity_map(pseu_output, recover_entity_candidate_map)
    elif args.recover_method == "prompt":
        recover_output = prompt_recover_with_recover_entity_map(
            pseu_output, recover_entity_candidate_map, dataset_config, tool_model, tool_model_tokenizer)
    elif args.recover_method == "gen":
        recover_output = replace_process_generative.replace_generative(
            args, pseu_output, recover_entity_candidate_map, repeat_model, repeat_model_tokenizer)

    return recover_output
