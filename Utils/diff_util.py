import difflib


# 获取两个输入序列之间的映射字典
## 1. 输入序列为 字符串 / token 列表时，输出 字符串 / token 级映射
## 2. 输入序列为单一字符串时，输出字符串间字符级映射
def process_token_diff_map(tag_mask_tokens, ori_text_tokens):
    diff_map = {}

    # 比对
    d = difflib.Differ()
    diff = d.compare(tag_mask_tokens, ori_text_tokens)  # 占位符为 key，原 token 为 value
    diff_list = [x for x in diff]

    # 输出
    tag_mask_token, ori_token = "", ""
    for diff_item in diff_list:
        if diff_item.startswith("-"):  # tag_mask_tokens 中存在但 ori_text_tokens 中不存在的 diff_item
            tag_mask_token += str(diff_item.split()[-1])
        elif diff_item.startswith("+"):  # ori_text_tokens 中存在但 tag_mask_tokens 中不存在的 diff_item
            ori_token += str(diff_item.split()[-1])
        else:  # diff_item.startswith("?") 或 diff_item.startswith(" ") 时，tag_mask_token 和 ori_token 累加完成
            if ori_token and tag_mask_token:  # 判断
                key = tag_mask_token.replace("Ġ", " ").strip()
                val = ori_token.replace("Ġ", " ").strip()
                diff_map.update({key: val})
            ori_token, tag_mask_token = "", ""  # 重置累加

    # 遍历完成再进行一次判断
    if ori_token and tag_mask_token:  # 出现 + - 对，表明存在替换，记录映射关系
        key = tag_mask_token.replace("Ġ", " ").strip()
        val = ori_token.replace("Ġ", " ").strip()
        diff_map.update({key: val})

    return diff_map
