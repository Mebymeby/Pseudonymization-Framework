import argparse
import ast
import json
import os

import datasets
from tqdm import tqdm


def load_cache_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [ast.literal_eval(line.strip()) for line in file]


if __name__ == '__main__':
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--data_size", type=int, default=4)
    parser.add_argument("--detect_method", type=str, default="ner")
    parser.add_argument("--generate_method", type=str, default="rand")
    parser.add_argument("--replace_method", type=str, default="direct")
    args = parser.parse_args()

    # 加载数据集
    dataset_path = os.path.join("/longzi05/houshilong", "Dataset", args.dataset)
    dataset = datasets.load_from_disk(dataset_path)[args.data_split].select(range(0, args.data_size))

    # 读取中间结果文件
    cache_base_path = os.path.join("Output", "Cache", args.dataset)
    entity_list = load_cache_file(
        os.path.join(cache_base_path, "Detect", "Train", f"{args.detect_method}.txt")
    )
    entity_map = load_cache_file(
        os.path.join(cache_base_path, "Generate", "Train", f"{args.detect_method}-{args.generate_method}.txt")
    )
    pseu_text = load_cache_file(
        os.path.join(cache_base_path, "Replace", "Train",
                     f"{args.detect_method}-{args.generate_method}-{args.replace_method}.txt")
    )

    ins1 = """
You are a privacy-focused text editor. Perform **named entity anonymization** by replacing all `PER` (Person), `LOC` (Location), and `ORG` (Organization) entities while preserving:
1. Original text semantics
2. Entity type consistency
3. Grammatical correctness

Execution Steps:
1. EXTRACT all `PER/LOC/ORG` entities from input text
2. REPLACE each entity with a plausible same-type substitute that is contextually appropriate BUT not be repeated within a single sentence
3. GENERATE anonymized text using replacements

Output Steps:
1. Named Entity Extraction Dict. Strictly formatted and with Tag as: <NED>{{"Entity1":"Type1",...}}</NED>
2. Entity Replacement Map. Strictly formatted as: <ERM>{{"Entity1":"Replacement1",...}}</ERM>
3. Anonymized Text. Strictly formatted as: <ATG>Anonymized Text</ATG>
    """

    ins2 = """
You are a professional privacy-focused text editor specializing in high-precision named entity anonymization. Your output must strictly comply with data privacy standards, ensuring no re-identification risk while preserving the original text’s semantics, grammatical correctness, and entity type consistency.

### Core Constraints (Mandatory)
1. **Uniqueness of Replacements**: In the **entire input text (including multiple sentences)**, the replacement results of *different original entities* must be unique (even if they are of the same type). For example:
    - Wrong: "Tom met Jerry in Paris" → "John met John in London" (duplicate replacement for different PER entities)
    - Correct: "Tom met Jerry in Paris" → "Mike met Lisa in London" (all replacements are unique)
2. **Type Consistency**: Replacements must strictly match the original entity type (PER → real-name-style text, LOC → place-name-style text, ORG → organization-name-style text). No cross-type replacement (e.g., cannot replace "Paris" (LOC) with "Apple" (ORG)).
3. **Contextual Appropriateness**: Replacements must fit the text’s scenario/field.
4. **No-Entity Handling Rule**: If the input text contains no PER/LOC/ORG entities (after strict extraction checks), perform the following:
    - Entity extraction result: Empty dict (no entities listed).
    - Entity replacement result: Empty dict (no replacements needed).
    - Anonymized text: Output the original text without any modifications (no added/removed characters, no format changes).

### Execution Steps (Strictly Follow)
1. **EXTRACT Entities**:
    - First, strictly check whether the input text contains any PER (Person names, e.g., "Tom Smith"), LOC (Locations, e.g., "Paris"), or ORG (Organizations, e.g., "Google").
    - If entities exist, List all entities with correct types and No missing entities.
    - No mislabeling (e.g., do not mark a place "Paris" as PER, or a person "Tom" as LOC).
    - If no entities exist: Confirm the result as an empty set (no entities to extract).
2. **REPLACE Entities**:
    - If entities exist: For each extracted entity, assign a plausible same-type substitute that meets the first 3 Core Constraints (uniqueness, type consistency, contextual appropriateness). Ensure no repetition of substitutes in the entire text.
    - If no entities exist: Skip replacement (no entities to replace), and the result is an empty set.
3. **GENERATE Anonymized Text**: 
    - If entities exist: Integrate substitutes into the original text, ensuring no grammar errors, no semantic distortion, and no trace of original entities.
    - If no entities exist: Output the original text exactly as it is (no edits, no replacements, no format adjustments).

### Output Steps (Strict Format, No Extra Content)
1. Named Entity Extraction Dict (NED): List all extracted entities with correct types. Format: <NED>{{"Original Entity 1":"Type 1", "Original Entity 2":"Type 2", ...}}</NED>
2. Entity Replacement Map (ERM): Map each original entity to its unique substitute. Format: <ERM>{{"Original Entity 1":"Replacement 1", "Original Entity 2":"Replacement 2", ...}}</ERM>
3. Anonymized Text (ATG): The final anonymized text. Format: <ATG>Full Anonymized Text Here</ATG>

### Test Case Examples (For Clarification)
Example 1: Input text with entities
Input: "Tom met Jerry at Google’s office in Paris last week."
    - NED: <NED>{{"Tom":"PER", "Jerry":"PER", "Google":"ORG", "Paris":"LOC"}}</NED>
    - ERM: <ERM>{{"Tom":"Mike", "Jerry":"Lisa", "Google":"Microsoft", "Paris":"Tokyo"}}</ERM>
    - ATG: <ATG>Mike met Lisa at Microsoft’s office in Tokyo last week.</ATG>

Example 2: Input text with no entities
Input: "The team completed the project and submitted the report last Friday."
    - NED: <NED>{{}}</NED>
    - ERM: <ERM>{{}}</ERM>
    - ATG: <ATG>The team completed the project and submitted the report last Friday.</ATG>
    """

    # 组织数据
    alpaca = []
    for i in tqdm(range(args.data_size)):
        alpaca.append({
            "instruction": ins1,
            "input": dataset[i]["document"],
            "output": "<NED>" + json.dumps(entity_list[i]) + "</NED>\n<ERM>" + json.dumps(
                entity_map[i]) + "</ERM>\n<ATG>" + pseu_text[i]["document"] + "</ATG>",
        })

    # 保存数据
    save_path = os.path.join("Train", "Dataset", "SFT")
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(
        save_path,
        f"{args.dataset}_{args.data_split}_{args.data_size}_{args.detect_method}_{args.generate_method}_{args.replace_method}.json"
    )

    with open(save_file, 'a', encoding='utf-8') as file:
        json.dump(alpaca, file, indent=4, ensure_ascii=False)
