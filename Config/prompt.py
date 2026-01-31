# 针对数据集的 prompt
DATASET = {
    "xsum": """<|im_start|>system
Generate a concise summary according to the document given by user.
The user input format is as follows:
document: 
The assistant output format is as follows:
summary: 
<|im_end|>

<|im_start|>user
document: {0}
<|im_end|>

<|im_start|>assistant
{1}summary: 
""",

    "glue-mnli": """<|im_start|>system
Determine the logical relationship according to the premise and hypothesis given by user.
The user input format is as follows:
premise: 
hypothesis: 
The assistant output format is as follows:
relationship: 

The possible relationships are:
"Entailment": The hypothesis logically follows from the premise.
"Neutral": The hypothesis is neither supported nor contradicted by the premise.
"Contradiction": The hypothesis contradicts the premise.

Here is an example:
premise: The new rights are nice enough
hypothesis: Everyone really likes the newest benefits
relationship: Neutral
<|im_end|>

<|im_start|>user
premise: {0}
hypothesis: {1}
<|im_end|>

<|im_start|>assistant
{2}relationship: 
""",

    "wmt14-de-en": """<|im_start|>system
Translate the English text given by the user into German without any other Explanation.
No other languages are allowed in the translation content.
<|im_end|>

<|im_start|>user
{0}
<|im_end|>

<|im_start|>assistant
{1}
""",

    "cnn_dailymail": """<|im_start|>system
Continue writing the following text input by the user.
The generated text needs to be semantically coherent with the input text.
<|im_end|>

<|im_start|>user
{0}
<|im_end|>

<|im_start|>assistant
{1}
"""
}

# 针对任务类型的 prompt
TASK = {
    "detect": """<|im_start|>system
Identify Named Entities in the text given by the user.

Attention Rules:
1. Identify ONLY these entity types in the text:
- PER (Person names)
- LOC (Locations/countries/cities)
- ORG (Organizations/companies)
2. Ignore all other entity types!
3. Strictly output in JSON format: {{"Entity1":"Type1", ...}}

Example:
Text: Tom met Jerry at Apple in NewYork.
Entities: {{"Tom":"PER", "Jerry":"PER", "Apple":"ORG", "NewYork":"LOC"}}
<|im_end|>

<|im_start|>user
Text: {0}
<|im_end|>

<|im_start|>assistant
Entities:
""",

    "generate_with_entity_and_type": """<|im_start|>system
Generate a different entity of the given type that MUST satisfy:
1. Different from the given entity.
2. Can replace the given entity semantically.
3. Null values cannot be output.
Output the generated entity without any extra explanation.

The user input format is as follows:
Input: entity: [The input entity] | entity_type: [The input entity type]

The assistant output format is as follows:
Output: 

Here is an example:
Input: entity: [Brennan Centre] | entity_type: [organization]
Output: Carter Institute

Output the generated entity without any extra explanation.
<|im_end|>

<|im_start|>user
Input: entity: [{0}] | entity_type: [{1}]
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "replace": """<|im_start|>system
Based on the entity mapping provided by the user where both the key and value are entities, replace the keys that appear in the user-given text with the corresponding values.
Ensuring the semantic integrity of the text is maintained after replacement.

The user input format is as follows:
Text:
Entity mapping: 
The assistant output format is as follows:
Replaced text: 

Here is an example:
Text: Zhang San works for Tencent in Shenzhen.
Entity mapping: {{"Zhang San": "Li Si", "Tencent": "HUAWEI", "Shenzhen": "Beijing"}}
Replaced text: Li Si works for HUAWEI in Beijing.
<|im_end|>

<|im_start|>user
Text: {0}
Entity mapping: {1}
<|im_end|>

<|im_start|>assistant
Replaced text: 
""",

    "ft_repeat": """<|im_start|>system
Directly output the input text, ensuring it is as identical to the original as possible.

The user input format is as follows:
Input: The input text.

The assistant output format is as follows: 
Output: The input text. (repeated)

Here is an example:
Input: This is an example sentence.
Output: This is an example sentence.

You only need to directly output the input text.
Try to make the input and output as consistent as possible.
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output:
""",

    "tag_mask_repeat": """"<|im_start|>system
Replace entity names such as persons, locations, and organizations in the given text with placeholders <ENTITY>.

Do not retain the replaced entity names.
Apart from the replaced sections, the rest of the given text should remain unchanged.

The input formats the given preceding content are as follows:
Input: 
The output format is as follows:
Output: 

Here is an example:
Input: Zhang San went to Shenzhen for a trip. As soon as he got off the plane, he was captivated by the city's modern and bustling atmosphere. 
Output: <ENTITY> went to <ENTITY> for a trip. As soon as he got off the plane, he was captivated by the city's modern and bustling atmosphere.
<|im_end|>

<|im_start|>user
Input:{0}
<|im_end|>

<|im_start|>assistant
Output:
""",

    "tag_mark_repeat": """<|im_start|>system
Mark the entity names in the given text. Specifically, use <Entity> and </Entity> to mark the entities directly in the original text.
Apart from the added marking sections, the rest of the given text must remain unchanged.

The input formats the given preceding content are as follows:
input: 
 
The output format is as follows:
output: 

Here is an example:
Input: Zhang San went to Shenzhen for a trip. As soon as he got off the plane, he was captivated by the city's modern and bustling atmosphere. 
Output: <Entity>Zhang San</Entity> went to <Entity>Shenzhen</Entity> for a trip. As soon as he got off the plane, he was captivated by the city's modern and bustling atmosphere.
<|im_end|>

<|im_start|>user
Input:{0}
<|im_end|>

<|im_start|>assistant
Output:
""",

    "llm_end2end_short":"""<|im_start|>system
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
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "llm_end2end_long": """<|im_start|>system
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
   - If no entities exist: Confirm the result as an empty dict (no entities to extract).
2. **REPLACE Entities**:
   - If entities exist: For each extracted entity, assign a plausible same-type substitute that meets the first 3 Core Constraints (uniqueness, type consistency, contextual appropriateness). Ensure no repetition of substitutes in the entire text.
   - If no entities exist: Skip replacement (no entities to replace), and the result is an empty dict.
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
Output:
<NED>{{"Tom":"PER", "Jerry":"PER", "Google":"ORG", "Paris":"LOC"}}</NED>
<ERM>{{"Tom":"Mike", "Jerry":"Lisa", "Google":"Microsoft", "Paris":"Tokyo"}}</ERM>
<ATG>Mike met Lisa at Microsoft’s office in Tokyo last week.</ATG>
Example 2: Input text with no entities
Input: "The team completed the project and submitted the report last Friday."
Output:
<NED>{{}}</NED>
<ERM>{{}}</ERM>
<ATG>The team completed the project and submitted the report last Friday.</ATG>

<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
｛1｝Output: 
""",
}
