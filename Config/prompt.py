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
summary: 
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
relationship: 
""",

    "wmt14-de-en": """<|im_start|>system
Translate the English text given by the user into German without any other Explanation.
No other languages are allowed in the translation content.
<|im_end|>

<|im_start|>user
{0}
<|im_end|>

<|im_start|>assistant
""",

    "cnn_dailymail": """<|im_start|>system
Continue writing the following text input by the user.
The generated text needs to be semantically coherent with the input text.
<|im_end|>

<|im_start|>user
{0}
<|im_end|>

<|im_start|>assistant
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

    "llm_end2end": """<|im_start|>system
You are a privacy-focused text editor. Perform **named entity anonymization** by replacing all `PER` (Person), `LOC` (Location), and `ORG` (Organization) entities while preserving:
1. Original text semantics
2. Entity type consistency
3. Grammatical correctness

Execution Steps:
1. EXTRACT all `PER/LOC/ORG` entities from input text
2. REPLACE each entity with a plausible same-type substitute that is contextually appropriate
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
}
