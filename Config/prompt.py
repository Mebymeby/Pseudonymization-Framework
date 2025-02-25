# 针对数据集的prompt
DATASET = {
    "squad": """Generate a concise answer according to the given context and question, and output the answer (without any extra explanation) in JSON format: {"answer":"your answer content"}""",
    "squad_v2": """Generate a concise answer according to the given context and question, and output the answer (without any extra explanation) in JSON format: {"answer":"your answer content"}. If the answer does not exist, fill in the JSON value with "".""",
    "hotpot_qa-distractor": """待适配""",

    "xsum": """Generate a concise summary according to the given document, and output the summary (without any extra explanation) in JSON format: {"summary":"your summary content"}""",
    "cnn_dailymail": """Generate a concise summary according to the given article, and output the summary (without any extra explanation) in JSON format: {"summary":"your summary content"}""",
    "samsum": """Generate a concise summary according to the given dialogue, and output the summary (without any extra explanation) in JSON format: {"summary":"your summary content"}""",

    "glue-mnli": """Determine the logical relationship according to the given premise and hypothesis, and output your answer (without any extra explanation) in JSON format: {"relationship":"your answer"}
The possible relationships are:
1. "Entailment": The hypothesis logically follows from the premise.
2. "Neutral": The hypothesis is neither supported nor contradicted by the premise.
3. "Contradiction": The hypothesis contradicts the premise.
""",

    "wikitext-103": """<|im_start|>system
Continue writing the following text input by the user.
The generated text needs to be consistent and natural with the input text.
<|im_end|>
<|im_start|>user
{0}
<|im_end|>
<|im_start|>assistant
""",

    "wmt14-de-en": """<|im_start|>system
Translate the English text given by the user into German.
The user input format is as follows:
Input: 
The assistant output format is as follows:
Output: 

Here is an example:
Input: Obama is in Paris on business.
Output: Obama ist auf einer geschäftsreise in paris.

<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
"""
}

# 针对任务类型的prompt
TASK = {
    "llm_end2end": """<|im_start|>system
You are a text editor.
To protect privacy and prevent sensitive information from being leaked, you need to identify named entities in the given input text, and replace them with other named entities with same type.
The named entity type to replace is: {1}
Critical: Does not change the basic semantics of the text.

The user input format is as follows:
Input: 

The assistant output format is as follows:
Output: 

You need to perform the following steps in sequence and output the result of the last step.
1. Extract all named entities from the given text.
2. Replace the extracted named entities with other different named entities, and generate a replacement mapping.
3. Replace the extracted named entities in the given text based on the replacement mapping without changing the basic semantics of the text, output the final text without any extra explanation.
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "llm_end2end_detect": """<|im_start|>system
Detects all named entities in the input text given by user, and saves all detected entities as a list format: ["entity1", "entity2", ...]
The named entities type to Detect is: {1}

The user input format is as follows:
Input: 

The assistant output format is as follows:
Output: 

Here is an example:
Input: Zhang San went to Shenzhen for a trip.
Output: ["Zhang San", "Shenzhen"]
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "llm_end2end_replace": """<|im_start|>system
According to the entity mapping given by the user (where both key and value are named entities), the key in the text given by the user is replaced with the corresponding value.
Keep the semantic integrity and smoothness of the processed sentences.
The user input format is as follows:
Input text:
Input entity_map: 

The assistant output format is as follows:
Output: 

Here is an example:
Input text: Zhang San went to Shenzhen for a trip.
Input entity_map: {{"Zhang San": "Li Si","Shenzhen": "Beijing"}}
Output: Li Si went to Beijing for a trip.
<|im_end|>

<|im_start|>user
Input text: {0}
Input entity_map: {1}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "generate_with_entity_and_type_JSON": """<|im_start|>system
Generate an different entity of the given entity type that is semantically similar to the given entity.

The input format is as follows:
Input: entity: The input entity | entity_type: The input entity type

The output format is as follows:
Output: {"entity":"The generated entity"}

Here is an example:
Input: entity: Brennan Centre | entity_type: organization
Output: {"entity":"Carter Institute"}

Output the generated entity in JSON format and without any extra explanation.
<|im_end|>

<|im_start|>user
Input: entity: {0} | entity_type: {1}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "generate_with_entity_and_type": """<|im_start|>system
Generates an different entity of the entity type given by the user that is semantically similar to the entity given by the user.

The user input format is as follows:
Input: entity: The input entity | entity_type: The input entity type

The assistant output format is as follows:
Output: 

Here is an example:
Input: entity: Brennan Centre | entity_type: organization
Output: Carter Institute

Output the generated entity without any extra explanation.
<|im_end|>

<|im_start|>user
Input: entity: {0} | entity_type: {1}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "generate_with_entity": """<|im_start|>system
Generate an different entity that is semantically similar to the entity given by user.

The user input format is as follows:
Input: 

The assistant output format is as follows:
Output: 

Here is an example:
Input: Brennan Centre
Output: Carter Institute

Output the generated entity without any extra explanation.
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "generate_with_type": """<|im_start|>system
Generate an different entity of the given entity type.

The input format is as follows:
Input: The input entity type

The output format is as follows:
Output: 

Here is an example:
Input: organization
Output: Carter Institute

Output the generated entity without any extra explanation.
<|im_end|>

<|im_start|>user
Input: {0}
<|im_end|>

<|im_start|>assistant
Output: 
""",

    "repeat_prompt": """<|im_start|>system
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

    "only_rep_prompt": """"<|im_start|>system
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

    "tag_wrap_prompt": """<|im_start|>system
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
"""
}
