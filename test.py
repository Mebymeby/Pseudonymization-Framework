import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = """<|im_start|>system
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
premise: The Kal tangled both of Adrin's arms, keeping the blades far away.
hypothesis: Adrin's arms were tangled, keeping his blades away from Kal.
<|im_end|>

<|im_start|>assistant

<think>
</think>

relationship: 
"""

model_path = "/longzi05/houshilong/Model/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

input_token = tokenizer(prompt, return_tensors="pt").to(model.device)
raw_output = model.generate(
    **input_token,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    num_beams=1,
)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(input_token.input_ids, raw_output)
]

decoded_output = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True
)[0]

print(decoded_output)