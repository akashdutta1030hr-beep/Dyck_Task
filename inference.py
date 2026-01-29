import torch
import json
from unsloth import FastLanguageModel
from tqdm import tqdm
import re

# ============================================================================
# Configuration
# ============================================================================
BASE_MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_PATH = "results"  # Path to fine-tuned model
# Use same cap as training so model context supports full sequences
MAX_LENGTH = 2048
DATASET_PATH = "conversation.jsonl"  # 10k samples (match generator default)
OUTPUT_PATH = "inference_results.jsonl"

# Bracket pairs (opening, closing) used in Dyck sequences - same as generator
BRACKET_PAIRS = [
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("⟨", "⟩"),
    ("⟦", "⟧"),
    ("⦃", "⦄"),
    ("⦅", "⦆"),
]

# ============================================================================
# Load Fine-tuned Model
# ============================================================================
print("Loading fine-tuned model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Set model to evaluation mode
FastLanguageModel.for_inference(model)

print("Model loaded successfully! ✓\n")

# ============================================================================

#prepare the input data
#Use the same format as the training dataset
sequence = "<[<⟨{(<[(("

# List bracket pairs so the model knows each opening bracket has one closing pair (e.g. ⟦/⟧)
bracket_list = ", ".join(f"{o}/{c}" for o, c in BRACKET_PAIRS)
prompt = (
    f"Bracket pairs (opening/closing): {bracket_list}.\n"
    f"Complete the following Dyck sequence: {sequence}"
)
messages = [
    {"role": "user", "content": prompt}
]

#Apply the chat template and ensure it adds the assistant generation prompt
user_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

#Tokenize the user text
user_tokenized = tokenizer(
    user_text,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
    padding=True,
    add_special_tokens=True
)

#Generate the response
input_ids = user_tokenized["input_ids"].to(model.device)
prompt_length = input_ids.shape[1]
output_ids = model.generate(
    input_ids,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    top_k=10,
    repetition_penalty=1.25,
    pad_token_id=tokenizer.pad_token_id,
)

# Decode only the newly generated tokens (exclude the prompt)
generated_ids = output_ids[0][prompt_length:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

#Print the response
print(response)

#Save the response to a file (only the model's response, not prompt + response)
with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
    f.write(json.dumps({"sequence": sequence, "response": response}, ensure_ascii=False) + '\n')


