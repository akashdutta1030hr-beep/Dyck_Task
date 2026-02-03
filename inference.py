"""
Dyck inference: load merged model (or base+adapter) and run completion.
Output style is dataset-only: # Thought N, # Step k: add 'X', FINAL ANSWER: ...
Not Qwen/DeepSeek-style prose (no "Wait...", "Let me recount", etc.).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model: use merged model (results_merged or HF repo with merged weights) for single-model load.
# Equivalently, base+adapter can be used via PEFT; merged = base+adapter baked into one model.
MODEL_ID = "akashdutta1030/dyck-deepseek-r1-lora"
MAX_LENGTH = 2048

# Generation (TEMPERATURE=0 + DO_SAMPLE=False = greedy, reproducible)
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
DO_SAMPLE = False

# Input sequence to complete
SEQUENCE = "<[<⟨{(<[(("


def format_prompt(sequence: str) -> str:
    return f"""Complete the following Dyck language sequence by adding the minimal necessary closing brackets.

Sequence: {sequence}

Rules:
- Add only the closing brackets needed to match all unmatched opening brackets
- Do not add any extra bracket pairs beyond what is required

Response format (dataset style only; do not use Qwen/DeepSeek-style prose):
- Use "# Thought N: ..." for each step (e.g. opening → push closing onto stack, stack state).
- Then "# So we add closing brackets one by one in this order:" and "# Step k: add 'X'." for each closing bracket.
- End with "FINAL ANSWER: " followed by the complete Dyck sequence only.
Do not add phrases like "Wait...", "Let me recount", or other conversational commentary."""


def main():
    print(f"Loading model: {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    prompt = format_prompt(SEQUENCE)
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    print("INPUT:")
    print(SEQUENCE)
    print()
    print("MODEL RESPONSE:")
    print(response)


if __name__ == "__main__":
    main()
