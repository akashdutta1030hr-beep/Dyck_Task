"""
Dyck inference: load Hugging Face model and run completion.
Matches inference.ipynb (Colab). Uses transformers + PEFT.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "akashdutta1030/dyck-deepseek-r1-lora"
BASE_MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_LENGTH = 2048
OUTPUT_PATH = "inference_results.jsonl"

# Inference controls (tune to match training / reduce rambling)
TEMPERATURE = 0.05       # Lower = more deterministic (try 0.01–0.1)
MAX_NEW_TOKENS = 256     # Cap length; training used reasoning + FINAL ANSWER
REPETITION_PENALTY = 1.1
TOP_P = 0.9
EXTRACT_ANSWER = True

BRACKET_PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ("⟨", "⟩"), ("⟦", "⟧"), ("⦃", "⦄"), ("⦅", "⦆")]
OPEN_TO_CLOSE = {o: c for o, c in BRACKET_PAIRS}
BRACKET_CHARS = set("()[]{}<>⟨⟩⟦⟧⦃⦄⦅⦆")


def format_prompt(sequence: str) -> str:
    """Same format as generator._format_question (and training)."""
    return f"""Complete the following Dyck language sequence by adding the minimal necessary closing brackets.

Sequence: {sequence}

Rules:
- Add only the closing brackets needed to match all unmatched opening brackets
- Do not add any extra bracket pairs beyond what is required

Provide only the complete valid sequence."""


def extract_answer(response: str) -> str:
    """
    Extract the Dyck completion from model output.
    Training format: reasoning + "FINAL ANSWER: " + full_sequence.
    """
    if not EXTRACT_ANSWER:
        return response.strip()
    # Prefer text after "FINAL ANSWER:" (same as training)
    if "FINAL ANSWER:" in response:
        part = response.split("FINAL ANSWER:")[-1].strip()
        # Take first line or up to next obvious end
        part = part.split("\n")[0].strip()
        if part:
            return part
    for line in reversed(response.split("\n")):
        line = line.strip()
        if line and all(c in BRACKET_CHARS or c.isspace() for c in line):
            return line.replace(" ", "")
    return response.strip()


def main():
    print("Loading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, MODEL_ID)
    model.eval()
    print("Model loaded.\n")

    sequence = "<[<⟨{(<[(("
    prompt = format_prompt(sequence)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
        )

    raw_response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()
    answer = extract_answer(raw_response)


    print("=" * 50)
    print("INPUT SEQUENCE:")
    print("=" * 50)
    print(sequence)
    print()
    print("=" * 50)
    print("MODEL OUTPUT:")
    print("=" * 50)
    print(raw_response[:500] + ("..." if len(raw_response) > 500 else ""))
    print()
    print("=" * 50)
    print("FINAL ANSWER (Dyck completion):")
    print("=" * 50)
    print(answer)
    print()
    print("=" * 50)
    print("Result (JSON):")
    print("=" * 50)
    result = {"sequence": sequence, "response": raw_response, "answer": answer}
    print(json.dumps({"sequence": sequence, "response": raw_response, "answer": answer}, ensure_ascii=False, indent=2))

    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\nAppended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
