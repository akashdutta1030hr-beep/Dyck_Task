"""
Check the dataset's sequence length distribution (in tokens).
Uses the same tokenizer and text-building logic as Train.py so the stats match training.
Run before training to confirm MAX_LENGTH in Train.py is sufficient.
"""
import json
import sys
from pathlib import Path

# Same as Train.py
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "conversation.jsonl"
MAX_SAMPLES = 5000   # Cap for speed; set to None to use full dataset


def build_full_text(conversation, tokenizer):
    """Build full_text exactly like Train.py preprocess (no tokenization yet)."""
    user_msg = None
    assistant_msg = None
    for msg in conversation:
        role = msg.get("role", "")
        if role == "user":
            user_msg = msg
        elif role == "assistant":
            assistant_msg = msg
    if not user_msg or not assistant_msg:
        return None
    user_content = user_msg.get("content", "")
    assistant_reasoning = assistant_msg.get("reasoning_content", "").strip()
    assistant_completion = assistant_msg.get("content", "").strip()
    if assistant_reasoning and assistant_completion:
        assistant_content = f"{assistant_reasoning}\n\nFINAL ANSWER: {assistant_completion}"
    else:
        assistant_content = assistant_completion or assistant_reasoning
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return full_text


def main():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    print(f"Loading tokenizer: {MODEL_NAME} ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = []
    n_skipped = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if MAX_SAMPLES and i >= MAX_SAMPLES:
                break
            line = line.strip()
            if not line:
                continue
            try:
                conversation = json.loads(line)
                full_text = build_full_text(conversation, tokenizer)
                if full_text is None:
                    n_skipped += 1
                    continue
                enc = tokenizer(
                    full_text,
                    truncation=False,
                    add_special_tokens=True,
                )
                length = len(enc["input_ids"])
                lengths.append(length)
            except Exception as e:
                n_skipped += 1
                continue

    if not lengths:
        print("No valid sequences found.")
        sys.exit(1)

    lengths.sort()
    n = len(lengths)
    max_len = max(lengths)
    min_len = min(lengths)
    mean_len = sum(lengths) / n
    p50 = lengths[n // 2]
    p95 = lengths[int(n * 0.95)] if n >= 20 else lengths[-1]
    p99 = lengths[int(n * 0.99)] if n >= 100 else lengths[-1]

    print()
    print("=" * 60)
    print("DATASET SEQUENCE LENGTH (tokens, same as Train.py input)")
    print("=" * 60)
    print(f"  Samples checked: {n}")
    if n_skipped:
        print(f"  Skipped:         {n_skipped}")
    print(f"  Min:             {min_len}")
    print(f"  Max:             {max_len}")
    print(f"  Mean:            {mean_len:.1f}")
    print(f"  Median (p50):    {p50}")
    print(f"  p95:             {p95}")
    print(f"  p99:             {p99}")
    print("=" * 60)

    max_in_train = 768  # Train.py MAX_LENGTH
    if max_len > max_in_train:
        print(f"\n  WARNING: Dataset max ({max_len}) > Train.py MAX_LENGTH ({max_in_train}).")
        print(f"  Consider increasing MAX_LENGTH in Train.py to at least {max_len} (e.g. {max_len + 64}).")
    else:
        print(f"\n  Train.py MAX_LENGTH ({max_in_train}) >= dataset max ({max_len}). OK.")
    print()


if __name__ == "__main__":
    main()
