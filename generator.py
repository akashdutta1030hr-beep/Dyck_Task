import random
import json
from config import (
    N_SAMPLES, MIN_LEN, MAX_LEN, BRACKETS, PAIRS,
    OUTPUT_PATH, SEED
)

random.seed(SEED)

# ===============================
# Dyck sequence generation
# ===============================

def generate_partial_dyck_sequence(min_len: int, max_len: int) -> str:
    """
    Generate a valid partial Dyck sequence consisting only of opening brackets.
    """
    length = random.randint(min_len, max_len)
    return "".join(random.choice(BRACKETS) for _ in range(length))


def compute_closing_sequence(seq: str) -> str:
    """
    Compute the minimal required closing brackets.
    """
    return "".join(PAIRS[ch] for ch in reversed(seq))


# ===============================
# Reasoning (Stack-Summary)
# ===============================

def generate_reasoning(seq: str) -> str:
    unmatched = ", ".join(seq)
    closing = ", ".join(PAIRS[ch] for ch in reversed(seq))

    return (
        "Use a stack to track unmatched opening brackets.\n"
        f"Unmatched openings in order: {unmatched}.\n"
        f"Close them in reverse order: {closing}."
    )


# ===============================
# Prompt template (USER)
# ===============================

def build_prompt(seq: str) -> str:
    return (
        "Complete the following Dyck language sequence by adding the minimal "
        "necessary closing brackets.\n\n"
        f"Sequence: {seq}\n\n"
        "Rules:\n"
        "- Add only the required closing brackets\n"
        "- Do not add extra pairs\n\n"
        "Provide only the complete valid sequence."
    )


# ===============================
# Response template (ASSISTANT)
# ===============================

def build_response(seq: str) -> str:
    closing = compute_closing_sequence(seq)
    reasoning = generate_reasoning(seq)
    answer = seq + closing

    return (
        "<|Assistant|>\n"
        "<think>\n"
        f"{reasoning}\n"
        "</think>\n"
        f"{answer}\n"
        "<|EOT|>"
    )


# ===============================
# Dataset generation
# ===============================

def generate_dataset(n_samples: int, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            seq = generate_partial_dyck_sequence(MIN_LEN, MAX_LEN)

            sample = {
                "prompt": build_prompt(seq),
                "response": build_response(seq)
            }

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if (i + 1) % 10_000 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")

    print(f"\nDataset saved to: {output_path}")


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    generate_dataset(N_SAMPLES, OUTPUT_PATH)
