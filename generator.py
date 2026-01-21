import random
import json

# -----------------------------
# Bracket configuration
# -----------------------------
OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">"
}
OPENING = list(OPEN_TO_CLOSE.keys())

# -----------------------------
# Generate a valid Dyck sequence
# -----------------------------
def generate_dyck_sequence(max_depth=5, max_length=40):
    stack = []
    sequence = []

    while len(sequence) < max_length:
        # Decide to open or close
        if not stack or (len(stack) < max_depth and random.random() < 0.6):
            op = random.choice(OPENING)
            stack.append(op)
            sequence.append(op)
        else:
            sequence.append(OPEN_TO_CLOSE[stack.pop()])

    # Close remaining
    while stack:
        sequence.append(OPEN_TO_CLOSE[stack.pop()])

    return "".join(sequence)

# -----------------------------
# Create partial input
# -----------------------------
def make_partial(sequence, min_cut=1):
    cut = random.randint(min_cut, len(sequence) - 1)
    return sequence[:cut]

# -----------------------------
# Stack summary after scan
# -----------------------------
def compute_stack(sequence):
    stack = []
    for ch in sequence:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
        else:
            stack.pop()
    return stack

# -----------------------------
# Reasoning template (TEMPLATE 1)
# -----------------------------
def make_reasoning(stack):
    if stack:
        stack_str = ", ".join(stack)
    else:
        stack_str = "(empty)"
    return (
        "<think>\n"
        "Scan the sequence from left to right.\n"
        "Push unmatched opening brackets onto a stack.\n"
        f"Remaining stack (bottom to top): {stack_str}\n"
        "Close brackets in reverse order using matching pairs.\n"
        "</think>\n"
    )

# -----------------------------
# Build one dataset sample
# -----------------------------
def build_sample():
    full_seq = generate_dyck_sequence()
    partial = make_partial(full_seq)

    stack = compute_stack(partial)
    closing = "".join(OPEN_TO_CLOSE[ch] for ch in reversed(stack))
    completed = partial + closing

    reasoning = make_reasoning(stack)

    return {
        "prompt": (
            "Complete the following Dyck language sequence by adding the minimal "
            "necessary closing brackets.\n\n"
            f"Sequence: {partial}"
        ),
        "response": reasoning + completed
    }

# -----------------------------
# Generate dataset
# -----------------------------
def generate_dataset(n_samples=1000, output_file="dyck_reasoning_dataset.jsonl"):
    with open(output_file, "w", encoding="utf-8") as f:
        for _ in range(n_samples):
            sample = build_sample()
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Generated {n_samples} samples → {output_file}")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    random.seed(42)
    generate_dataset(n_samples=100000)
