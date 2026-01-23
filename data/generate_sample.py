import random
from data.generate_dyck_sequence import generate_dyck_sequence
from data.generate_partial_dyck_sequence import generate_partial_dyck_sequence
from data.generate_reasoning import generate_reasoning_and_completion

def generate_sample() -> dict:
    n_types = random.randint(1, 4)
    total_length = random.choice([8, 10, 12, 14, 16, 18, 20, 24])

    complete_sequence = generate_dyck_sequence(n_types, total_length)
    partial_sequence = generate_partial_dyck_sequence(complete_sequence)

    reasoning, completed_sequence = generate_reasoning_and_completion(partial_sequence)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a highly capable assistant that completes partial Dyck sequences. "
                    "Use stack-based reasoning. Explain your reasoning step by step using the <think> tag. "
                    "After reasoning, output the completed Dyck sequence."
                )
            },
            {
                "role": "user",
                "content": f"Complete this Dyck sequence: {partial_sequence}"
            },
            {
                "role": "assistant",
                "content": (
                    "<｜Assistant｜><think>\n"
                    + "\n".join(reasoning)
                    + "\n</think>"
                    f"Here is the completed Dyck sequence: {completed_sequence}"
                    "<｜end▁of▁sentence｜>"
                )
            }
        ]
    }