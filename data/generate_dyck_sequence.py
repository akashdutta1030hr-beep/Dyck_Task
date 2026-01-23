import random
from data.config import BRACKET_PAIRS, OPEN_TO_CLOSE

def generate_dyck_sequence(n_types: int, total_length: int) -> str:
    """
    Generate a valid Dyck sequence of even length.
    """
    assert total_length % 2 == 0, "total_length must be even"

    brackets = BRACKET_PAIRS[n_types]
    open_brackets = brackets[::2]

    stack = []
    sequence = []

    remaining_open = total_length // 2
    remaining_close = total_length // 2

    for _ in range(total_length):
        if remaining_open > 0 and (remaining_close == 0 or random.random() < 0.5):
            # Add opening bracket
            b = random.choice(open_brackets)
            stack.append(b)
            sequence.append(b)
            remaining_open -= 1
        else:
            # Add closing bracket
            if stack:
                open_b = stack.pop()
                sequence.append(OPEN_TO_CLOSE[open_b])
                remaining_close -= 1
            else:
                # Force open if stack empty
                b = random.choice(open_brackets)
                stack.append(b)
                sequence.append(b)
                remaining_open -= 1

    # Close any remaining opens
    while stack:
        sequence.append(OPEN_TO_CLOSE[stack.pop()])

    return "".join(sequence)