import random
from data.config import OPEN_TO_CLOSE

def generate_partial_dyck_sequence(complete_sequence: str) -> str:
    """
    Remove some closing brackets to create a partial sequence.
    """
    partial = []
    stack = []

    for ch in complete_sequence:
        if ch in OPEN_TO_CLOSE:
            partial.append(ch)
            stack.append(ch)
        else:
            # Randomly drop closing brackets
            if random.random() < 0.6:
                partial.append(ch)
                if stack:
                    stack.pop()
            else:
                continue

    return "".join(partial)