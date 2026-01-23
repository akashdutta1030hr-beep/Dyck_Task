from typing import List, Tuple
from data.config import OPEN_TO_CLOSE, CLOSE_TO_OPEN

# Function to generate reasoning text
def generate_reasoning_and_completion(partial_sequence: str) -> Tuple[List[str], str]:
    stack = []
    reasoning = []

    for ch in partial_sequence:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
            reasoning.append(
                f"Encountered opening bracket '{ch}', pushing it onto the stack."
            )
        else:
            open_match = CLOSE_TO_OPEN[ch]
            reasoning.append(
                f"Encountered closing bracket '{ch}', matching it with '{open_match}' and popping from the stack."
            )
            if stack:
                stack.pop()

    completion = []
    while stack:
        open_bracket = stack.pop()
        close_bracket = OPEN_TO_CLOSE[open_bracket]
        reasoning.append(
            f"Stack still contains '{open_bracket}', so append its matching closing bracket '{close_bracket}'."
        )
        completion.append(close_bracket)

    reasoning.append(
        "All brackets are matched. The Dyck sequence is now complete."
    )

    completed_sequence = partial_sequence + "".join(completion)
    return reasoning, completed_sequence

