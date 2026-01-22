# Function to complete the Dyck sequence and explain the reasoning
def complete_dyck_sequence(partial_sequence):
    stack = []
    closing_brackets = { '(': ')', '[': ']', '{': '}', '<': '>' }
    opening_brackets = { ')': '(', ']': '[', '}': '{', '>': '<' }
    
    # Keep track of the reasoning steps
    reasoning_steps = []
    
    for char in partial_sequence:
        if char in closing_brackets.values():
            if stack and stack[-1] == opening_brackets[char]:
                stack.pop()
                reasoning_steps.append(f"Matched {opening_brackets[char]} with {char}.")
            else:
                raise ValueError(f"Invalid sequence: unexpected closing bracket {char}")
        else:
            stack.append(char)
            reasoning_steps.append(f"Added {char} to the stack.")
    
    # Now, add the missing closing brackets and explain
    result = partial_sequence
    while stack:
        opening = stack.pop()
        closing = closing_brackets[opening]
        result += closing
        reasoning_steps.append(f"Added closing bracket {closing} for {opening}.")
    
    return result, reasoning_steps
