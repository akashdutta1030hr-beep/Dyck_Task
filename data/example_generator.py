import random
from data.dyck_completion import complete_dyck_sequence

# Function to generate a single dataset example with reasoning inside <think> tag
def generate_dyck_task_example():
    # Define possible types of brackets
    opening = ['(', '[', '{', '<']
    closing = [')', ']', '}', '>']
    
    # Randomly generate a variable-length Dyck sequence (the length will vary)
    length = random.randint(4, 12)  # Generate sequence length between 4 and 12
    
    partial_sequence = ""
    for _ in range(length):
        partial_sequence += random.choice(opening)  # Random opening bracket
    
    # Complete the Dyck sequence
    completed_sequence, reasoning_steps = complete_dyck_sequence(partial_sequence)
    
    # Create reasoning content for the assistant's response
    reasoning_content = "\n".join(reasoning_steps)
    
    # Format the dataset to match the provided chat template, using <think> tag for reasoning
    dataset_entry = {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that completes partial Dyck sequences with valid matching brackets, explaining the reasoning behind each step using the <think> tag."
            },
            {
                "role": "user",
                "content": f"Complete this Dyck sequence: {partial_sequence}"
            },
            {
                "role": "assistant",
                "content": f"<｜Assistant｜><think>\n{reasoning_content}\n</think><｜end▁of▁sentence｜>Here is the completed Dyck sequence: {completed_sequence}<｜end▁of▁sentence｜>"
            }
        ]
    }

    return dataset_entry