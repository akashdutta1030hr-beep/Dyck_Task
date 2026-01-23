# Jinja Template for dataset format
JINJA_TEMPLATE = """
{
    "messages": [
        {
            "role": "system",
            "content": "You are a highly capable assistant that completes partial Dyck sequences with valid matching brackets. "
                       "For each given incomplete sequence, explain the reasoning step by step using the <think> tag. "
                       "Each step should detail the process of matching brackets: opening brackets are added to the stack, "
                       "and when a closing bracket matches an opening one, it is removed. Make sure each explanation is detailed and comprehensive, "
                       "using line breaks (\\\\n) to separate reasoning steps. Once reasoning is complete, output the completed Dyck sequence and end the message with <｜end▁of▁sentence｜>."
        },
        {
            "role": "user",
            "content": "Complete this Dyck sequence: {{partial_sequence}}"
        },
        {
            "role": "assistant",
            "content": "<｜Assistant｜><think>\\n{{reasoning}}\\n</think>Here is the completed Dyck sequence: {{completed_sequence}}<｜end▁of▁sentence｜>"
        }
    ]
}
"""