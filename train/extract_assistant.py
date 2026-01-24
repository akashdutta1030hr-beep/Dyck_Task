# Function to extract assistant content from messages
def extract_assistant_content(messages):
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""