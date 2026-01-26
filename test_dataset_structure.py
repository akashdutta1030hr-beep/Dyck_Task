"""Test script to show how datasets library loads JSONL files"""

import json
from datasets import load_dataset

# Load the dataset
print("Loading dataset...")
dataset = load_dataset("jsonl", data_files="conversation.jsonl")["train"]

# Check the first example
print("\n" + "="*60)
print("First example structure:")
print("="*60)
first_example = dataset[0]
print(f"Type: {type(first_example)}")
print(f"Keys: {list(first_example.keys())}")
print(f"\nHas 'text' field: {'text' in first_example}")

if 'text' in first_example:
    text_content = first_example['text']
    print(f"\n'text' field type: {type(text_content)}")
    print(f"'text' field length: {len(text_content)}")
    print(f"\nFirst 200 characters of 'text':")
    print(text_content[:200])
    
    # Try to parse it
    print("\n" + "-"*60)
    print("Parsing 'text' field as JSON:")
    print("-"*60)
    try:
        parsed = json.loads(text_content)
        print(f"Parsed type: {type(parsed)}")
        print(f"Parsed length: {len(parsed) if isinstance(parsed, list) else 'N/A'}")
        if isinstance(parsed, list) and len(parsed) > 0:
            print(f"First item type: {type(parsed[0])}")
            print(f"First item keys: {list(parsed[0].keys()) if isinstance(parsed[0], dict) else 'N/A'}")
    except Exception as e:
        print(f"Error parsing: {e}")

print("\n" + "="*60)
print("Raw JSONL file structure (first line):")
print("="*60)
with open("conversation.jsonl", "r", encoding="utf-8") as f:
    first_line = f.readline().strip()
    print(f"First line type: {type(first_line)}")
    print(f"First line length: {len(first_line)}")
    print(f"First 200 characters: {first_line[:200]}")
