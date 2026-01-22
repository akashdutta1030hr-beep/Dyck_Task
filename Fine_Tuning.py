"""
Fine-tuning script for DeepSeek-R1-Distill-Qwen-1.5B model
Converted from Fine_Tuning.ipynb

Uses dyck_language_with_reasoning_dataset.json. The JSON has no "prompt" or "response"
fields; each item has "messages": [system, user, assistant]. We derive:
  - prompt  = user message "content"
  - response = assistant message "content"

Note: Before running, install: pip install transformers datasets torch
"""

import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Configuration
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(SCRIPT_DIR, "dyck_language_with_reasoning_dataset.json")
MAX_LENGTH = 512  # Maximum sequence length


def load_from_messages(file_path: str):
    """
    Load dyck_language_with_reasoning_dataset.json. The file has no 'prompt' or 'response'
    keys; only 'messages' with role/content. We derive prompt from user, response from assistant.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    responses = []
    for entry in data:
        messages = entry.get("messages", [])
        if len(messages) < 2:
            continue
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        assistant_msg = next((m for m in messages if m.get("role") == "assistant"), None)
        if user_msg is None or assistant_msg is None:
            continue
        prompts.append(user_msg.get("content", ""))
        responses.append(assistant_msg.get("content", ""))
    return prompts, responses


# Load the dataset (prompt/response derived from messages)
print("Loading dataset...")
if not os.path.isfile(TRAIN_FILE):
    raise FileNotFoundError(f"Dataset not found: {TRAIN_FILE}")
prompts, responses = load_from_messages(TRAIN_FILE)
print(f"Loaded {len(prompts)} examples from dyck_language_with_reasoning_dataset.json")

# Build HuggingFace Dataset with derived 'prompt' and 'response'
train_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})

# Check the first few examples to confirm it loaded correctly
print("Dataset loaded. First 5 examples:")
print(train_dataset[:5])

# Load the tokenizer for your model
print(f"\nLoading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    # Format: <|User|>\n{prompt}\n{response}{eos}
    # Dataset assistant "content" already starts with <|Assistant|><think>...; do not add extra <|Assistant|>.
    formatted_texts = [f"<|User|>\n{p}\n{r}{tokenizer.eos_token}" for p, r in zip(examples['prompt'], examples['response'])]

    # Tokenize the combined texts
    tokenized_inputs = tokenizer(formatted_texts,
                                 max_length=MAX_LENGTH,
                                 truncation=True,
                                 padding='max_length')  # Ensure padding to max_length for consistent input size

    # Set input_ids as labels for language modeling tasks
    tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()

    return tokenized_inputs

# Apply the preprocessing function to the training dataset
print("\nPreprocessing dataset...")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'response'])

# Check the structure of the tokenized dataset
print("Tokenized dataset structure:")
print(tokenized_train_dataset[0])
print(f"Number of features in tokenized dataset: {len(tokenized_train_dataset.column_names)}")

# Load the pre-trained model
print(f"\nLoading model from {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory/performance benefits
)

# Print the model to verify its architecture
print("Model loaded:")
print(model)

# Configure training arguments
print("\nSetting up training arguments...")
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    max_steps=4000,
    learning_rate=1.5e-5,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    lr_scheduler_type='cosine',
    warmup_ratio=0.03,
    bf16=True,  # Changed from fp16=True to bf16=True
    gradient_checkpointing=True,
    logging_steps=1,
    save_steps=100,
    output_dir='./results',
    overwrite_output_dir=True,
    seed=42,
    report_to='none'  # Disable wandb reporting
)

print("Training arguments:")
print(training_args)

# Initialize the Trainer
print("\nInitializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)

# Start training
print("\nStarting training...")
trainer.train()

print("\nTraining complete.")
