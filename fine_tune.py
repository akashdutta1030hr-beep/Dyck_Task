from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json
import os

# ===============================
# Configuration
# ===============================
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR = "./output"  # where to save the fine-tuned model
TRAIN_FILE = "dyck_prompt_response_chat_reasoning.jsonl"  # path to your dataset
NUM_TRAINING_STEPS = 4000  # Fine-tuning steps (adjust based on dataset size)
BATCH_SIZE = 8  # Change based on VRAM availability
LR = 1.5e-5  # Learning rate, adjust if necessary
EPOCHS = 1  # Training for 1 epoch as reasoning models tend to overfit fast
WARMUP_STEPS = 200  # Learning rate warmup steps

# ===============================
# Load the dataset
# ===============================

# Load the dataset from JSONL
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f.readlines()]

train_data = load_dataset(TRAIN_FILE)

# Create a Dataset object
train_dataset = Dataset.from_dict({
    'text': [entry['text'] for entry in train_data]
})

# ===============================
# Tokenization
# ===============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)

# ===============================
# Model Loading
# ===============================

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ===============================
# Training Arguments
# ===============================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,  # Where to save the model
    num_train_epochs=EPOCHS,  # Number of epochs
    per_device_train_batch_size=BATCH_SIZE,  # Batch size per device
    gradient_accumulation_steps=16,  # To simulate larger batch size
    learning_rate=LR,  # Learning rate
    warmup_steps=WARMUP_STEPS,  # Warmup steps for LR scheduler
    weight_decay=0.1,  # Weight decay for optimizer
    logging_dir='./logs',  # Log directory
    logging_steps=200,  # Log every 200 steps
    save_steps=500,  # Save the model every 500 steps
    save_total_limit=2,  # Save only the last 2 models
    fp16=True,  # Mixed precision
    evaluation_strategy="no",  # No evaluation during training
    load_best_model_at_end=False,  # Do not load the best model at the end
    optim="adamw_torch",  # Use AdamW optimizer
)

# ===============================
# Trainer Setup
# ===============================

trainer = Trainer(
    model=model,  # Model to fine-tune
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training data
)

# ===============================
# Start Training
# ===============================

trainer.train()

# Save the fine-tuned model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuned model saved at {OUTPUT_DIR}")
