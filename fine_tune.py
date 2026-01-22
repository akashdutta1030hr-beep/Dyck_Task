from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json
import os
from config import (
    MODEL_NAME, OUTPUT_DIR, TRAIN_FILE, NUM_TRAINING_STEPS,
    BATCH_SIZE, LR, EPOCHS, WARMUP_STEPS, GRADIENT_ACCUMULATION_STEPS,
    WEIGHT_DECAY, MAX_LENGTH, SAVE_STEPS, SAVE_TOTAL_LIMIT,
    LOGGING_STEPS, USE_FP16, OPTIMIZER
)

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
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

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
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # To simulate larger batch size
    learning_rate=LR,  # Learning rate
    warmup_steps=WARMUP_STEPS,  # Warmup steps for LR scheduler
    weight_decay=WEIGHT_DECAY,  # Weight decay for optimizer
    logging_dir='./logs',  # Log directory
    logging_steps=LOGGING_STEPS,  # Log every N steps
    save_steps=SAVE_STEPS,  # Save the model every N steps
    save_total_limit=SAVE_TOTAL_LIMIT,  # Save only the last N models
    fp16=USE_FP16,  # Mixed precision
    evaluation_strategy="no",  # No evaluation during training
    load_best_model_at_end=False,  # Do not load the best model at the end
    optim=OPTIMIZER,  # Optimizer type
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
