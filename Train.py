# Install required packages
# !pip install -U unsloth transformers datasets accelerate bitsandbytes sentencepiece

import torch
from datasets import load_dataset
from Train.preprocess import preprocess
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Constants
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "/content/dyck_language_with_reasoning_dataset.json"
OUTPUT_DIR = "/content/results"
MAX_LENGTH = 512

# Load pre-trained model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ensure the pad token is set (required for Qwen/DeepSeek)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add LoRA (Low-Rank Adaptation) to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# Load and preprocess dataset
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Print a sample from the dataset to check
print(train_dataset[0])


# Apply preprocessing to train and eval datasets
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
)

# Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16
    num_train_epochs=2,
    learning_rate=5e-5,  # QLoRA LR
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save final model and tokenizer
model.save_pretrained("/content/final_lora")
tokenizer.save_pretrained("/content/final_lora")

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)