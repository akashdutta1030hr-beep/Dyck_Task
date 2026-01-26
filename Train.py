import torch
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "conversation.jsonl"
OUTPUT_DIR = "results"
MAX_LENGTH = 512

# ============================================================================
# Model Setup
# ============================================================================
print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
)

# Ensure the pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Add LoRA (Low-Rank Adaptation) for efficient fine-tuning
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

# ============================================================================
# Dataset Loading
# ============================================================================
print(f"Loading dataset from {DATA_PATH}...")
dataset = load_dataset("jsonl", data_files=DATA_PATH)["train"]
print(f"Loaded {len(dataset)} samples")

# Split dataset into train and eval
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# ============================================================================
# Preprocessing Function
# ============================================================================
def preprocess(example):
    """
    Preprocess function for reasoning model training.
    
    Dataset Structure (JSONL format):
    Each line is a JSON array: [{"role": "user", ...}, {"role": "assistant", ...}]
    
    Training Method:
    - Input: user's content (the question/task)
    - Output: assistant's content (reasoning_content + content/completion)
    - Loss: computed ONLY on assistant tokens (both reasoning and completion)
    - User tokens are ignored in loss calculation (label = -100)
    
    This teaches the model: P(assistant_output | user_input)
    """
    # Parse conversation from JSONL
    # IMPORTANT: When load_dataset("jsonl", ...) loads a JSONL file, it stores each line
    # in a "text" field. So if your JSONL file has:
    #   [{"role": "user", ...}, {"role": "assistant", ...}]
    # The datasets library creates: {"text": "[{\"role\": \"user\", ...}, ...]"}
    # We need to parse this JSON string to get the actual conversation array.
    text_content = example.get("text", "")
    conversation = json.loads(text_content) if isinstance(text_content, str) else text_content

    # Extract user and assistant messages from the conversation array
    user_msg = None
    assistant_msg = None

    for msg in conversation:
        role = msg.get("role", "")
        if role == "user":
            user_msg = msg
        elif role == "assistant":
            assistant_msg = msg

    # ========================================================================
    # Extract Input and Output
    # ========================================================================
    # INPUT: User's content (the question/task)
    user_content = user_msg.get("content", "")

    # OUTPUT: Assistant's content (reasoning + completion)
    # - reasoning_content: step-by-step reasoning (e.g., "# Thought 1: ...")
    # - content: the final completion/answer
    assistant_reasoning = assistant_msg.get("reasoning_content", "").strip()
    assistant_completion = assistant_msg.get("content", "").strip()
    
    # Combine reasoning and completion for assistant's full response
    if assistant_reasoning:
        assistant_content = f"{assistant_reasoning}\n\n{assistant_completion}"
    else:
        assistant_content = assistant_completion
    
    # ========================================================================
    # Format for Chat Template
    # ========================================================================
    # Format as messages for the model's chat template
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    # Apply chat template to get the full conversation text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # ========================================================================
    # Tokenization and Label Alignment
    # ========================================================================
    # Step 1: Tokenize user input only to find where assistant response starts
    user_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=False
    )
    user_tokenized = tokenizer(
        user_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True
    )
    assistant_start_idx = len(user_tokenized["input_ids"])
    
    # Step 2: Tokenize full conversation (user + assistant)
    # The model receives both, but loss is computed only on assistant tokens
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True
    )
    
    input_ids = tokenized["input_ids"]
    
    # Step 3: Create labels
    # - User tokens: label = -100 (ignored in loss calculation)
    # - Assistant tokens: label = actual token ID (loss computed here)
    labels = [-100] * len(input_ids)
    
    # Adjust start index if truncation occurred
    assistant_start_idx = min(assistant_start_idx, len(input_ids))
    
    # Mark assistant tokens for training
    # Loss is computed only on these tokens (includes both reasoning and completion)
    for i in range(assistant_start_idx, len(input_ids)):
        if input_ids[i] != tokenizer.pad_token_id:
            labels[i] = input_ids[i]
    
    tokenized["labels"] = labels
    return tokenized

# ============================================================================
# Apply Preprocessing
# ============================================================================
print("Preprocessing training dataset...")
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
    desc="Preprocessing train",
    num_proc=1,
)

print("Preprocessing evaluation dataset...")
eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
    desc="Preprocessing eval",
    num_proc=1,
)

# Validate preprocessing
if len(train_dataset) > 0:
    sample = train_dataset[0]
    if "input_ids" not in sample or "labels" not in sample:
        raise ValueError("Preprocessing failed: missing input_ids or labels")
    
    trainable_labels = sum(1 for label in sample["labels"] if label != -100)
    if trainable_labels == 0:
        print("Warning: No trainable labels found in sample.")
    else:
        print(f"Preprocessing validated ✓")
        print(f"  - Total tokens: {len(sample['input_ids'])}")
        print(f"  - Assistant tokens (loss computed on): {trainable_labels}")
        print(f"  - User tokens (ignored in loss): {len(sample['input_ids']) - trainable_labels}")

# ============================================================================
# Training Setup
# ============================================================================
# Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# ============================================================================
# Training Arguments - Optimized Configuration
# ============================================================================
# Model: 1.5B parameters (DeepSeek-R1-Distill-Qwen-1.5B)
# Dataset: ~10,000 samples (small dataset)
# Method: LoRA fine-tuning (parameter-efficient)
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=250,
    save_steps=250,
    logging_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
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

# ============================================================================
# Training
# ============================================================================
if __name__ == "__main__":
    print("Starting training...")
    print("=" * 60)
    print("Training Method:")
    print("  - Input: User's content (question/task)")
    print("  - Output: Assistant's content (reasoning + completion)")
    print("  - Loss: Computed only on assistant tokens")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model and tokenizer
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved successfully to {OUTPUT_DIR}")
    
    # Evaluate the model after training
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
