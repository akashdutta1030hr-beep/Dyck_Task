import torch
import json
import matplotlib.pyplot as plt
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "conversation.jsonl"
OUTPUT_DIR = "results"
MAX_LENGTH = 2048

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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# ============================================================================
# Dataset Loading
# ============================================================================
print(f"Loading dataset from {DATA_PATH}...")
data = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data.append({"text": line.strip()})
dataset = Dataset.from_list(data)
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
    - Assistant has TWO fields: reasoning_content (reasoning) + content (answer)
    
    Training Method:
    - Input: user's content (the question/task)
    - Output: {reasoning}\n\nFINAL ANSWER: {answer}
    - Loss: computed ONLY on assistant tokens
    - User tokens are ignored in loss calculation (label = -100)
    
    This teaches the model to:
    1. Generate step-by-step reasoning
    2. Provide a clearly delimited final answer
    
    Format: P(reasoning + "FINAL ANSWER:" + answer | user_input)
    """
    text_content = example.get("text", "")
    conversation = json.loads(text_content)

    # Extract user and assistant messages from the conversation array
    user_msg = None
    assistant_msg = None

    for msg in conversation:
        role = msg.get("role", "")
        if role == "user":
            user_msg = msg
        elif role == "assistant":
            assistant_msg = msg

    user_content = user_msg.get("content", "")

    assistant_reasoning = assistant_msg.get("reasoning_content", "").strip()
    assistant_completion = assistant_msg.get("content", "").strip()
    
    if assistant_reasoning and assistant_completion:
        assistant_content = f"{assistant_reasoning}\n\nFINAL ANSWER: {assistant_completion}"
    else:
        assistant_content = assistant_completion or assistant_reasoning

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    user_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=False
    )
    user_tokenized = tokenizer(
        user_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        add_special_tokens=True
    )
    assistant_start_idx = len(user_tokenized["input_ids"])
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
    
    pad_id = tokenizer.pad_token_id
    current_len = len(input_ids)
    if current_len < MAX_LENGTH:
        padding_len = MAX_LENGTH - current_len
        input_ids = input_ids + [pad_id] * padding_len
        labels = labels + [-100] * padding_len  # ignore padding in loss
        attention_mask = [1] * current_len + [0] * padding_len
    else:
        attention_mask = [1] * MAX_LENGTH
    tokenized["input_ids"] = input_ids
    tokenized["labels"] = labels
    tokenized["attention_mask"] = attention_mask
    return tokenized

# ============================================================================
# Apply Preprocessing
# ============================================================================
print("Preprocessing training dataset...")
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
    desc="Preprocessing train",
    num_proc=8,
)

print("Preprocessing evaluation dataset...")
eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
    desc="Preprocessing eval",
    num_proc=8,
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.005,
    bf16=True,
    max_grad_norm=1.0,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
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
    trainer.train()
    
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved successfully to {OUTPUT_DIR}")
    
    print(f"\nMerging LoRA with base model and saving to {OUTPUT_DIR_MERGED}...")
    model.save_pretrained_merged(OUTPUT_DIR_MERGED, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved successfully to {OUTPUT_DIR_MERGED} (use this path for inference)")
    
    # Evaluate the model after training
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")

    print("\nGenerating training loss graph...")
    history = trainer.state.log_history
    
    # Extract training loss and evaluation loss
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    for log in history:
        if 'loss' in log and 'step' in log:
            train_losses.append(log['loss'])
            train_steps.append(log['step'])
        if 'eval_loss' in log and 'step' in log:
            eval_losses.append(log['eval_loss'])
            eval_steps.append(log['step'])
    plt.figure(figsize=(12, 6))
    
    if train_losses:
        plt.plot(train_steps, train_losses, label='Training Loss', marker='o', markersize=3, linewidth=1.5)
    
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Evaluation Loss', marker='s', markersize=3, linewidth=1.5)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Evaluation Loss Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    loss_graph_path = f"{OUTPUT_DIR}/training_loss.png"
    plt.savefig(loss_graph_path, dpi=300, bbox_inches='tight')
    print(f"Training loss graph saved to {loss_graph_path}")
    
    # Optionally display the plot (comment out if running in headless environment)
    # plt.show()
    plt.close()

    print("Training complete! 🎉")
    

