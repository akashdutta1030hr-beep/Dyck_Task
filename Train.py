import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "conversation.jsonl"
OUTPUT_DIR = "results"
MAX_LENGTH = 2048
DATASET_SIZE = 10000  # conversation.jsonl line count (10k); step schedule tuned for this

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
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# ============================================================================
# Dataset Loading
# ============================================================================
print(f"Loading dataset from {DATA_PATH}...")
# Load JSONL file - each line is a JSON array (not a JSON object)
# We need to read it manually since load_dataset expects JSON objects
data = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            # Store the raw JSON line as "text" field (preprocessing will parse it)
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
    # Parse conversation from JSONL
    # IMPORTANT: When load_dataset("jsonl", ...) loads a JSONL file, it stores each line
    # in a "text" field. So if your JSONL file has:
    #   [{"role": "user", ...}, {"role": "assistant", ...}]
    # The datasets library creates: {"text": "[{\"role\": \"user\", ...}, ...]"}
    # We need to parse this JSON string to get the actual conversation array.
    text_content = example.get("text", "")
    
    # Handle potential formatting issues (extra quotes, whitespace)
    if isinstance(text_content, str):
        text_content = text_content.strip()
        # Remove all leading/trailing quotes (handle cases like '""[...]' or '"[...]"')
        while text_content.startswith('"'):
            text_content = text_content[1:]
        while text_content.endswith('"'):
            text_content = text_content[:-1]
        text_content = text_content.strip()
        
        try:
            conversation = json.loads(text_content)
        except json.JSONDecodeError as e:
            # If parsing fails, try to extract JSON array from the string
            # Find the first '[' and last ']'
            start_idx = text_content.find('[')
            end_idx = text_content.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                conversation = json.loads(text_content[start_idx:end_idx+1])
            else:
                raise ValueError(f"Failed to parse JSON from text: {text_content[:100]}... Error: {e}")
    else:
        conversation = text_content

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

    # OUTPUT: Assistant's content (reasoning + completion with clear delimiter)
    # - reasoning_content: step-by-step reasoning (e.g., "# Thought 1: ...")
    # - content: the final completion/answer (standalone)
    # 
    # Format: {reasoning}\n\nFINAL ANSWER: {answer}
    # This teaches the model to provide both reasoning AND a clearly delimited answer
    assistant_reasoning = assistant_msg.get("reasoning_content", "").strip()
    assistant_completion = assistant_msg.get("content", "").strip()
    
    # Combine reasoning + answer with clear delimiter
    if assistant_reasoning and assistant_completion:
        assistant_content = f"{assistant_reasoning}\n\nFINAL ANSWER: {assistant_completion}"
    else:
        # Fallback: use whichever is available
        assistant_content = assistant_completion or assistant_reasoning
    
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
        padding=True,
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
    
    # Pad to MAX_LENGTH so all samples have the same length (required for batching)
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

# ============================================================================
# Training Setup
# ============================================================================
# Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

# ============================================================================
# Training Arguments - Optimized for 10k Dataset
# ============================================================================
# Model: 1.5B parameters (DeepSeek-R1-Distill-Qwen-1.5B)
# Dataset: ~10,000 samples -> ~9,500 train / ~500 eval, ~148 steps per epoch
# Method: LoRA fine-tuning (parameter-efficient)
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=50,                   # Eval ~3x per epoch with 10k dataset
    save_steps=50,
    logging_steps=30,
    per_device_train_batch_size=16,   # Increased from 8 (larger batch for stability)
    per_device_eval_batch_size=4,     # Smaller than train to avoid OOM during eval (eval runs with train state in VRAM)
    gradient_accumulation_steps=4,    # Reduced from 8 (maintains effective batch size ~64)
    num_train_epochs=2,
    learning_rate=2e-5,              # Keep current LR (don't reduce)
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,                # Reduced from 0.25 (0.15 was already high, 0.1 is reasonable)
    weight_decay=0.005,
    bf16=True,
    max_grad_norm=1.0,               # Reverted to 1.0 (reasonable value)
    save_total_limit=1,
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
    print("  - Output: Assistant's reasoning_content (step-by-step thoughts + answer)")
    print("  - Loss: Computed only on assistant tokens")
    print("  - Format: # Thought N: ... \\n Here is the completed sequence: ...")
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

    # ========================================================================
    # Plot Training Loss Graph
    # ========================================================================
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
    
    # Create the plot
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
    

