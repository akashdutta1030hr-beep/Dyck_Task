import torch
import json
import torch.nn.functional as F
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
MAX_LENGTH = 768  # Must be >= dataset max token length. Run: python check_dataset_seq_len.py

# Hugging Face: upload adapter after training (set repo name to enable).
HF_REPO = None  # e.g. "your-username/dyck-1.5b-lora"

# Stronger learning signal on the final answer (reasoning is long, answer is short).
# Tokens from "FINAL ANSWER: " onward get this weight; reasoning tokens get 1.0.
ANSWER_LOSS_WEIGHT = 5.0

# Response style: training target is dataset format only (# Thought N, # Step k: add 'X', FINAL ANSWER: ...).
# Not Qwen/DeepSeek-style prose; generator and inference prompts enforce dataset style.

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

# LoRA: r=64 for 60k dataset (good capacity); alpha=128, dropout=0.05 for stability on larger data.
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=128,
    lora_dropout=0.05,
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
dataset = dataset.train_test_split(test_size=0.04)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train samples {len(train_dataset)}, eval samples {len(eval_dataset)}")

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
        padding=False,  # No padding: we need actual user length for assistant_start_idx
        add_special_tokens=True
    )
    assistant_start_idx = len(user_tokenized["input_ids"])
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized.pop("offset_mapping")

    input_ids = tokenized["input_ids"]

    # Step 3: Create labels
    # - User tokens: label = -100 (ignored in loss calculation)
    # - Assistant tokens: label = actual token ID (loss computed here)
    labels = [-100] * len(input_ids)

    # Adjust start index if truncation occurred
    assistant_start_idx = min(assistant_start_idx, len(input_ids))

    # Mark assistant tokens for training
    for i in range(assistant_start_idx, len(input_ids)):
        if input_ids[i] != tokenizer.pad_token_id:
            labels[i] = input_ids[i]

    # Step 4: Label weights — stronger signal on "FINAL ANSWER: <sequence>" (answer tokens get ANSWER_LOSS_WEIGHT)
    label_weights = [1.0] * len(input_ids)
    answer_start_char = full_text.find("FINAL ANSWER: ")
    if answer_start_char >= 0 and offset_mapping:
        answer_start_token = None
        for idx in range(assistant_start_idx, len(offset_mapping)):
            if offset_mapping[idx][0] >= answer_start_char:
                answer_start_token = idx
                break
        if answer_start_token is not None:
            for j in range(assistant_start_idx, len(input_ids)):
                if labels[j] != -100:
                    label_weights[j] = ANSWER_LOSS_WEIGHT if j >= answer_start_token else 1.0

    pad_id = tokenizer.pad_token_id
    current_len = len(input_ids)
    if current_len < MAX_LENGTH:
        padding_len = MAX_LENGTH - current_len
        input_ids = input_ids + [pad_id] * padding_len
        labels = labels + [-100] * padding_len
        label_weights = label_weights + [0.0] * padding_len
        attention_mask = [1] * current_len + [0] * padding_len
    else:
        attention_mask = [1] * MAX_LENGTH
    tokenized["input_ids"] = input_ids
    tokenized["labels"] = labels
    tokenized["attention_mask"] = attention_mask
    tokenized["label_weights"] = label_weights
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

class DataCollatorWithWeights(DataCollatorForLanguageModeling):
    """Pads and stacks label_weights so Trainer can use weighted loss."""

    def __call__(self, features):
        batch = super().__call__(features)
        if not features or "label_weights" not in features[0]:
            return batch
        max_len = batch["input_ids"].shape[1]
        weights = []
        for f in features:
            w = f.get("label_weights", [1.0] * len(f["input_ids"]))
            if len(w) < max_len:
                w = w + [0.0] * (max_len - len(w))
            else:
                w = w[:max_len]
            weights.append(w)
        batch["label_weights"] = torch.tensor(weights, dtype=torch.float32)
        return batch


data_collator = DataCollatorWithWeights(
    tokenizer=tokenizer,
    mlm=False,
)

# Keep training loss low and avoid spike: conservative LR, longer warmup, tight grad clip.
# With 60k data (~57k train), steps/epoch ~148, 2 epochs ~296 steps; warmup 25% ≈ 74 steps.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=10,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=6e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.25,
    weight_decay=0.01,
    bf16=True,
    max_grad_norm=0.5,
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)


class WeightedLossTrainer(Trainer):
    """Trainer that weights loss on 'FINAL ANSWER: ...' tokens by ANSWER_LOSS_WEIGHT."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Accept kwargs (e.g. num_items_in_batch from Unsloth) so we don't break when the training loop passes extra args.
        label_weights = inputs.pop("label_weights", None)
        if label_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Causal LM: predict next token; loss at position i is for predicting labels[i+1]
        shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().view(-1)
        shift_weights = label_weights[..., 1:].contiguous().view(-1)

        loss_per_token = F.cross_entropy(
            shift_logits, shift_labels, ignore_index=-100, reduction="none"
        )
        mask = (shift_labels != -100).float()
        weighted_sum = (loss_per_token * shift_weights * mask).sum()
        weight_sum = (shift_weights * mask).sum().clamp(min=1e-8)
        loss = weighted_sum / weight_sum

        return (loss, outputs) if return_outputs else loss


trainer = WeightedLossTrainer(
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
    print(f"LoRA adapter saved to {OUTPUT_DIR}")
    print("Tip: If eval loss spiked then dropped at the end, use the checkpoint with lowest eval_loss from training_loss.png (e.g. results/checkpoint-XXX), not the final run.")
    
    # Save merged model (base + LoRA) for inference — single load, equivalent to base+adapter at every layer.
    # Inference should use dataset-style output only (# Thought N, # Step k, FINAL ANSWER: ...), not Qwen/DeepSeek prose.
    print("Saving merged model (base + LoRA) for inference...")
    model.save_pretrained_merged(f"{OUTPUT_DIR}_merged", tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {OUTPUT_DIR}_merged")

    if HF_REPO:
        print(f"Uploading to Hugging Face: {HF_REPO} ...")
        try:
            model.push_to_hub(HF_REPO, safe_serialization=True)
            tokenizer.push_to_hub(HF_REPO)
            print(f"Upload complete: {HF_REPO}")
        except Exception as e:
            print(f"Upload failed: {e}. Run 'huggingface-cli login' or set HF_TOKEN, then re-run or upload manually: huggingface-cli upload {HF_REPO} {OUTPUT_DIR} . --repo-type model")
    else:
        print("Set HF_REPO to upload the trained adapter to Hugging Face after training.")

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
    

