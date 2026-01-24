import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState

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

# Ensure the pad token is set
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

# Load and shuffle dataset
print(f"Loading dataset from {DATA_PATH}...")
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"Loaded {len(dataset)} samples")

# Validate dataset format
if len(dataset) > 0:
    sample = dataset[0]
    if "messages" not in sample:
        raise ValueError("Dataset must contain 'messages' field in each sample")
    print("Dataset format validated ✓")

dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# Helper function to extract assistant's content
def extract_assistant_content(messages):
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""

# Preprocess function: tokenize chat and align labels with assistant's response
def preprocess(example):
    messages = example["messages"]
    
    # Extract assistant's content for label alignment
    assistant_text = extract_assistant_content(messages)
    if assistant_text == "":
        # If no assistant message, tokenize and return with all labels ignored
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(
            chat_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        tokenized["labels"] = [-100] * len(tokenized["input_ids"])
        return tokenized
    
    # Create messages without assistant response to find where it starts
    messages_before_assistant = [msg for msg in messages if msg["role"] != "assistant"]
    
    # Get full chat text
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Tokenize text before assistant response to find start position
    if messages_before_assistant:
        chat_text_before = tokenizer.apply_chat_template(
            messages_before_assistant, 
            tokenize=False, 
            add_generation_prompt=False
        )
        # Tokenize the prefix to get the number of tokens before assistant response
        tokenized_before = tokenizer(
            chat_text_before,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        assistant_start_token_idx = len(tokenized_before["input_ids"])
    else:
        assistant_start_token_idx = 0
    
    # Tokenize the entire chat text
    tokenized = tokenizer(
        chat_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    
    input_ids = tokenized["input_ids"]
    
    # Initialize labels: -100 means ignore in loss calculation
    labels = [-100] * len(input_ids)
    
    # Adjust start index if truncation occurred
    if assistant_start_token_idx >= len(input_ids):
        # If assistant start is beyond truncation, we can't align properly
        # Fallback: try to find <think> tag in the tokenized sequence
        # This is a heuristic - tokenize a search string
        search_text = "<think>"
        search_tokenized = tokenizer(search_text, add_special_tokens=False)["input_ids"]
        if len(search_tokenized) > 0:
            # Try to find the search pattern in input_ids
            for i in range(len(input_ids) - len(search_tokenized) + 1):
                if input_ids[i:i+len(search_tokenized)] == search_tokenized:
                    assistant_start_token_idx = i
                    break
    
    # Ensure we don't go out of bounds
    assistant_start_token_idx = min(assistant_start_token_idx, len(input_ids))
    
    # Mark all tokens from assistant response onwards as trainable
    # The model should learn to generate both reasoning (<think>...</think>) and the final answer
    for i in range(assistant_start_token_idx, len(input_ids)):
        # Skip padding tokens, but include all other tokens (including special tokens if needed)
        if input_ids[i] != tokenizer.pad_token_id:
            labels[i] = input_ids[i]
    
    tokenized["labels"] = labels
    return tokenized

# Apply preprocessing to train and eval datasets
print("Preprocessing training dataset...")
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
    desc="Preprocessing train",
    num_proc=1,  # Set to 1 to avoid multiprocessing issues, can increase if needed
)

print("Preprocessing evaluation dataset...")
eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
    desc="Preprocessing eval",
    num_proc=1,
)

# Validate preprocessing worked correctly
if len(train_dataset) > 0:
    sample = train_dataset[0]
    if "input_ids" not in sample or "labels" not in sample:
        raise ValueError("Preprocessing failed: missing input_ids or labels")
    # Check that we have some trainable labels (not all -100)
    trainable_labels = sum(1 for label in sample["labels"] if label != -100)
    if trainable_labels == 0:
        print("Warning: No trainable labels found in sample. Check preprocessing function.")
    else:
        print(f"Preprocessing validated ✓ (found {trainable_labels} trainable tokens in sample)")

# Data Collator for Language Modeling (no masking, as it's a causal LM)
# pad_to_multiple_of=8 can help with performance on some hardware
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
    pad_to_multiple_of=None,  # Can set to 8 for tensor core optimization if needed
)

# Training arguments - optimized for better loss performance
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,  # Increased from 2 for better convergence
    learning_rate=4e-5,  # Slightly lower LR for more stable training
    warmup_ratio=0.1,  # Increased warmup for better initial training
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    save_total_limit=3,  # Keep more checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,  # Lower eval_loss is better
    report_to="none",
    dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
    remove_unused_columns=False,  # Keep all columns for debugging
    prediction_loss_only=True,  # Only compute loss, not other metrics
)

# Track training and evaluation loss for plotting
class LossPlottingCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])

    def plot_losses(self):
        # Plot the training and evaluation losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.eval_losses, label="Evaluation Loss", linestyle="--")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Evaluation Losses")
        plt.show()

# Initialize Trainer with the callback for loss plotting
loss_plotter = LossPlottingCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[loss_plotter]
)

if __name__ == "__main__":
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Plot losses after training
    print("Plotting training losses...")
    loss_plotter.plot_losses()
    
    # Save final model and tokenizer
    print(f"Saving model to {MODEL_SAVE_DIR}...")
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(f"Model saved successfully to {MODEL_SAVE_DIR}")
    
    # Evaluate the model after training
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")