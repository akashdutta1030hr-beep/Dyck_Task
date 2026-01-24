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
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.05)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Helper function to extract assistant's content
def extract_assistant_content(messages):
    for msg in messages:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""

# Preprocess function: tokenize chat and align labels with assistant's response
def preprocess(example):
    messages = example["messages"]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Tokenize the entire chat text
    tokenized = tokenizer(
        chat_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    
    input_ids = tokenized["input_ids"]
    labels = [-100] * len(input_ids)  # Default label is -100 (ignore token)

    assistant_text = extract_assistant_content(messages)
    if assistant_text == "":
        tokenized["labels"] = labels
        return tokenized

    # Find where the assistant's response starts
    start_idx = chat_text.find(assistant_text)
    if start_idx == -1:
        tokenized["labels"] = labels
        return tokenized

    prefix = chat_text[:start_idx]
    prefix_ids = tokenizer(prefix, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    assistant_ids = tokenizer(
        assistant_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    # Label the assistant's response correctly
    start = len(prefix_ids)
    end = min(start + len(assistant_ids), len(input_ids))

    for i in range(start, end):
        labels[i] = input_ids[i]

    tokenized["labels"] = labels
    return tokenized

# Apply preprocessing to train and eval datasets
train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    preprocess,
    remove_columns=eval_dataset.column_names,
)

# Data Collator for Language Modeling (no masking, as it's a causal LM)
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
    gradient_accumulation_steps=8,
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

# Start training
trainer.train()

# Plot losses after training
loss_plotter.plot_losses()

# Save final model and tokenizer
model.save_pretrained("/content/final_lora")
tokenizer.save_pretrained("/content/final_lora")

# Evaluate the model after training
eval_results = trainer.evaluate()
print(f"Final evaluation results: {eval_results}")