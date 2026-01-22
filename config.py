"""
Configuration file for Dyck Language Dataset Generator and Fine-tuning
"""

# ===============================
# Dataset Generation Configuration
# ===============================

# Number of samples to generate
N_SAMPLES = 100_000

# Sequence length range
MIN_LEN = 4
MAX_LEN = 80

# Bracket configuration
BRACKETS = ["(", "[", "{", "<"]
PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">"
}

# Output file path for generated dataset
OUTPUT_PATH = "dyck_prompt_response_chat_reasoning.jsonl"

# Random seed for reproducibility
SEED = 42

# ===============================
# Fine-tuning Configuration
# ===============================

# Base model to fine-tune
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B"

# Directory to save the fine-tuned model
OUTPUT_DIR = "./output"

# Path to training dataset
TRAIN_FILE = "dyck_prompt_response_chat_reasoning.jsonl"

# Training hyperparameters
NUM_TRAINING_STEPS = 4000  # Fine-tuning steps (adjust based on dataset size)
BATCH_SIZE = 8  # Batch size per device (change based on VRAM availability)
LR = 1.5e-5  # Learning rate
EPOCHS = 1  # Number of training epochs
WARMUP_STEPS = 200  # Learning rate warmup steps

# Advanced training parameters
GRADIENT_ACCUMULATION_STEPS = 16  # To simulate larger batch size
WEIGHT_DECAY = 0.1  # Weight decay for optimizer
MAX_LENGTH = 512  # Maximum sequence length for tokenization
SAVE_STEPS = 500  # Save checkpoint every N steps
SAVE_TOTAL_LIMIT = 2  # Keep only the last N checkpoints
LOGGING_STEPS = 200  # Log every N steps

# Training options
USE_FP16 = True  # Mixed precision training
OPTIMIZER = "adamw_torch"  # Optimizer type
