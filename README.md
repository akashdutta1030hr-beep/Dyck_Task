# Reasoning Model for Dyck Sequence Completion

This project trains a language model to complete partial Dyck sequences (properly matched bracket sequences) with step-by-step reasoning. The model learns to explain its reasoning process using `<think>` tags before providing the final answer.

## Overview

The project consists of two main components:
1. **Data Generation**: Creates synthetic datasets of partial Dyck sequences with reasoning annotations
2. **Model Training**: Fine-tunes a pre-trained language model using LoRA/QLoRA for efficient training

## Project Structure

```
Reasoning-Model/
├── data/                          # Data generation module
│   ├── __init__.py
│   ├── config.py                  # Bracket pair configurations
│   ├── chat_template.py           # Jinja template for chat format
│   ├── generate_dataset.py        # Main dataset generation function
│   ├── generate_sample.py        # Single sample generation
│   ├── generate_dyck_sequence.py  # Valid Dyck sequence generator
│   ├── generate_partial_dyck_sequence.py  # Partial sequence creator
│   └── generate_reasoning.py      # Reasoning step generator
├── data_generator.py              # Script to generate dataset JSON file
├── Train.py                       # Training script (Python)
├── Train.ipynb                    # Training script (Jupyter Notebook)
└── README.md                      # This file
```

## Features

- **Dyck Sequence Generation**: Generates valid bracket sequences with 1-4 types of brackets: `()`, `[]`, `{}`, `<>`
- **Partial Sequence Creation**: Creates incomplete sequences by randomly removing closing brackets
- **Reasoning Generation**: Automatically generates step-by-step reasoning using stack-based logic
- **Efficient Training**: Uses Unsloth with LoRA/QLoRA for memory-efficient fine-tuning
- **Loss Visualization**: Includes callback to plot training and evaluation losses

## Requirements

Install the required dependencies:

```bash
pip install transformers datasets unsloth matplotlib torch
```

## Usage

### 1. Generate Dataset

First, generate the training dataset:

```bash
python data_generator.py
```

This will create `dyck_language_with_reasoning_dataset.json` with 100,000 samples by default. You can modify the number of samples in `data_generator.py`.

**Configuration in `data_generator.py`:**
- `NUM_SAMPLES`: Number of samples to generate (default: 100,000)
- `OUTPUT_FILE`: Output filename (default: "dyck_language_with_reasoning_dataset.json")

### 2. Train the Model

You can train using either the Python script or Jupyter notebook:

**Option A: Python Script**
```bash
python Train.py
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook Train.ipynb
```

**Training Configuration:**
- **Model**: `unsloth/DeepSeek-R1-Distill-Qwen-1.5B`
- **Max Sequence Length**: 512 tokens
- **LoRA Parameters**:
  - `r=16`: LoRA rank
  - `lora_alpha=32`: LoRA alpha
  - `lora_dropout=0.05`: Dropout rate
- **Training Parameters**:
  - Batch size: 2 per device
  - Gradient accumulation: 8 steps
  - Learning rate: 5e-5
  - Epochs: 2
  - Evaluation: Every 500 steps

**Important**: Update the `DATA_PATH` in `Train.py` or `Train.ipynb` to point to your generated dataset file.

## Data Format

Each sample in the dataset follows this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a highly capable assistant that completes partial Dyck sequences..."
    },
    {
      "role": "user",
      "content": "Complete this Dyck sequence: (([{"
    },
    {
      "role": "assistant",
      "content": "<think>\nEncountered opening bracket '(', pushing it onto the stack.\nEncountered opening bracket '(', pushing it onto the stack.\nEncountered opening bracket '[', pushing it onto the stack.\nEncountered opening bracket '{', pushing it onto the stack.\nStack still contains '{', so append its matching closing bracket '}'.\nStack still contains '[', so append its matching closing bracket ']'.\nStack still contains '(', so append its matching closing bracket ')'.\nStack still contains '(', so append its matching closing bracket ')'.\nAll brackets are matched. The Dyck sequence is now complete.\n</think>Here is the completed Dyck sequence: (([{}]))"
    }
  ]
}
```

## How It Works

### 1. Dyck Sequence Generation
- Generates valid bracket sequences of even length (8-24 characters)
- Supports 1-4 types of brackets: `()`, `[]`, `{}`, `<>`
- Uses a stack-based algorithm to ensure proper matching

### 2. Partial Sequence Creation
- Takes a complete Dyck sequence
- Randomly removes ~40% of closing brackets (60% keep rate)
- Results in an incomplete sequence that needs completion

### 3. Reasoning Generation
- Processes the partial sequence character by character
- Tracks opening brackets on a stack
- Generates step-by-step reasoning for each bracket operation
- Completes the sequence by closing remaining open brackets

### 4. Model Training
- Uses chat template formatting for the conversation
- Labels only the assistant's response (reasoning + completion) for training
- Uses causal language modeling with proper label masking
- Fine-tunes with LoRA for efficient parameter updates

## Model Output

The trained model will:
1. Receive a partial Dyck sequence as input
2. Generate reasoning steps inside `<think>` tags
3. Output the completed Dyck sequence

## Training Output

After training, the model will be saved to:
- `/content/final_lora/` (default path - update as needed)

The training script also:
- Plots training and evaluation losses
- Saves checkpoints every 500 steps
- Evaluates on a held-out test set (5% of data)

## Customization

### Modify Dataset Size
Edit `NUM_SAMPLES` in `data_generator.py`:
```python
NUM_SAMPLES = 50000  # Change to desired number
```

### Adjust Sequence Lengths
Edit `generate_sample.py`:
```python
total_length = random.choice([8, 10, 12, 14, 16, 18, 20, 24])  # Modify as needed
```

### Change Bracket Types
Edit `data/config.py` to add or modify bracket pairs:
```python
BRACKET_PAIRS: Dict[int, List[str]] = {
    1: ['(', ')'],
    2: ['(', ')', '[', ']'],
    # Add more types...
}
```

### Adjust Training Parameters
Modify `TrainingArguments` in `Train.py` or `Train.ipynb`:
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `per_device_train_batch_size`: Batch size
- `gradient_accumulation_steps`: Gradient accumulation

## Notes

- The project uses **Unsloth** for efficient training with 4-bit quantization and LoRA
- Training is optimized for GPUs with limited memory
- The model learns to generate both reasoning and the final answer
- All reasoning is contained within `<think>` tags, following DeepSeek-R1 format

## License

[Add your license information here]

## Author

[Add author information here]
