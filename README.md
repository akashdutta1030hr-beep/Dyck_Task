# Reasoning Model for Dyck Sequence Completion

This project trains a language model to complete partial Dyck sequences (properly matched bracket sequences) with step-by-step reasoning. The model learns to explain its reasoning process using step-by-step thoughts before providing the final answer.

## Overview

The project consists of two main components:
1. **Data Generation**: Creates synthetic datasets of partial Dyck sequences with reasoning annotations
2. **Model Training**: Fine-tunes a pre-trained language model using LoRA/QLoRA for efficient training

## Project Structure

```
Dyck_Task/
├── generator.py                   # Data generation script
├── conversation.jsonl             # Generated dataset (JSONL format)
├── Train.py                       # Training script (Python)
├── Train.ipynb                    # Training script (Jupyter Notebook)
└── README.md                      # This file
```

## Features

- **Dyck Sequence Generation**: Generates valid bracket sequences with up to 8 types of brackets: `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`
- **Partial Sequence Creation**: Creates incomplete sequences by cutting at a specific point, leaving unmatched opening brackets
- **Reasoning Generation**: Automatically generates step-by-step reasoning using stack-based logic with "# Thought" format
- **Efficient Training**: Uses Unsloth with LoRA/QLoRA for memory-efficient fine-tuning
- **Reasoning Model Training**: Trains the model to predict both reasoning and completion given user input

## Requirements

Install the required dependencies:

```bash
pip install transformers datasets unsloth torch
```

## Usage

### 1. Generate Dataset

First, generate the training dataset:

```bash
python generator.py
```

This will create `conversation.jsonl` with 10,000 samples by default. The generator creates a JSONL file where each line contains a JSON array with user and assistant messages.

**Configuration in `generator.py`:**
- Number of samples: Modify the `range(10000)` in the `__main__` block (default: 10,000)
- `n_types`: Number of bracket types to use (default: 6, range: 1-8)
- `total_length`: Total sequence length (default: 40)
- `to_fill_length`: Length of closing brackets to generate (default: 20)
- `nesting_depth`: Minimum nesting depth (default: 3)

### 2. Train the Model

Train using the Python script:

```bash
python Train.py
```

**Training Configuration:**
- **Model**: `unsloth/DeepSeek-R1-Distill-Qwen-1.5B`
- **Max Sequence Length**: 512 tokens
- **LoRA Parameters**:
  - `r=16`: LoRA rank
  - `lora_alpha=32`: LoRA alpha
  - `lora_dropout=0.05`: Dropout rate
- **Training Parameters**:
  - Batch size: 4 per device
  - Gradient accumulation: 4 steps (effective batch size: 16)
  - Learning rate: 2e-4
  - Epochs: 5
  - Learning rate scheduler: cosine
  - Warmup ratio: 0.1
  - Weight decay: 0.01
  - Gradient clipping: 1.0
  - Mixed precision: fp16
  - Evaluation: Every 250 steps
  - Checkpoint saving: Every 250 steps
  - Logging: Every 50 steps
  - Best model selection: Based on eval_loss (keeps top 3 checkpoints)

**Important**: 
- Update the `DATA_PATH` in `Train.py` to point to your generated dataset file (default: `conversation.jsonl`)
- The training script automatically handles the JSONL format

## Data Format

The generator creates a JSONL file (`conversation.jsonl`) where each line is a JSON array containing a conversation:

```json
[
  {
    "role": "user",
    "content": "Complete the following Dyck language sequence by adding the minimal necessary closing brackets.\n\nSequence: {{[{⟨⟦({[{(⟨{[(⟨(<<{\n\nRules:\n- Add only the closing brackets needed to match all unmatched opening brackets\n- Do not add any extra bracket pairs beyond what is required\n\nProvide only the complete valid sequence.",
    "reasoning_content": ""
  },
  {
    "role": "assistant",
    "content": "{{[{⟨⟦({[{(⟨{[(⟨(<<{}>>)⟩)]}⟩)}]})⟧⟩}]}}",
    "reasoning_content": "# Thought 1: 1th character is an opening bracket '{', pushing it onto the stack. Stack: ['}']\n# Thought 2: 2th character is an opening bracket '{', pushing it onto the stack. Stack: ['}', '}']\n...\n# Thought 20: All brackets are matched. The Dyck sequence is now complete. Stack: [...]\nHere is the completed Dyck sequence: {{[{⟨⟦({[{(⟨{[(⟨(<<{}>>)⟩)]}⟩)}]})⟧⟩}]}}"
  }
]
```

**Fields:**
- `role`: Either "user" or "assistant"
- `content`: The message content (user's question or assistant's completion)
- `reasoning_content`: Step-by-step reasoning (only present for assistant messages, in "# Thought N:" format)

## Training Method

The training follows a **reasoning model** approach:

### Input-Output Structure
- **Input**: User's `content` (the question/task about completing the Dyck sequence)
- **Output**: Assistant's response combining:
  - `reasoning_content`: Step-by-step reasoning (e.g., "# Thought 1: ...")
  - `content`: The final completion/answer

### Loss Computation
- **User tokens**: Label = `-100` (ignored in loss calculation)
- **Assistant tokens**: Label = actual token ID (loss computed here)
- The model receives the full conversation (user + assistant) but learns to predict only the assistant's response

### Training Objective
The model learns: **P(assistant_output | user_input)**

This means:
- Given the user's question, predict the assistant's reasoning and completion
- Loss is computed only on the assistant's tokens (both reasoning and completion)
- User tokens provide context but don't contribute to the loss

## How It Works

### 1. Dyck Sequence Generation
- Generates valid bracket sequences with configurable length (default: 40 characters)
- Supports up to 8 types of brackets: `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`
- Uses a stack-based algorithm to ensure proper matching
- Ensures minimum nesting depth is met

### 2. Partial Sequence Creation
- Generates a complete valid Dyck sequence
- Cuts the sequence at a specific point (`cut_point = total_length - fill_length`)
- The prefix (up to cut point) contains the partial sequence with unmatched opening brackets
- The suffix (after cut point) contains the closing brackets needed to complete the sequence

### 3. Reasoning Generation
- Processes the partial sequence character by character
- Tracks opening brackets on a stack
- Generates step-by-step reasoning using "# Thought N:" format for each bracket operation
- Shows the stack state after each operation
- Completes the sequence by closing remaining open brackets

### 4. Model Training
- **Preprocessing**:
  1. Parses JSONL format (each line is a JSON array)
  2. Extracts user and assistant messages
  3. Combines assistant's `reasoning_content` and `content` into a single response
  4. Formats as messages for chat template
  5. Tokenizes full conversation (user + assistant)
  6. Creates labels: `-100` for user tokens, actual token IDs for assistant tokens
  
- **Training**:
  - Uses causal language modeling
  - Loss computed only on assistant tokens
  - Fine-tunes with LoRA for efficient parameter updates
  - Model learns to predict assistant's reasoning + completion given user input

## Model Output

The trained model will:
1. Receive a partial Dyck sequence as input (user's question)
2. Generate step-by-step reasoning (in "# Thought" format)
3. Output the completed Dyck sequence

Example:
```
Input: "Complete the following Dyck sequence: [⟨([<⟨{<⟦[{[<⟨<⟨⟦⟨{⟨"

Output:
# Thought 1: 1th character is an opening bracket '[', pushing it onto the stack. Stack: [']']
# Thought 2: 2th character is an opening bracket '⟨', pushing it onto the stack. Stack: [']', '⟩']
...
# Thought 20: All brackets are matched. The Dyck sequence is now complete.
[⟨([<⟨{<⟦[{[<⟨<⟨⟦⟨{⟨⟩}⟩⟧⟩>⟩>]}]⟧>}⟩>])⟩]
```

## Training Output

After training, the model will be saved to:
- `results/` (default path - update `OUTPUT_DIR` in `Train.py` as needed)

The training script:
- Saves checkpoints every 250 steps
- Evaluates on a held-out test set (5% of data)
- Keeps the best 3 checkpoints based on evaluation loss
- Prints training progress and final evaluation results

## Customization

### Modify Dataset Size
Edit the range in `generator.py`:
```python
for i in range(10000):  # Change to desired number
```

### Adjust Sequence Parameters
Edit the `generate()` call in `generator.py`:
```python
task = generator.generate(
    seed=task_id,
    n_types=6,          # Number of bracket types (1-8)
    total_length=40,     # Total sequence length
    to_fill_length=20,   # Length of closing brackets to generate
    nesting_depth=3,     # Minimum nesting depth
    max_attempts=1000    # Max attempts to generate valid sequence
)
```

### Change Bracket Types
The generator supports 8 bracket types by default. To use fewer types, modify `n_types` in the `generate()` call. The available brackets are:
- `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`

### Adjust Training Parameters
Modify `TrainingArguments` in `Train.py`:
- `num_train_epochs`: Number of training epochs (default: 5)
- `learning_rate`: Learning rate (default: 2e-4)
- `per_device_train_batch_size`: Batch size per device (default: 4)
- `gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `lr_scheduler_type`: Learning rate scheduler (default: "cosine")
- `warmup_ratio`: Warmup ratio (default: 0.1)
- `weight_decay`: Weight decay for regularization (default: 0.01)
- `max_grad_norm`: Gradient clipping (default: 1.0)
- `eval_steps`: Evaluation frequency (default: 250)
- `save_steps`: Checkpoint saving frequency (default: 250)
- `MAX_LENGTH`: Maximum sequence length (default: 512)

## Key Points

- **Reasoning Model Training**: The model learns to generate both reasoning and completion
- **Loss Computation**: Loss is computed only on assistant tokens (both reasoning and completion)
- **Efficient Training**: Uses Unsloth with 4-bit quantization and LoRA for memory efficiency
- **JSONL Format**: The dataset uses JSONL format where each line is a JSON array
- **Chat Template**: Uses the model's chat template for proper formatting
- **Training Method**: User content as input, assistant content (reasoning + completion) as output

## Notes

- The project uses **Unsloth** for efficient training with 4-bit quantization and LoRA
- Training is optimized for GPUs with limited memory
- The model learns to generate both reasoning (in "# Thought" format) and the final answer
- The training script automatically handles the JSONL format and combines reasoning with completion
- User tokens provide context but don't contribute to loss calculation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**akashdutta**
