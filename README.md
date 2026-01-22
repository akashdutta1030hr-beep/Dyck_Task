# Dyck Language Dataset Generator

A Python tool for generating training datasets for Dyck language bracket matching tasks with step-by-step reasoning.

## Overview

This project generates synthetic datasets for training language models on Dyck language completion tasks. The Dyck language consists of properly matched brackets, and the task is to complete a partial sequence (containing only opening brackets) by adding the minimal necessary closing brackets.

## Files

### `generator.py`
The main dataset generator script. It contains:

- **Configuration constants** (at the top):
  - `N_SAMPLES`: Number of samples to generate (default: 100,000)
  - `MIN_LEN`: Minimum sequence length (default: 4)
  - `MAX_LEN`: Maximum sequence length (default: 80)
  - `OUTPUT_PATH`: Output file path (default: "dyck_prompt_response_chat_reasoning.jsonl")
  - `SEED`: Random seed for reproducibility (default: 42)

- **Core functions**:
  - **`generate_partial_dyck_sequence(min_len, max_len)`**: Generates a partial Dyck sequence consisting only of opening brackets
  - **`compute_closing_sequence(seq)`**: Computes the minimal required closing brackets for a given sequence
  - **`generate_reasoning(seq)`**: Generates step-by-step reasoning text explaining the stack-based approach
  - **`build_prompt(seq)`**: Creates the user prompt with instructions and rules
  - **`build_response(seq)`**: Creates the assistant response with reasoning and answer
  - **`generate_dataset(n_samples, output_path)`**: Generates the full dataset in JSONL format

**Usage:**
```bash
python generator.py
```

This will generate 100,000 samples and save them to `dyck_prompt_response_chat_reasoning.jsonl`.

### `fine_tune.py`
Fine-tuning script for training language models on the generated dataset. Uses Hugging Face Transformers to fine-tune a causal language model.

**Configuration constants** (at the top):
- `MODEL_NAME`: Base model to fine-tune (default: "unsloth/DeepSeek-R1-Distill-Qwen-1.5B")
- `OUTPUT_DIR`: Directory to save the fine-tuned model (default: "./output")
- `TRAIN_FILE`: Path to the training dataset (default: "dyck_prompt_response_chat_reasoning.jsonl")
- `NUM_TRAINING_STEPS`: Number of training steps (default: 4000)
- `BATCH_SIZE`: Training batch size per device (default: 8)
- `LR`: Learning rate (default: 1.5e-5)
- `EPOCHS`: Number of training epochs (default: 1)
- `WARMUP_STEPS`: Learning rate warmup steps (default: 200)

**Features:**
- Automatic dataset loading from JSONL
- Tokenization with max length 512
- Mixed precision training (FP16)
- Gradient accumulation for effective larger batch sizes
- Automatic model and tokenizer saving

**Usage:**
```bash
python fine_tune.py
```

**Note:** Before running, ensure you have installed the required dependencies:
```bash
pip install transformers datasets torch
```

### `run.sh`
Bash script wrapper for running the dataset generation. Automatically detects and uses the available Python interpreter (python3 or python).

**Usage:**
```bash
bash run.sh
```

### `dyck_prompt_response_chat_reasoning.jsonl`
The generated dataset file in JSONL format (JSON Lines). Each line contains a JSON object with:
- `prompt`: The task prompt asking to complete the Dyck sequence with detailed rules
- `response`: The model response containing:
  - `<|Assistant|>` tag
  - `<think>` section with step-by-step reasoning (stack-based approach)
  - The completed sequence
  - `</think>` tag
  - `<|EOT|>` end token

**Example entry:**
```json
{
  "prompt": "Complete the following Dyck language sequence by adding the minimal necessary closing brackets.\n\nSequence: ({[[[((<((([[([<[<\n\nRules:\n- Add only the required closing brackets\n- Do not add extra pairs\n\nProvide only the complete valid sequence.",
  "response": "<|Assistant|>\n<think>\nUse a stack to track unmatched opening brackets.\nUnmatched openings in order: (, {, [, [, [, (, (, <, (, (, (, [, [, (, [, <, [, <.\nClose them in reverse order: >, ], >, ], ), ], ], ), ), ), >, ), ), ], ], ], }, ).\n</think>\n({[[[((<((([[([<[<>]>])]])))>))]]]})\n<|EOT|>"
}
```

## Dataset Format

The dataset uses JSONL (JSON Lines) format where:
- Each line is a valid JSON object
- Each object contains:
  - `prompt`: Instruction, partial sequence (opening brackets only), and rules
  - `response`: Assistant response with reasoning in `<think>` tags and the completed sequence

## Bracket Types

The generator supports 4 bracket types:
- `()` - Parentheses
- `[]` - Square brackets
- `{}` - Curly braces
- `<>` - Angle brackets

## Algorithm

1. **Generate partial sequence**: Create a sequence containing only opening brackets (length between MIN_LEN and MAX_LEN)
2. **Compute closing sequence**: Determine the minimal required closing brackets by reversing the sequence and mapping to closing brackets
3. **Generate reasoning**: Create step-by-step explanation showing:
   - Unmatched opening brackets in order
   - Closing brackets needed in reverse order
4. **Build prompt**: Format the user prompt with instructions and rules
5. **Build response**: Format the assistant response with reasoning in `<think>` tags and the completed sequence

## Requirements

### For Dataset Generation
- Python 3.x
- Standard library only (no external dependencies)

### For Fine-tuning
- Python 3.8+
- PyTorch
- transformers
- datasets

Install fine-tuning dependencies:
```bash
pip install transformers datasets torch
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Customization

You can modify the generator behavior by editing the configuration constants at the top of `generator.py`:

- **Number of samples**: Change `N_SAMPLES` (default: 100,000)
- **Sequence length range**: Modify `MIN_LEN` and `MAX_LEN` (defaults: 4 and 80)
- **Output file**: Change `OUTPUT_PATH` (default: "dyck_prompt_response_chat_reasoning.jsonl")
- **Random seed**: Change `SEED` for different random sequences (default: 42)

## Example Usage

```python
from generator import generate_dataset

# Generate 1000 samples
generate_dataset(n_samples=1000, output_file="my_dataset.jsonl")
```

## Fine-tuning

### Quick Start

1. **Generate the dataset:**
   ```bash
   python generator.py
   ```

2. **Fine-tune the model:**
   ```bash
   python fine_tune.py
   ```

### Dataset Format for Fine-tuning

**Note:** The `fine_tune.py` script expects the dataset to have a `text` field. If your dataset uses `prompt` and `response` fields, you'll need to combine them into a single `text` field or modify the script accordingly.

### Fine-tuning Configuration

Edit the configuration constants in `fine_tune.py` to customize training:

- **Model selection**: Change `MODEL_NAME` to use a different base model
- **Training steps**: Adjust `NUM_TRAINING_STEPS` based on dataset size
- **Batch size**: Modify `BATCH_SIZE` based on available VRAM (reduce if OOM errors occur)
- **Learning rate**: Tune `LR` for optimal convergence (typically 1e-5 to 5e-5)
- **Output directory**: Change `OUTPUT_DIR` to save models elsewhere

### Training Parameters

The default training configuration uses:
- **1 epoch**: Reasoning models tend to overfit quickly
- **Gradient accumulation**: 16 steps to simulate larger batch size
- **Mixed precision**: FP16 for faster training and lower memory usage
- **AdamW optimizer**: With weight decay 0.1
- **Checkpointing**: Saves model every 500 steps (keeps last 2 checkpoints)

### Model Output

After training, the fine-tuned model will be saved in `OUTPUT_DIR` with:
- Model weights (`pytorch_model.bin` or `model.safetensors`)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- Configuration files (`config.json`)

### Loading the Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./output"  # or your OUTPUT_DIR
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
```

## Key Features

- **Stack-based reasoning**: Each response includes explicit reasoning about unmatched brackets
- **Chat format**: Responses use `<|Assistant|>` tags and `</think>` markers for structured output
- **Configurable**: Easy to adjust sample count, sequence lengths, and output file
- **Reproducible**: Fixed random seed ensures consistent dataset generation
- **Fine-tuning ready**: Complete fine-tuning script included for training reasoning models

## License

This project is provided as-is for research and educational purposes.

# Reasoning-Model
