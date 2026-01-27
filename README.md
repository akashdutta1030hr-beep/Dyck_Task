# Dyck Sequence Completion with Reasoning

Train a language model to complete partial Dyck sequences (matched bracket sequences) with step-by-step reasoning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generator.py

# 3. Train model
python Train.py

# 4. Run inference
python inference.py
```

## Project Structure

```
Dyck_Task/
├── generator.py          # Generate training data
├── Train.py              # Training script
├── inference.py          # Inference script
├── conversation.jsonl    # Generated dataset
├── results/              # Trained model outputs
└── requirements.txt      # Dependencies
```

## Features

- Generates Dyck sequences with 8 bracket types: `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`
- Step-by-step reasoning generation using stack-based logic
- Efficient training with Unsloth (LoRA/QLoRA, 4-bit quantization)
- Fine-tuned model generates reasoning + completion

## Usage

### Generate Dataset

```bash
python generator.py
```

Creates `conversation.jsonl` with 10,000 samples by default. Each line contains a JSON array with user/assistant messages.

**Configuration**: Edit `generator.py` to modify:
- Number of samples (default: 10,000)
- Bracket types: `n_types` (default: 6, range: 1-8)
- Sequence length: `total_length` (default: 40)
- Closing brackets length: `to_fill_length` (default: 20)

### Train Model

```bash
python Train.py
```

**Model**: `unsloth/DeepSeek-R1-Distill-Qwen-1.5B`  
**Training**: LoRA fine-tuning with 4-bit quantization  
**Output**: Model saved to `results/` directory with training loss graph

**Key Parameters** (edit in `Train.py`):
- Learning rate: `2e-4`
- Epochs: `5`
- Batch size: `4` per device (effective: 16 with gradient accumulation)
- Max sequence length: `512`

### Run Inference

**Interactive mode:**
```bash
python inference.py
```

**Single sequence:**
```bash
python inference.py "{{[{⟨⟦({[{(⟨{[(⟨(<<{"
```

**Examples:**
```bash
python inference.py --examples
```

## How It Works

1. **Data Generation**: Creates valid Dyck sequences, cuts them at a point, and generates step-by-step reasoning
2. **Training**: Model learns to predict reasoning + completion given partial sequence
3. **Inference**: Model generates step-by-step thoughts followed by the completed sequence

## Training Details

- **Input**: User's question with partial Dyck sequence
- **Output**: Assistant's reasoning (in "# Thought N:" format) + completion
- **Loss**: Computed only on assistant tokens (user tokens ignored)
- **Method**: Causal language modeling with LoRA fine-tuning

## Output Files

- `results/`: Fine-tuned model and checkpoints
- `results/training_loss.png`: Training and evaluation loss graph
- `conversation.jsonl`: Generated training dataset

## Requirements

See `requirements.txt` for full list. Main dependencies:
- `torch`
- `transformers`
- `datasets`
- `unsloth`
- `matplotlib`

## Author

**akashdutta**
