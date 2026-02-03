# Dyck Task

Fine-tune a language model to **complete Dyck sequences** (balanced bracket sequences) with **step-by-step reasoning**.

## Task

Given a prefix of opening brackets, complete it with the **minimal closing brackets** so the full sequence is a valid Dyck word.

**Bracket pairs:** `()`, `[]`, `{}`, `<>`

## Project Structure

| File | Description |
|------|-------------|
| `generator.py` | Generates `conversation.jsonl` (user/assistant with reasoning). |
| `Train.py` | Fine-tunes [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) using Unsloth (LoRA, 4-bit). |
| `inference.py` | Loads trained model and runs Dyck completion. |
| `check_dataset_seq_len.py` | Reports dataset token-length stats; run before training. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

**GPU (PyTorch):** `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Unsloth (latest):** `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

## Dataset

```bash
python generator.py
```

Creates `conversation.jsonl` with reasoning + final answer per sample. Default 60k samples (edit loop in `generator.py` to change).

Before training, run `python check_dataset_seq_len.py` and set `MAX_LENGTH` in `Train.py` to at least the reported max.

## Training

```bash
python Train.py
```

**Config:** 60k data → ~57k train / ~3k eval; 2 epochs; LoRA r=64; effective batch 384; LR 6e-6, warmup 25%, max_grad_norm=0.5; weighted loss on "FINAL ANSWER" tokens (5×).

**Outputs:** `results/` (adapter), `results_merged/` (merged model for inference), `results/training_loss.png`. If loss spiked, use the checkpoint with lowest eval_loss from the plot.

## Inference

```bash
python inference.py
```

Loads merged model (or set `MODEL_ID` in `inference.py`). Edit `SEQUENCE` to try other inputs. Output format: dataset style (`# Thought N:`, `# Step k: add 'X'.`, `FINAL ANSWER: <sequence>`).

## Model

**Base:** [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B). Fine-tuned adapter in `results/`, merged in `results_merged/`.
