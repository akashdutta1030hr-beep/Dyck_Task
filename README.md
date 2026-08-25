# Dyck Task

**Generate and fine-tune on Dyck-language completion with explicit step-by-step reasoning.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Unsloth-yellow)](https://github.com/unslothai/unsloth)

Formal-language tasks stress **structure and reasoning**, not world knowledge. Dyck languages (balanced bracket sequences) are a clean benchmark: the model must track a stack, emit minimal closings, and explain each step.

This repo is a full pipeline: **synthetic data → length audit → LoRA fine-tune → inference**.

## What this demonstrates

| Skill | Where in repo |
|-------|----------------|
| Synthetic dataset design | `generator.py` — reasoning traces + `FINAL ANSWER` format |
| Training hygiene | `check_dataset_seq_len.py` before setting `MAX_LENGTH` |
| Efficient fine-tuning | `Train.py` — Unsloth + 4-bit LoRA on Qwen distill |
| Weighted loss on answer tokens | 5× on `FINAL ANSWER` span — teaches format without ignoring reasoning |
| Reproducible inference | `inference.py` loads merged weights |

## Task

Given a prefix of opening brackets, complete with the **minimal closing brackets** for a valid Dyck word.

**Bracket pairs:** `()`, `[]`, `{}`, `<>`

Example output format:

```text
# Thought 1: ...
# Step 1: add ']'.
FINAL ANSWER: ([]){}
```

## Quick start

```bash
pip install -r requirements.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
# Unsloth: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

python generator.py              # conversation.jsonl (default ~60k samples)
python check_dataset_seq_len.py  # set MAX_LENGTH in Train.py from output
python Train.py                  # LoRA → results/ and results_merged/
python inference.py              # edit SEQUENCE in file to probe
```

## Project layout

| File | Role |
|------|------|
| `generator.py` | JSONL with user/assistant turns and reasoning |
| `Train.py` | LoRA fine-tune [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) |
| `inference.py` | Run merged model on custom prefixes |
| `check_dataset_seq_len.py` | Token-length stats for `MAX_LENGTH` |
| `conversation.jsonl` | Generated training data |

## Training defaults

- ~60k samples → ~57k train / ~3k eval
- 2 epochs, LoRA r=64, effective batch 384
- LR 6e-6, warmup 25%, `max_grad_norm=0.5`
- Weighted loss on final-answer tokens (5×)

Outputs: `results/` (adapter), `results_merged/` (full model), `results/training_loss.png`.

## Why Dyck for ML interviews

Stack discipline is **computable ground truth**. You can measure exact match on the sequence, audit reasoning steps, and discuss data scaling without benchmark contamination debates.

## License

MIT © 2026 Akash Dutta
