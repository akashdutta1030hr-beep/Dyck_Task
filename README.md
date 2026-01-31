# Dyck Task

Fine-tune a language model to **complete Dyck sequences** (balanced bracket sequences) with **step-by-step reasoning**.

## Task

Given a prefix of opening brackets, complete it with the **minimal closing brackets** so the full sequence is a valid Dyck word.

**Bracket pairs:** `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`

## Project structure

| File | Description |
|------|-------------|
| `generator.py` | Generates `conversation.jsonl` (user/assistant with reasoning). |
| `Train.py` | Fine-tunes [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) with Unsloth (LoRA, 4-bit). |
| `inference.py` | Loads [akashdutta1030/dyck-deepseek-r1-lora](https://huggingface.co/akashdutta1030/dyck-deepseek-r1-lora) and runs Dyck completion. |
| `inference.ipynb` | Same inference flow for **Google Colab**. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Data

```bash
python generator.py
```

Creates `conversation.jsonl`: one JSON array per line `[user_message, assistant_message]` with `reasoning_content` and `content`. Default 40k lines; edit the loop in `generator.py` to change.

## Training

```bash
python Train.py
```

Uses `conversation.jsonl`, saves LoRA adapter to `results/` and merged model to `results_merged/`. Config: `Train.py` (epochs, batch size, `MAX_LENGTH=2048`, etc.).

## Inference

```bash
python inference.py
```

Loads the Hugging Face model **akashdutta1030/dyck-deepseek-r1-lora** (transformers + PEFT), runs one Dyck completion, prints INPUT SEQUENCE, MODEL OUTPUT, and Result (JSON), and appends to `inference_results.jsonl`.

Edit `sequence` in `inference.py` to try other inputs.

**Colab:** Open `inference.ipynb`, set GPU runtime, run all cells.

## License

See repository license if applicable.
