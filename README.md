# Dyck Task

Fine-tune a language model to **complete Dyck sequences** (balanced bracket sequences) with **step-by-step reasoning**. The model learns to output reasoning followed by a final answer in the form `FINAL ANSWER: <closing brackets>`.

## Task

Given a prefix of opening (and possibly some closing) brackets, complete it with the **minimal necessary closing brackets** so the full sequence is a valid Dyck word.

**Validation rules:**

1. **Balanced** – Every opening bracket has a matching closing bracket.
2. **Proper nesting** – Brackets are properly nested (no crossing).
3. **Unique completion** – Only one valid way to complete the sequence (LIFO).

**Scoring:**

- **Correct: 1.0** – Exact match of the complete valid sequence.
- **Incorrect: 0.0** – Any other output.

**Bracket pairs:** `()`, `[]`, `{}`, `<>`, `⟨⟩`, `⟦⟧`, `⦃⦄`, `⦅⦆`

## Project structure

| File | Description |
|------|--------------|
| `generator.py` | Generates Dyck tasks and writes `conversation.jsonl` (user/assistant with reasoning). |
| `Train.py` | Fine-tunes [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) with Unsloth (LoRA, 4-bit). |
| `Train.ipynb` | Same training flow for **Google Colab**. |
| `inference.py` | Loads the fine-tuned model and runs inference on a Dyck prompt. |
| `inference.ipynb` | Same inference flow for **Google Colab**. |
| `verifier.py` | Verifies model output vs ground truth (extract answer, compare, score 1.0/0.0). |
| `requirements.txt` | Python dependencies. |

## Setup

**Requirements:** Python 3.10+, GPU with enough VRAM for 1.5B model (4-bit LoRA).

```bash
pip install -r requirements.txt
```

For CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Data

Generate the training dataset (10k samples by default):

```bash
python generator.py
```

This creates `conversation.jsonl`: each line is a JSON array `[user_message, assistant_message]` where the assistant has `reasoning_content` (step-by-step) and `content` (final Dyck completion). The training script formats this as `{reasoning}\n\nFINAL ANSWER: {answer}`.

**Note:** `conversation.jsonl` is large (~147 MB). It is in `.gitignore`; generate it locally or use Git LFS if you need it in the repo.

## Training

**Script:**

```bash
python Train.py
```

Config in `Train.py`: `DATA_PATH="conversation.jsonl"`, `OUTPUT_DIR="results"`, base model `unsloth/DeepSeek-R1-Distill-Qwen-1.5B`, LoRA r=32, 2 epochs, etc. The script saves the model and tokenizer to `results/` and writes a training loss plot to `results/training_loss.png`.

**Colab:** Open `Train.ipynb`, upload `conversation.jsonl` (or set `DATA_PATH` to your Drive path), enable GPU, and run all cells.

## Inference

**Script:**

Edit `sequence` in `inference.py` if needed, then:

```bash
python inference.py
```

This loads the model from `results/`, runs one Dyck completion, prints the response, and appends a JSON line to `inference_results.jsonl` with `sequence` and `response`.

**Colab:** Open `inference.ipynb`, set `MODEL_PATH` to your saved model (e.g. `results/` or Drive path), and run all cells.

## Verifier

Use `verifier.py` to score model output against ground truth:

```python
from verifier import verify, extract_answer, is_valid_dyck, get_required_closing

# Score: 1.0 if exact match, 0.0 otherwise
score = verify(
    model_output="...",           # raw model text
    correct_full_sequence="...", # full Dyck sequence (prefix + closing)
    question_prefix="...",       # optional; if model outputs only closing part
)
# score is 1.0 or 0.0

# Helpers
extracted = extract_answer(model_output)  # after "FINAL ANSWER:" or longest valid bracket sequence
valid = is_valid_dyck(sequence)
closing = get_required_closing(prefix)
```

Run built-in checks:

```bash
python verifier.py
```

## License

See repository license if applicable.
