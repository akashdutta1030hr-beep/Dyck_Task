# Dyck Language Reasoning Model

A Python project for generating Dyck-language completion datasets with step-by-step reasoning and fine-tuning language models (e.g. DeepSeek-R1–style) on them.

## Overview

This repo:

1. **Generates** a synthetic dataset of Dyck sequences (properly matched brackets `()`, `[]`, `{}`, `<>`). Each example has **messages** (system, user, assistant) with reasoning in `<think>` tags.
2. **Fine-tunes** a causal LM on `dyck_language_with_reasoning_dataset.json` using `Fine_Tuning.py`.

The dataset uses a **messages** format only (no `prompt` / `response` keys). Prompt and response are derived from `user` and `assistant` message `content` when loading for training.

---

## Project layout

```
Reasoning-Model/
├── README.md
├── config.py                 # Config for dataset generation (and legacy fine-tune settings)
├── generator.py              # Entry point: generates dataset → dyck_language_with_reasoning_dataset.json
├── Fine_Tuning.py            # Fine-tuning script (uses dyck_language_with_reasoning_dataset.json)
├── Fine_Tuning.ipynb         # Notebook version of Fine_Tuning.py
├── run.sh                    # Wrapper to run dataset generation
├── chat_template.py          # (Reserved for chat template utilities)
├── dyck_language_with_reasoning_dataset.json   # Generated/training dataset (JSON)
└── data/
    ├── __init__.py
    ├── dataset_generator.py  # Orchestrates generation of N examples
    ├── example_generator.py  # Single example: messages [system, user, assistant]
    ├── dyck_completion.py    # Stack-based completion + reasoning steps
    └── file_utils.py         # save_dataset_to_file() → JSON
```

---

## Dataset format

**File:** `dyck_language_with_reasoning_dataset.json`

- Single JSON **array** of objects.
- Each object has **only** `"messages"` (no `"prompt"` or `"response"`).
- Each `messages` item has `"role"` and `"content"`.

Example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that completes partial Dyck sequences with valid matching brackets, explaining the reasoning behind each step using the <think> tag."
    },
    {
      "role": "user",
      "content": "Complete this Dyck sequence: ({[[["
    },
    {
      "role": "assistant",
      "content": "<|Assistant|><think>\nAdded ( to the stack.\n...\n</think>Here is the completed Dyck sequence: ({[[[]]]})"
    }
  ]
}
```

- **Prompt** (for training) = `user` message `content`.
- **Response** (for training) = `assistant` message `content`. It may include `<|Assistant|>`, `<think>...</think>`, and the final completed sequence.

---

## 1. Dataset generation

### Dependencies

- Python 3.x
- Standard library only (no extra packages for generation)

### Usage

```bash
python generator.py
```

This:

- Calls `data.dataset_generator.generate_dyck_dataset(100_000)`
- Saves via `data.file_utils.save_dataset_to_file()` to **`dyck_language_with_reasoning_dataset.json`** (in the project root).

### Configuration

- **`config.py`**: `N_SAMPLES`, `MIN_LEN`, `MAX_LEN`, `OUTPUT_PATH`, `SEED`, etc.  
- **`generator.py`**: You can change the `generate_dyck_dataset(...)` sample count.
- **`data/file_utils.py`**: Default output path is `dyck_language_with_reasoning_dataset.json`.

### What the generator does

- **`data/example_generator.py`**: Builds one example with `messages` [system, user, assistant]. User asks to complete a partial Dyck sequence; assistant responds with reasoning in `<think>` and the completed sequence.
- **`data/dyck_completion.py`**: Implements stack-based completion and produces the reasoning steps.
- **`data/dataset_generator.py`**: Loops `generate_dyck_task_example()` and collects the list.
- **`data/file_utils.py`**: Writes the list to JSON (indented).

### Run script

```bash
bash run.sh
```

`run.sh` runs the dataset generation (via `generator.py`). It uses `python3` or `python` if `PYTHON` is not set.

---

## 2. Fine-tuning

### Script: `Fine_Tuning.py`

**Training data:** `dyck_language_with_reasoning_dataset.json` (same directory as the script).

The script:

- Loads the JSON and **derives** `prompt` / `response` from `messages` (user → prompt, assistant → response).
- Builds a HuggingFace `Dataset` with `prompt` and `response`.
- Tokenizes with format: `"<|User|>\n{prompt}\n{response}{eos}"`.  
  The dataset’s assistant `content` already starts with `<|Assistant|><think>...`, so the script does **not** add an extra `<|Assistant|>` before the response.
- Fine-tunes **`unsloth/DeepSeek-R1-Distill-Qwen-1.5B`** (configurable at top of script).

### Dependencies

```bash
pip install transformers datasets torch
```

### Usage

```bash
python Fine_Tuning.py
```

Ensure `dyck_language_with_reasoning_dataset.json` exists in the project root (or adjust `TRAIN_FILE` / paths in the script).

### Main configuration (in `Fine_Tuning.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `unsloth/DeepSeek-R1-Distill-Qwen-1.5B` | Base model |
| `TRAIN_FILE` | `dyck_language_with_reasoning_dataset.json` | Training data (path relative to script) |
| `MAX_LENGTH` | 512 | Max token length |
| `output_dir` | `./results` | Checkpoint directory |

Training uses bf16, gradient checkpointing, cosine LR, and the `Trainer` API. Details are in the script.

### Notebook

`Fine_Tuning.ipynb` mirrors `Fine_Tuning.py` for interactive use.

---

## 3. Bracket types

The Dyck setup uses four bracket pairs:

- `()` `[]` `{}` `<>`

---

## 4. `config.py` (optional / legacy)

`config.py` holds options for dataset generation and, historically, for another fine-tune setup. **`Fine_Tuning.py` does not use `config.py`**; it defines its own config. **`generator.py`** also does not import `config`; it uses hardcoded defaults (e.g. 100k samples) and `data.file_utils` default output path. You can use `config.py` for reference or your own scripts.

---

## 5. Quick start

1. **Generate dataset**
   ```bash
   python generator.py
   ```
   → creates/overwrites `dyck_language_with_reasoning_dataset.json`.

2. **Fine-tune**
   ```bash
   pip install transformers datasets torch
   python Fine_Tuning.py
   ```
   → reads `dyck_language_with_reasoning_dataset.json`, trains, writes checkpoints to `./results`.

---

## Repository

**GitHub:** [https://github.com/akashdutta1030hr-beep/Reasoning-Model](https://github.com/akashdutta1030hr-beep/Reasoning-Model)

### Pushing changes

Commit and push from your machine (authenticate with GitHub via SSH or a personal access token):

```bash
git add .
git commit -m "Your message"
git push -u origin update-dataset   # or main
```

**Note:** `dyck_language_with_reasoning_dataset.json` is in `.gitignore` (>100MB); generate it locally with `python generator.py` or host it elsewhere (e.g. Hugging Face Datasets).

---

## License

This project is provided as-is for research and education.
