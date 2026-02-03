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
| `train_colab.ipynb` | Training notebook for **Google Colab** (chunked training for time limits). |
| `inference.py` | Loads trained model and runs Dyck completion. |
| `inference.ipynb` | Inference notebook for **Google Colab**. |
| `upload.py` | Pushes the project to GitHub (`python upload.py` or `python upload.py "commit message"`). Set repo URL in `upload.py`. |
| `upload_to_hf.py` | Uploads trained adapter (and optionally merged model) to Hugging Face. See below. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

**GPU (PyTorch):** `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**Unsloth (latest):** `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`

## Dataset

### Generate dataset

```bash
python generator.py
```

Creates `conversation.jsonl`: one JSON array per line `[user_message, assistant_message]` with:
- **`reasoning_content`:** Step-by-step reasoning (e.g., "Opening brackets: [ < ( ...")
- **`content`:** Final answer (e.g., `) > ]`)

**Default:** 60k samples. Edit the loop in `generator.py` (line ~227) to change.

**Check dataset max sequence length (tokens):** Before training, run `python check_dataset_seq_len.py` to see max/mean/p95 token length. Ensure `MAX_LENGTH` in `Train.py` is at least the reported max.

## Training

### Full dataset (A100 80GB, ~6-7h)

```bash
python Train.py
```

**Config (Train.py):**
- **Dataset:** e.g. 60k samples → ~57k train (95%), ~3k eval (5%)
- **GPU:** A100 80GB recommended
- **Epochs:** 2 (configurable; ~296 steps for 60k data)
- **LoRA:** r=64, alpha=128, dropout=0.05
- **Batch:** per_device_train_batch_size=96, gradient_accumulation_steps=4 → effective batch 384
- **Sequence length:** MAX_LENGTH=768 (run `check_dataset_seq_len.py` to confirm)
- **LR:** 6e-6, cosine schedule, warmup 25%, max_grad_norm=0.5
- **Stability:** Conservative LR and grad clip to keep loss low and avoid spikes
- **Weighted loss:** "FINAL ANSWER: " tokens weighted 5× (ANSWER_LOSS_WEIGHT); reasoning tokens 1×

**Outputs:**
- `results/` → LoRA adapter (use with base model)
- `results_merged/` → Merged model (base + LoRA, ready for inference)
- `results/training_loss.png` → Loss graph

**Config notes:**
- `load_best_model_at_end=False`; use the checkpoint with lowest eval_loss from `training_loss.png` if loss spiked.
- `results_merged/` = base + final adapter merged for single-model inference.

**Optional: Upload to Hugging Face**  
1. **Create repo and token:** Go to [huggingface.co/new](https://huggingface.co/new) and create a model repo (e.g. `dyck-1.5b-lora`). Then go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), create a token with **Write** scope, and copy it.  
2. **Upload after training:** Either set `HF_REPO = "your-username/dyck-1.5b-lora"` in `Train.py` (line 17) so training auto-uploads, or run:
   ```bash
   set HF_TOKEN=hf_xxxx
   set HF_REPO=your-username/dyck-1.5b-lora
   python upload_to_hf.py
   ```
   Use `python upload_to_hf.py --help` for options (e.g. `--merged-repo`, `--create-repo`).

### Chunked training (Google Colab, time-limited)

Open `train_colab.ipynb` in Colab, set GPU runtime (T4 or better), and run cells. The notebook trains in 2k chunks, saving/uploading after each chunk.

## Inference

```bash
python inference.py
```

Loads the **merged model** (or HF repo with merged weights) — i.e. base+adapter baked into one model; no separate PEFT load. Prints INPUT and MODEL RESPONSE.

**Response style:** The model is trained to output in **dataset format only**: `# Thought N: ...`, `# Step k: add 'X'.`, then `FINAL ANSWER: <sequence>`. It should not produce Qwen/DeepSeek-style prose ("Wait...", "Let me recount", etc.); the inference prompt enforces dataset style.

**Edit `SEQUENCE` in `inference.py`** to try other inputs.

**Colab:** Open `inference.ipynb`, set GPU runtime, run all cells.

## Model

**Base:** [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B)  
**Fine-tuned:** Adapter in `results/`; merged (base+adapter) in `results_merged/`. Inference uses the merged model (single load). Output style is **dataset-only**, not base-model (Qwen/DeepSeek) style.

## Training Details

- **Task:** Teach the model to generate step-by-step reasoning, then output the closing brackets.
- **Loss:** Computed only on assistant tokens (user prompt is masked).
- **Format:** Model outputs dataset-style reasoning (`# Thought N:`, `# Step k: add 'X'.`) then `FINAL ANSWER: {full_sequence}`. Style is constrained to the dataset format, not Qwen/DeepSeek prose.
- **Evaluation:** Eval set ~5% of data; `eval_loss` logged every 99 steps.
- **Optimization:** MAX_LENGTH=768; weighted loss on "FINAL ANSWER" tokens; LR 6e-6, warmup 25%, max_grad_norm=0.5 for stable training.

## License

See repository license if applicable.
