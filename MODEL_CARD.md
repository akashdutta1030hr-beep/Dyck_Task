---
license: apache-2.0
base_model: unsloth/DeepSeek-R1-Distill-Qwen-1.5B
tags:
  - dyck
  - reasoning
  - brackets
  - fine-tuning
  - lora
  - unsloth
language:
  - en
datasets:
  - conversation.jsonl
pipeline_tag: text-generation
---

# Dyck Completion Model (Reasoning)

This model is fine-tuned to **complete Dyck sequences** (balanced bracket sequences) with **step-by-step reasoning**. Given a prefix of opening brackets, it outputs the minimal closing brackets so the full sequence is a valid Dyck word.

**Response style:** Output follows the **dataset format only** (structured `# Thought N:`, `# Step k: add 'X'.`, then `FINAL ANSWER: <sequence>`). It is not intended to mimic Qwen/DeepSeek-style prose (e.g. no "Wait...", "Let me recount", or conversational commentary). Training and inference prompts enforce this dataset style.

## Task

- **Input:** A prefix of opening brackets (e.g. `[ < (`).
- **Output:** Step-by-step reasoning, then the **complete valid Dyck sequence** (e.g. `) > ]` appended).
- **Bracket pairs:** `()`, `[]`, `{}`, `<>`

## Base Model

- **Architecture:** [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B) (Unsloth)
- **Fine-tuning:** LoRA (r=64, alpha=128, dropout=0.05) on q/k/v/o and MLP projections
- **Training:** Causal LM; loss on assistant tokens only; format: `{reasoning}\n\nFINAL ANSWER: {full_sequence}`

## Intended Use

- Research and education on formal language (Dyck) and chain-of-thought reasoning.
- Benchmarking reasoning models on bracket completion.

## How to Use

**Inference:** Use the **merged model** (single load, base+LoRA already merged) or load base + adapter via PEFT. Merged model = one `AutoModelForCausalLM`; computation is equivalent to base+adapter at every layer.

### With merged model (this repo, if uploaded as merged)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "YOUR_USERNAME/YOUR_REPO"  # e.g. akashdutta1030/dyck-deepseek-r1-lora
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = """Complete the following Dyck language sequence by adding the minimal necessary closing brackets.

Sequence: [ < (

Rules:
- Add only the closing brackets needed to match all unmatched opening brackets
- Response format (dataset style only): Use "# Thought N: ..." for each step, then "# Step k: add 'X'.", then "FINAL ANSWER: " followed by the complete Dyck sequence. Do not add Qwen/DeepSeek-style prose or conversational commentary."""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.05)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# Parse "FINAL ANSWER: ..." from response for the completed sequence
```

### With LoRA adapter (load base + adapter)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length=768,
)
model, tokenizer = FastLanguageModel.from_pretrained(
    "YOUR_USERNAME/YOUR_REPO",  # adapter repo
    max_seq_length=768,
)
# Then generate as above
```

## Training Details

- **Data:** JSONL conversations (user question → assistant reasoning + final answer). Dataset size configurable (e.g. 60k).
- **Split:** ~95% train, ~5% eval.
- **Sequence length:** 768 tokens (run `check_dataset_seq_len.py` to confirm max).
- **Optimization:** AdamW, cosine LR 6e-6, warmup 25%, max_grad_norm=0.5. 2 epochs typical.
- **Weighted loss:** Tokens from "FINAL ANSWER: " onward get weight 5.0; reasoning tokens 1.0 (stronger signal on the answer).

## Limitations

- Trained on synthetic Dyck data; may not generalize to arbitrary bracket-like tasks.
- Performance depends on prefix length and bracket vocabulary.

## Citation

If you use this model, please cite the base model (DeepSeek-R1-Distill-Qwen) and this fine-tuning setup as appropriate.
