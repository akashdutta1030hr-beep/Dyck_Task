# Training vs Poor Response — Analysis and Fixes

## What the loss curve actually shows

- **Not classic overfitting.** Train and eval losses *rise together* after ~step 150 and plateau around **0.7**. In overfitting we’d see train loss keep falling while eval rises; here both are high.
- **Brief good region then instability.** Training loss drops to ~0.22 around step 150, then jumps to ~0.77 by step 300. Eval is ~0.25 around steps 100–200, then follows train up to ~0.7. So the model briefly reached a good regime then left it (likely overshooting due to learning rate / schedule).
- **Suspicious last eval point.** The sharp drop in eval loss at ~step 490 is likely an artifact (e.g. final `evaluate()` appended to history, or a single-batch fluke). If `load_best_model_at_end=True` uses that point, we may load the **wrong** checkpoint (end of training, high loss) as “best.”

## Why the model’s response is poor

1. **Wrong “best” checkpoint.** We may be loading either (a) the final checkpoint because of the anomalous low eval at the end, or (b) an early checkpoint (e.g. step 100) when eval was ~0.25 but the model hadn’t learned the task yet. In both cases the deployed model is bad.
2. **Training instability.** The joint rise of train and eval loss after step 150 suggests the optimizer stepped out of a good minimum. That can come from learning rate being too high or cosine decay being too aggressive.
3. **Loss dominated by reasoning tokens.** Most of the target sequence is long reasoning; the actual “FINAL ANSWER: &lt;sequence&gt;” is short. So the model can get reasonable loss by generating plausible-looking reasoning while getting the answer wrong. It never gets a strong learning signal on the exact bracket sequence.
4. **Reasoning in data doesn’t show closing order.** The generator’s reasoning traces only the *prefix* (stack pushes) and then jumps to “Here is the completed Dyck sequence: &lt;full&gt;”. The model never sees the step-by-step “add `)`, then `}`, then `]`…” so it may not learn the correct closing order.

## Changes made in code

### 1. `Train.py`

- **Best-model selection:** `load_best_model_at_end=False` by default so we don’t accidentally pick the anomalous last eval. You can re-enable it after fixing the eval spike or when you’re sure the best step is correct. Prefer using the checkpoint from the step where eval loss was **genuinely** lowest in the middle of training (e.g. around step 150–200 from your plot).
- **Stability (loss spike fix):** Lower LR to `8e-6`, warmup to `0.2`, and `max_grad_norm=0.5` so the optimizer doesn’t overshoot the good minimum around step 200.
- **Eval/plot:** The script that builds the loss graph is unchanged; consider ignoring the very last point when deciding which checkpoint to use for deployment.

### 2. `generator.py`

- **Reinforce the answer in reasoning:** After the stack trace for the prefix, add an explicit line like “Closing brackets in reverse stack order: &lt;closing_sequence&gt;” so the model sees the correct closing order in the reasoning, not only in the final answer.

### 3. `Train.py` (answer-weighted loss)

- **Stronger learning signal on the final answer:** Tokens from "FINAL ANSWER: " onward get loss weight `ANSWER_LOSS_WEIGHT` (e.g. 5.0); reasoning tokens get weight 1.0. So the model gets a much stronger gradient on the bracket sequence and cannot get low loss by only generating plausible reasoning with a wrong answer.
- **Custom `WeightedLossTrainer`** and **`DataCollatorWithWeights`** implement per-token weighted cross-entropy using a `label_weights` field from preprocessing.

### 4. `inference.py`

- **Reproducibility:** Use `temperature=0` (greedy) for deterministic outputs.
- **Response style:** Inference prompt requires **dataset format only** (# Thought N, # Step k: add 'X', FINAL ANSWER: ...). No Qwen/DeepSeek-style prose (e.g. "Wait...", "Let me recount"). The model is trained to match dataset style, not base-model style.

### 5. Merged model vs base+adapter

- **Merged model** (`results_merged/`): Single model load; weights = base + LoRA merged. Forward pass is equivalent to base+adapter at every layer.
- **Adapter only:** Load base then adapter via PEFT for a two-step load; same computation, different storage.
- Training saves both; inference in this repo uses the merged model by default.

After these changes, **regenerate data**, **retrain**, and **deploy the checkpoint from the step with the lowest eval loss** (e.g. before the big rise), not necessarily the last one.
