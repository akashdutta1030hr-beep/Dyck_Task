# Why the Model Generates Wrong Format - Explanation

## The Problem

Your model is generating conversational reasoning like:
```
"Alright, so I have this Dyck language sequence to complete..."
"First, I'll analyze the given Dyck language sequence..."
```

Instead of the structured format in your training data:
```
# Thought 1: 1th character is an opening bracket '{', pushing it onto the stack...
# Thought 2: 2th character is an opening bracket '{', pushing it onto the stack...
```

## Root Cause

The base model **`unsloth/DeepSeek-R1-Distill-Qwen-1.5B`** is a **reasoning model** that was pre-trained to generate conversational, natural language reasoning. This is a strong prior that's hard to override with limited fine-tuning.

### Why This Happens:

1. **Base Model Behavior**: DeepSeek-R1 was trained on reasoning tasks with conversational formats
2. **Insufficient Fine-tuning**: The current training (2 epochs, small dataset) isn't strong enough to override the base model's reasoning style
3. **Pattern Matching**: The model sees reasoning patterns and defaults to its pre-trained conversational style rather than the structured "# Thought N:" format

## The Solution

You need to **strengthen the fine-tuning** to override the base model's behavior. Here are the key changes needed:

### 1. Increase Training Intensity

**Current Settings:**
- `num_train_epochs=2` (too few)
- `learning_rate=5e-5` (might be too low)
- `warmup_ratio=0.25` (too much warmup)

**Recommended Changes:**
```python
num_train_epochs=5,           # More epochs
learning_rate=2e-4,           # Higher learning rate
warmup_ratio=0.1,             # Less warmup
```

### 2. Increase Dataset Size

10,000 samples might not be enough. Consider:
- Generating 20,000-50,000 samples
- Or using data augmentation

### 3. Adjust LoRA Parameters

**Current:**
```python
r=16, lora_alpha=32, lora_dropout=0.05
```

**Try:**
```python
r=32,              # Higher rank = more capacity
lora_alpha=64,     # Scale with rank
lora_dropout=0.1,  # More dropout for regularization
```

### 4. Use Different Base Model (Alternative)

If the above doesn't work, consider using a non-reasoning base model:
- `unsloth/Qwen-1.5B` (standard model, not reasoning-focused)
- This would be easier to train on structured formats

## Verification Steps

1. **Check if model was retrained**: Delete `results/` and retrain
2. **Verify training data format**: Ensure preprocessing uses `reasoning_content` correctly
3. **Monitor training loss**: Should decrease significantly (target: < 0.1)
4. **Check eval loss**: Should track training loss closely (no overfitting)

## Expected Training Metrics

After proper training, you should see:
- **Training loss**: Starts ~0.9, ends < 0.1
- **Eval loss**: Similar to training loss (within 0.05)
- **Loss reduction**: > 0.8 (from start to end)

If your loss doesn't decrease much, the model isn't learning the format properly.
