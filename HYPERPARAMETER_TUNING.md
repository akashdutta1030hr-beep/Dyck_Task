# Hyperparameter Tuning Guide

## Current Configuration (Updated for Better Performance)

After analyzing the issue, the following hyperparameters have been optimized:

### Training Hyperparameters
- **Epochs**: `5` (increased from 2)
- **Learning Rate**: `2e-4` (increased from 5e-5)
- **Warmup Ratio**: `0.1` (decreased from 0.25)
- **Weight Decay**: `0.01` (decreased from 0.05)
- **Batch Size**: `4 × 4 = 16` effective

### LoRA Hyperparameters
- **Rank (r)**: `32` (increased from 16)
- **Alpha**: `64` (increased from 32)
- **Dropout**: `0.1` (increased from 0.05)

## How to Control Hyperparameters to Improve Performance

### 1. Learning Rate (`learning_rate`)

**What it does**: Controls how much the model updates per step.

**Current**: `2e-4`

**Tuning Guide**:
- **Too High** (> 5e-4): Loss explodes, training unstable
- **Too Low** (< 1e-5): Model learns too slowly, may not converge
- **Optimal Range**: `1e-4` to `3e-4` for fine-tuning

**When to adjust**:
- **Increase** if loss decreases very slowly
- **Decrease** if loss is unstable or oscillating

**Example**:
```python
learning_rate=2e-4,  # Good starting point
# If loss doesn't decrease: try 3e-4
# If unstable: try 1e-4
```

### 2. Number of Epochs (`num_train_epochs`)

**What it does**: How many times the model sees the entire dataset.

**Current**: `5`

**Tuning Guide**:
- **Too Few** (< 3): Model may not learn the format properly
- **Too Many** (> 10): Risk of overfitting
- **Optimal**: `5-7` for small datasets

**When to adjust**:
- **Increase** if model still generates wrong format after training
- **Decrease** if eval loss starts increasing (overfitting)

**Example**:
```python
num_train_epochs=5,  # Good for 10K samples
# For larger datasets: 3-4 epochs may be enough
# For smaller datasets: 7-10 epochs may be needed
```

### 3. Warmup Ratio (`warmup_ratio`)

**What it does**: Gradually increases learning rate at the start of training.

**Current**: `0.1` (10% of training steps)

**Tuning Guide**:
- **Too High** (> 0.3): Wastes training time on low learning rate
- **Too Low** (< 0.05): May cause instability at start
- **Optimal**: `0.1` for most cases

**When to adjust**:
- **Decrease** if you want more training time at full learning rate
- **Increase** if training is unstable at the beginning

### 4. LoRA Rank (`r`)

**What it does**: Controls the capacity of LoRA adapters (higher = more parameters).

**Current**: `32`

**Tuning Guide**:
- **Too Low** (< 16): May not have enough capacity to learn format
- **Too High** (> 64): Risk of overfitting, slower training
- **Optimal**: `16-32` for most tasks, `32-64` for complex format learning

**When to adjust**:
- **Increase** if model struggles to learn the format (try 64)
- **Decrease** if overfitting occurs (try 16)

**Example**:
```python
r=32,           # Current
lora_alpha=64,  # Should be 2× rank
```

### 5. LoRA Alpha (`lora_alpha`)

**What it does**: Scaling factor for LoRA weights.

**Current**: `64`

**Rule**: Should be `2×` the rank value.

**Tuning Guide**:
- Keep ratio: `alpha = 2 × r`
- Higher alpha = stronger LoRA influence

### 6. LoRA Dropout (`lora_dropout`)

**What it does**: Regularization to prevent overfitting.

**Current**: `0.1`

**Tuning Guide**:
- **Too Low** (< 0.05): May overfit
- **Too High** (> 0.2): May underfit
- **Optimal**: `0.05-0.1`

**When to adjust**:
- **Increase** if eval loss >> training loss (overfitting)
- **Decrease** if both losses are high (underfitting)

### 7. Batch Size (`per_device_train_batch_size × gradient_accumulation_steps`)

**What it does**: Effective batch size affects training stability.

**Current**: `4 × 4 = 16` effective

**Tuning Guide**:
- **Larger batch**: More stable, but needs more memory
- **Smaller batch**: Less stable, but uses less memory
- **Optimal**: `16-32` for most cases

**When to adjust**:
- **Increase** if you have more GPU memory (try 8 × 4 = 32)
- **Decrease** if you run out of memory (try 2 × 8 = 16)

### 8. Weight Decay (`weight_decay`)

**What it does**: L2 regularization to prevent overfitting.

**Current**: `0.01`

**Tuning Guide**:
- **Too High** (> 0.1): May prevent learning
- **Too Low** (< 0.001): May overfit
- **Optimal**: `0.01-0.05`

## Performance Monitoring

### Key Metrics to Watch

1. **Training Loss Reduction**
   - Should decrease from ~0.9 to < 0.1
   - If reduction < 0.5: Increase learning rate or epochs

2. **Eval Loss vs Training Loss**
   - Should be similar (within 0.05)
   - If eval >> training: Overfitting (increase dropout, decrease epochs)
   - If both high: Underfitting (increase epochs, learning rate)

3. **Learning Rate Schedule**
   - Should peak early, then decrease smoothly
   - Check `training_metrics.json` after training

### How to Extract All Metrics

After training, all metrics are saved to:
- `results/training_metrics.json` - Detailed metrics
- `results/training_loss.png` - Loss graph
- Console output - Summary metrics

To view metrics:
```bash
python -c "import json; print(json.dump(json.load(open('results/training_metrics.json')), indent=2))"
```

## Recommended Hyperparameter Sets

### For Learning Structured Format (Current Issue)

```python
# Strong fine-tuning to override base model behavior
num_train_epochs=5,
learning_rate=2e-4,
warmup_ratio=0.1,
r=32,
lora_alpha=64,
lora_dropout=0.1,
```

### For Faster Training (Less GPU Time)

```python
num_train_epochs=3,
learning_rate=3e-4,
warmup_ratio=0.05,
r=16,
lora_alpha=32,
lora_dropout=0.05,
```

### For Maximum Quality (More GPU Time)

```python
num_train_epochs=7,
learning_rate=1.5e-4,
warmup_ratio=0.1,
r=64,
lora_alpha=128,
lora_dropout=0.15,
```

## Troubleshooting

### Problem: Model generates wrong format
**Solution**: 
- Increase epochs to 7-10
- Increase learning rate to 3e-4
- Increase LoRA rank to 64

### Problem: Loss not decreasing
**Solution**:
- Increase learning rate
- Check if data preprocessing is correct
- Verify training data format

### Problem: Overfitting (eval loss >> training loss)
**Solution**:
- Increase dropout to 0.15
- Decrease epochs
- Increase weight decay to 0.05

### Problem: Underfitting (both losses high)
**Solution**:
- Increase epochs
- Increase learning rate
- Decrease dropout

## Next Steps

1. **Retrain with updated hyperparameters** (already updated in Train.py)
2. **Monitor training metrics** during training
3. **Check training_metrics.json** after training
4. **Test inference** and verify format
5. **Adjust if needed** based on results
