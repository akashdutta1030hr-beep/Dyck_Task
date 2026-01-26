# Complete Training Arguments Guide

## Default Training Arguments by Fine-Tuning Method

### 1. Full Fine-Tuning (All Parameters)

**Use case:** Training all model parameters (requires most memory)

**Default Configuration:**
```python
TrainingArguments(
    per_device_train_batch_size=2-4,      # Smaller batch (memory intensive)
    gradient_accumulation_steps=4-8,      # Increase effective batch
    learning_rate=1e-5 to 5e-5,          # Lower LR (conservative)
    num_train_epochs=3-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    save_total_limit=3,
    load_best_model_at_end=True,
)
```

**Characteristics:**
- Lower learning rate (1e-5 to 5e-5)
- Smaller batch sizes (memory constraint)
- More epochs may be needed
- Highest memory usage

---

### 2. LoRA Fine-Tuning (Low-Rank Adaptation)

**Use case:** Efficient fine-tuning with trainable adapters (your current method)

**Default Configuration:**
```python
TrainingArguments(
    per_device_train_batch_size=4-8,      # Larger batch possible
    gradient_accumulation_steps=2-4,       # Moderate accumulation
    learning_rate=1e-4 to 5e-4,           # Higher LR (10x full fine-tuning)
    num_train_epochs=3-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    save_total_limit=3,
    load_best_model_at_end=True,
)
```

**Characteristics:**
- Higher learning rate (1e-4 to 5e-4)
- Can use larger batch sizes
- Faster training
- Lower memory usage

---

### 3. QLoRA Fine-Tuning (Quantized LoRA)

**Use case:** Maximum memory efficiency (4-bit quantization + LoRA)

**Default Configuration:**
```python
TrainingArguments(
    per_device_train_batch_size=4-16,     # Can be larger (quantized model)
    gradient_accumulation_steps=1-4,
    learning_rate=1e-4 to 5e-4,           # Similar to LoRA
    num_train_epochs=3-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,  # or bf16=True
    max_grad_norm=1.0,
    save_total_limit=3,
    load_best_model_at_end=True,
)
```

**Characteristics:**
- Similar to LoRA but with quantization
- Can use even larger batch sizes
- Lowest memory usage
- Slightly slower than LoRA (quantization overhead)

---

### 4. Adapter Fine-Tuning

**Default Configuration:**
```python
TrainingArguments(
    per_device_train_batch_size=4-8,
    gradient_accumulation_steps=2-4,
    learning_rate=1e-4 to 3e-4,           # Similar to LoRA
    num_train_epochs=3-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
)
```

---

## How to Control Training Arguments to Improve Loss

### Problem 1: Training Loss Not Decreasing

**Symptoms:**
- Training loss stays high or increases
- Model not learning

**Solutions:**

#### A. Learning Rate Too High
```python
# Current: learning_rate=2e-4
# Try: Lower the learning rate
learning_rate=1e-4,  # Reduce by 2x
# Or even: learning_rate=5e-5  # Reduce by 4x
```

#### B. Learning Rate Too Low
```python
# Current: learning_rate=2e-4
# Try: Increase the learning rate
learning_rate=3e-4,  # Increase by 1.5x
# Or: learning_rate=5e-4  # Increase by 2.5x
```

#### C. Batch Size Too Small
```python
# Current: per_device_train_batch_size=4
# Try: Increase batch size (if memory allows)
per_device_train_batch_size=8,
gradient_accumulation_steps=2,  # Adjust to maintain effective batch
```

#### D. Need More Training
```python
# Current: num_train_epochs=5
# Try: Train for more epochs
num_train_epochs=10,  # Double the training
```

---

### Problem 2: Training Loss Decreasing but Eval Loss Increasing (Overfitting)

**Symptoms:**
- Training loss: 2.5 → 1.0 → 0.5 (decreasing)
- Eval loss: 2.5 → 1.5 → 2.0 (increasing after initial decrease)

**Solutions:**

#### A. Increase Regularization
```python
# Current: weight_decay=0.01
# Try: Increase weight decay
weight_decay=0.1,  # 10x stronger regularization
```

#### B. Reduce Training
```python
# Current: num_train_epochs=5
# Try: Train for fewer epochs
num_train_epochs=3,  # Stop earlier
# Use early stopping based on eval_loss
```

#### C. Increase Dropout (if using LoRA)
```python
# In LoRA configuration:
lora_dropout=0.1,  # Increase from 0.05 to 0.1
```

#### D. More Frequent Evaluation
```python
# Current: eval_steps=250
# Try: Evaluate more frequently to catch overfitting early
eval_steps=100,  # Check every 100 steps
```

---

### Problem 3: Both Training and Eval Loss Not Decreasing

**Symptoms:**
- Training loss: 2.5 → 2.4 → 2.5 (stuck)
- Eval loss: 2.5 → 2.4 → 2.5 (stuck)

**Solutions:**

#### A. Learning Rate Too Low
```python
# Try: Increase learning rate
learning_rate=5e-4,  # Increase from 2e-4
```

#### B. Need More Data or Better Data
- Check data quality
- Increase dataset size if possible
- Check data preprocessing

#### C. Model Capacity
- Model might be too small for the task
- Consider using a larger base model

#### D. Learning Rate Schedule
```python
# Try: Different scheduler
lr_scheduler_type="linear",  # Instead of "cosine"
# Or: lr_scheduler_type="constant_with_warmup"
```

---

### Problem 4: Loss Decreasing Too Slowly

**Symptoms:**
- Loss decreases but very slowly
- Training takes too long

**Solutions:**

#### A. Increase Learning Rate
```python
learning_rate=3e-4,  # Increase from 2e-4
```

#### B. Increase Batch Size
```python
per_device_train_batch_size=8,  # Increase from 4
gradient_accumulation_steps=2,   # Adjust accordingly
```

#### C. Reduce Warmup
```python
warmup_ratio=0.05,  # Reduce from 0.1 (faster start)
```

---

### Problem 5: Loss Oscillating (Unstable Training)

**Symptoms:**
- Loss jumps up and down
- Training is unstable

**Solutions:**

#### A. Reduce Learning Rate
```python
learning_rate=1e-4,  # Reduce from 2e-4
```

#### B. Increase Batch Size
```python
per_device_train_batch_size=8,
gradient_accumulation_steps=4,  # Effective batch = 32
```

#### C. Stronger Gradient Clipping
```python
max_grad_norm=0.5,  # More aggressive clipping (from 1.0)
```

---

## Comprehensive Adjustment Guide

### To Reduce Training Loss:

1. **Increase Learning Rate** (if too low)
   ```python
   learning_rate=3e-4  # from 2e-4
   ```

2. **Increase Batch Size** (more stable gradients)
   ```python
   per_device_train_batch_size=8  # from 4
   ```

3. **Train Longer**
   ```python
   num_train_epochs=10  # from 5
   ```

4. **Reduce Warmup** (start learning faster)
   ```python
   warmup_ratio=0.05  # from 0.1
   ```

---

### To Reduce Eval Loss (Improve Generalization):

1. **Increase Regularization**
   ```python
   weight_decay=0.1  # from 0.01
   ```

2. **Early Stopping** (prevent overfitting)
   ```python
   # Monitor eval_loss, stop when it starts increasing
   evaluation_strategy="steps",
   eval_steps=100,  # More frequent evaluation
   load_best_model_at_end=True,  # Already set ✓
   ```

3. **Reduce Training Epochs**
   ```python
   num_train_epochs=3  # from 5
   ```

4. **Increase Dropout** (in LoRA config)
   ```python
   lora_dropout=0.1  # from 0.05
   ```

---

### To Prevent Overfitting:

1. **Stronger Regularization**
   ```python
   weight_decay=0.1,  # Increase from 0.01
   ```

2. **More Frequent Evaluation**
   ```python
   eval_steps=100,  # Evaluate more often
   ```

3. **Early Stopping**
   - Monitor eval_loss
   - Stop training when eval_loss stops decreasing

4. **Data Augmentation** (if applicable)
   - Increase dataset diversity

---

### To Speed Up Training:

1. **Increase Batch Size** (if memory allows)
   ```python
   per_device_train_batch_size=8,
   gradient_accumulation_steps=2,  # Maintain effective batch
   ```

2. **Reduce Evaluation Frequency**
   ```python
   eval_steps=500,  # Less frequent evaluation
   ```

3. **Use Mixed Precision**
   ```python
   fp16=True,  # Already set ✓
   # Or bf16=True if available
   ```

---

## Quick Reference: Default Values by Method

| Method | Learning Rate | Batch Size | Epochs | Weight Decay |
|--------|---------------|------------|--------|--------------|
| **Full Fine-tuning** | 1e-5 to 5e-5 | 2-4 | 3-5 | 0.01 |
| **LoRA** | 1e-4 to 5e-4 | 4-8 | 3-5 | 0.01 |
| **QLoRA** | 1e-4 to 5e-4 | 4-16 | 3-5 | 0.01 |
| **Adapter** | 1e-4 to 3e-4 | 4-8 | 3-5 | 0.01 |

---

## Monitoring and Adjusting During Training

### Watch These Metrics:

1. **Training Loss**
   - Should decrease smoothly
   - If not decreasing: increase LR or batch size
   - If decreasing too fast: might overfit

2. **Eval Loss**
   - Should decrease with training loss
   - If increasing while train loss decreases: overfitting
   - If not decreasing: model not learning

3. **Gap Between Train and Eval Loss**
   - Small gap (< 0.2): Good generalization
   - Large gap (> 0.5): Overfitting

### Adjustment Strategy:

```python
# If eval_loss > train_loss by a lot (overfitting):
weight_decay = 0.1  # Increase regularization
num_train_epochs = 3  # Train less
eval_steps = 100  # Monitor more closely

# If both losses not decreasing (not learning):
learning_rate = 5e-4  # Increase LR
per_device_train_batch_size = 8  # Increase batch

# If training unstable (loss oscillating):
learning_rate = 1e-4  # Decrease LR
max_grad_norm = 0.5  # Stronger clipping
per_device_train_batch_size = 8  # Larger batch
```

---

## Your Current Configuration Analysis

**Your setup (LoRA fine-tuning):**
```python
learning_rate=2e-4,           # ✓ Good (within LoRA range 1e-4 to 5e-4)
per_device_train_batch_size=4, # ✓ Good (can increase if memory allows)
gradient_accumulation_steps=4, # ✓ Good (effective batch = 16)
num_train_epochs=5,            # ✓ Good for small dataset
weight_decay=0.01,             # ✓ Standard
max_grad_norm=1.0,             # ✓ Standard
```

**This is a well-balanced configuration for your use case!**

---

## Summary

**Default Training Arguments:**
- **Full fine-tuning**: Lower LR (1e-5 to 5e-5), smaller batches
- **LoRA/QLoRA**: Higher LR (1e-4 to 5e-4), larger batches possible
- **All methods**: weight_decay=0.01, max_grad_norm=1.0 are standard

**To Improve Loss:**
- **Training loss not decreasing**: Increase LR or batch size
- **Eval loss increasing (overfitting)**: Increase weight_decay, reduce epochs
- **Both not decreasing**: Increase LR, check data quality
- **Loss oscillating**: Decrease LR, increase batch size

**Key Principle:**
- Monitor both train and eval loss
- Adjust based on what you observe
- Start with defaults, then tune based on results
