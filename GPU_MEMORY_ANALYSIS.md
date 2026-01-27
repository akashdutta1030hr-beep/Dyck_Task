# GPU Memory Requirements Analysis

## Current Configuration

```python
Model: DeepSeek-R1-Distill-Qwen-1.5B (1.5B parameters)
Quantization: 4-bit (load_in_4bit=True)
LoRA: r=32, alpha=64
Batch Size: 8 per device
Gradient Accumulation: 8 steps
Sequence Length: 512 tokens
Precision: bf16 (bfloat16)
Gradient Checkpointing: Enabled
```

## Memory Breakdown

### 1. Base Model (4-bit Quantized)
- **Model Size**: 1.5B parameters
- **4-bit Quantization**: ~0.5 bytes per parameter
- **Memory**: 1.5B × 0.5 bytes = **~750 MB**

### 2. LoRA Adapters
- **Rank (r)**: 32
- **Target Modules**: 7 modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Approximate Size**: ~5-10% of base model
- **Memory**: **~150-300 MB** (depends on hidden size)

### 3. Activations (Forward Pass)
- **Batch Size**: 8
- **Sequence Length**: 512 tokens
- **Hidden Size**: ~1536 (typical for 1.5B model)
- **With Gradient Checkpointing**: Reduces memory by ~50%
- **Memory**: **~2-3 GB** (with checkpointing)

### 4. Gradients (Backward Pass)
- **Only LoRA Parameters**: ~200-300 MB
- **Memory**: **~250 MB**

### 5. Optimizer States (AdamW)
- **AdamW**: 2x parameter memory (momentum + variance)
- **Only for LoRA params**: ~400-600 MB
- **Memory**: **~500 MB**

### 6. System Overhead
- **CUDA overhead**: ~200-500 MB
- **PyTorch overhead**: ~200-300 MB
- **Memory**: **~500 MB**

## Total Memory Estimate

### Minimum (Optimized)
- Base model: 750 MB
- LoRA: 200 MB
- Activations: 2 GB
- Gradients: 250 MB
- Optimizer: 500 MB
- Overhead: 500 MB
- **Total: ~4.2 GB**

### Typical (Realistic)
- Base model: 750 MB
- LoRA: 250 MB
- Activations: 3 GB
- Gradients: 300 MB
- Optimizer: 600 MB
- Overhead: 600 MB
- **Total: ~5.5 GB**

### Maximum (Peak Usage)
- Base model: 800 MB
- LoRA: 300 MB
- Activations: 4 GB
- Gradients: 350 MB
- Optimizer: 700 MB
- Overhead: 700 MB
- **Total: ~6.9 GB**

## Recommended GPU Memory

**Minimum**: **6 GB VRAM** (may be tight)
**Recommended**: **8-12 GB VRAM** (comfortable)
**Optimal**: **16+ GB VRAM** (allows larger batches)

## Memory Optimization Tips

### If You're Running Out of Memory:

1. **Reduce Batch Size**
   ```python
   per_device_train_batch_size=4,  # Reduce from 8
   gradient_accumulation_steps=16, # Increase to maintain effective batch size
   ```

2. **Reduce Sequence Length**
   ```python
   MAX_LENGTH = 256  # Reduce from 512
   ```

3. **Reduce LoRA Rank**
   ```python
   r=16,              # Reduce from 32
   lora_alpha=32,     # Scale with rank
   ```

4. **Use FP16 instead of BF16** (if supported)
   ```python
   fp16=True,  # Instead of bf16=True
   ```

5. **Increase Gradient Checkpointing**
   - Already enabled, but you can verify it's working

### If You Have More Memory:

1. **Increase Batch Size**
   ```python
   per_device_train_batch_size=16,  # Increase from 8
   gradient_accumulation_steps=4,    # Reduce to maintain same effective batch
   ```

2. **Increase Sequence Length**
   ```python
   MAX_LENGTH = 1024  # Increase from 512
   ```

3. **Increase LoRA Rank** (for better learning)
   ```python
   r=64,              # Increase from 32
   lora_alpha=128,     # Scale with rank
   ```

## Current Configuration Analysis

Your current settings:
- **Batch Size**: 8 × 8 = **64 effective batch size**
- **Memory Usage**: **~5-7 GB** (estimated)

This is a **moderate memory usage** configuration. Good for:
- GPUs with 8-12 GB VRAM (RTX 3060, RTX 3070, etc.)
- Training with reasonable speed

## Memory Monitoring

To monitor actual GPU memory usage during training:

```python
# Add this to your training script
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Call before training
print_gpu_memory()
```

Or use `nvidia-smi` in another terminal:
```bash
watch -n 1 nvidia-smi
```

## Quick Reference

| GPU Model | VRAM | Can Run? | Recommended Batch Size |
|-----------|------|----------|----------------------|
| RTX 3060  | 12GB | ✅ Yes   | 8 (current)          |
| RTX 3070  | 8GB  | ⚠️ Tight | 4-6                  |
| RTX 3080  | 10GB | ✅ Yes   | 8-12                 |
| RTX 3090  | 24GB | ✅ Yes   | 16-32                |
| RTX 4090  | 24GB | ✅ Yes   | 16-32                |
| A100      | 40GB | ✅ Yes   | 32+                  |

## Summary

**Your current configuration requires approximately 5-7 GB of GPU memory.**

This should work well on:
- ✅ RTX 3060 (12GB)
- ✅ RTX 3080 (10GB)
- ✅ RTX 3090/4090 (24GB)
- ⚠️ RTX 3070 (8GB) - may need to reduce batch size to 4-6
