"""
Check training status and available models/checkpoints.
Run this to see what's ready for merging and upload.
"""
import os
from pathlib import Path

RESULTS_DIR = Path("results")

print("=" * 70)
print("TRAINING STATUS CHECK")
print("=" * 70)

if not RESULTS_DIR.exists():
    print("❌ No results/ folder found. Training hasn't started or failed.")
    exit(1)

print(f"✓ Results folder exists: {RESULTS_DIR}\n")

# Check for final trained model (adapter at root of results/)
adapter_files = ["adapter_model.safetensors", "adapter_config.json"]
has_final_adapter = all((RESULTS_DIR / f).exists() for f in adapter_files)

if has_final_adapter:
    print("=" * 70)
    print("✓ FINAL TRAINED MODEL FOUND")
    print("=" * 70)
    print("Files:")
    for f in adapter_files:
        size_mb = (RESULTS_DIR / f).stat().st_size / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    print("\nRecommendation: Use this for upload!")
    print("Set in merge_checkpoint_to_hf.py:")
    print('  CHECKPOINT_DIR = "results"')
    print()
else:
    print("=" * 70)
    print("⚠ FINAL TRAINED MODEL NOT FOUND")
    print("=" * 70)
    print("Training may still be in progress or only saved checkpoints.")
    print()

# Check for checkpoints
checkpoints = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

if checkpoints:
    print("=" * 70)
    print(f"CHECKPOINTS FOUND: {len(checkpoints)}")
    print("=" * 70)
    for cp in checkpoints:
        step = cp.name.split("-")[1]
        has_adapter = (cp / "adapter_model.safetensors").exists()
        status = "✓ Has adapter" if has_adapter else "✗ No adapter"
        
        # Try to estimate progress
        # Assuming ~2376 total steps for 60k dataset, 4 epochs
        try:
            step_num = int(step)
            progress = (step_num / 2376) * 100
            print(f"  - {cp.name}: {status} (~{progress:.1f}% of training)")
        except:
            print(f"  - {cp.name}: {status}")
    
    print("\n⚠ Checkpoints are incomplete models.")
    if len(checkpoints) > 0:
        last_cp = checkpoints[-1].name
        print(f"Latest: {last_cp}")
        print(f'To test, set in merge_checkpoint_to_hf.py: CHECKPOINT_DIR = "results/{last_cp}"')
else:
    print("=" * 70)
    print("⚠ NO CHECKPOINTS FOUND")
    print("=" * 70)
    print("Training may have just started or checkpoint saving is disabled.")

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
if has_final_adapter:
    print("✓ Use final model: CHECKPOINT_DIR = \"results\"")
elif checkpoints:
    last_cp = checkpoints[-1].name
    step = int(last_cp.split("-")[1])
    if step < 500:
        print(f"⚠ Latest checkpoint ({last_cp}) is too early.")
        print("Wait for training to complete before merging.")
    else:
        print(f"✓ Test with: CHECKPOINT_DIR = \"results/{last_cp}\"")
        print("  (Still incomplete, but better than earlier checkpoints)")
else:
    print("⚠ Wait for training to complete.")

print("=" * 70)
