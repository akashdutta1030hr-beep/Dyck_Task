import json
from typing import List
from data.generate_dataset import generate_dataset


# ============================================================
# Save dataset
# ============================================================

def save_dataset(dataset: List[dict], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    NUM_SAMPLES = 100000
    OUTPUT_FILE = "dyck_language_with_reasoning_dataset.json"

    print("Generating dataset...")
    samples = generate_dataset(NUM_SAMPLES)

    print("Saving dataset...")
    save_dataset(samples, OUTPUT_FILE)

    print(f"Done. Generated {len(samples)} samples → {OUTPUT_FILE}")
