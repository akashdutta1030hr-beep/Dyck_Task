from data.generate_sample import generate_sample
from typing import List

# Function to generate the entire dataset
def generate_dataset(num_samples: int = 100000) -> List[dict]:
    dataset = []
    for _ in range(num_samples):
        dataset.append(generate_sample())
    return dataset
