from data.dataset_generator import generate_dyck_dataset
from data.file_utils import save_dataset_to_file
import random

# Set the seed for reproducibility
random.seed(42)

# Generate and save the dataset
dataset = generate_dyck_dataset(100000)  # Change the number of examples as needed
save_dataset_to_file(dataset)

print("Dataset generation complete.")
