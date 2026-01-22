import json

# Save the dataset to a file (in JSON format)
def save_dataset_to_file(dataset, filename="dyck_language_with_reasoning_dataset.json"):
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=4)
