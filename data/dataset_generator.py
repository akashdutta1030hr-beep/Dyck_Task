from data.example_generator import generate_dyck_task_example

# Function to generate the full dataset
def generate_dyck_dataset(num_examples =100000):
    dataset = []
    for _ in range(num_examples):
        dataset.append(generate_dyck_task_example())
    
    return dataset
