import os
from openai import OpenAI
from instruction_following_eval import get_examples, evaluate_instruction_following
from datasets import Dataset, DatasetDict
import random

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the examples and create dataset_data
def create_dataset_data():
    dataset_data = {"id": [], "messages": [], "variables": []}
    examples = get_examples()
    i = 0
    print(f"Processing {len(examples)} examples")
    
    for example in examples:
        i += 1
        dataset_data["id"].append(example["key"])
        dataset_data["messages"].append([
            {"role": "user", "content": example["prompt"]}, 
            {"role": "assistant", "content": "n/a"}
        ])
        dataset_data["variables"].append(example["kwargs"])
    
    return dataset_data

def convert_to_huggingface_format(dataset_data):
    # Create a list of dictionaries, where each dictionary is a sample
    samples = []
    
    for i in range(len(dataset_data["id"])):
        sample = {
            "id": dataset_data["id"][i],
            "messages": dataset_data["messages"][i],
            "variables": dataset_data["variables"][i]
        }
        samples.append(sample)
    
    # Shuffle the data to ensure a random split
    random.seed(42)  # For reproducibility
    random.shuffle(samples)
    
    # Split the data into train and test sets (50% each)
    split_idx = len(samples) // 2
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_samples)
    test_dataset = Dataset.from_list(test_samples)
    
    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    return dataset_dict

if __name__ == "__main__":
    # Create the dataset data first
    dataset_data = create_dataset_data()
    
    # Then convert to HuggingFace format
    hf_dataset = convert_to_huggingface_format(dataset_data)
    
    # Save the dataset
    hf_dataset.save_to_disk("instruction_following_dataset")
    
    # Optional: Push to HuggingFace Hub (if authenticated)
    # hf_dataset.push_to_hub("username/instruction_following_dataset")
    
    # Print some stats
    print(f"Train set size: {len(hf_dataset['train'])}")
    print(f"Test set size: {len(hf_dataset['test'])}")
    
    # Preview samples
    print("\nTrain sample:")
    print(hf_dataset["train"][0])
    
    print("\nTest sample:")
    print(hf_dataset["test"][0])