from datasets import Dataset
import json

def load_prompt_dataset(prompt_json_path):
    dataset = Dataset.from_json(prompt_json_path)
    return dataset

def save_prompt_dataset(dataset, output_json_path):
    with open(output_json_path, 'w') as f:
        for row in dataset:
            json.dump(row, f)
            f.write('\n')
