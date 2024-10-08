import json
from datasets import load_dataset

def load_prompt_datasets(prompt_keys):
    full_dataset = load_dataset("toxigen/toxigen-data", "prompts")
    datasets = []
    for key in prompt_keys:
        datasets.append(full_dataset[key])
    return datasets

def save_prompt_dataset(dataset, output_json_path):
    with open(output_json_path, 'w') as f:
        for row in dataset:
            json.dump(row, f)
            f.write('\n')