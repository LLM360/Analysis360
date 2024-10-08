from datasets import Dataset

def load_dataset_from_text(txt_path):
    dataset = Dataset.from_text(txt_path)
    return dataset