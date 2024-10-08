import torch

from datasets import load_dataset, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
)

from tqdm import tqdm

def get_text_dataset(topic, random_seed=42):
    '''
    load the raw text dataset for the given topic
    '''
    if topic == 'bio-forget':
        dset_path = 'data/bio_forget.jsonl'
        dset = load_dataset('json', data_files=dset_path, split='train')
        dset = dset.remove_columns(['title', 'abstract', 'doi'])
    elif topic == 'bio-retain':
        dset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split='train')
    elif topic == 'wikitext-test':
        dset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    dset = dset.shuffle(seed=random_seed)
    return dset

def get_text_dataloader(topic, 
                        tokenizer,
                        random_seed=42, 
                        min_len=0, 
                        max_len=2000, 
                        num_samples=500,
                        batch_size=4):
    
    dataset = get_text_dataset(topic, random_seed=random_seed)
    dataset = dataset.filter(lambda x: len(x['text']) > min_len)
    dataset = dataset.select(range(num_samples))
    
    data = {"input_ids": [], "attention_mask": [], "start_locs": []}

    for entry in tqdm(dataset, desc=f"Loading {topic} dataset", total=len(dataset)):
        example = entry['text'].strip()
        if len(example) < min_len:
            continue
        tokens = tokenizer(
            example,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        data["input_ids"].append(tokens["input_ids"])
        data["attention_mask"].append(tokens["attention_mask"])
        data["start_locs"].append(0)
    

    tokenized_dataset = Dataset.from_dict(data)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
    )
