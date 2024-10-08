from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import os

data_template = "train_{id}.jsonl"

def transform_sample(sample, prompt_len=32, continuation_len=32):
    return {
        'prompt': sample['token_ids'][:prompt_len],
        'continuation': sample['token_ids'][prompt_len:prompt_len + continuation_len],
        'source': sample['source']
    }

def load_dataloader_with_id(data_id, args):
    # NOTE: only support Amber ckpts for now
    data_id = str(data_id).zfill(3)
    data_name = data_template.format(id=data_id)
    
    local_path = os.path.join(args.data_path, data_name)
    
    if os.path.exists(local_path) and not args.skip_local:
        # If local version exists & 'skip_local' is set to False
        print(f"Loading local {data_name} from {local_path}")
        dataset = Dataset.from_json(local_path)
    else:
        # Otherwise, load from Huggingface
        dataset = load_dataset(
            'LLM360/AmberDatasets',
            data_files=f'train/{data_name}',
            split='train'
        )

        dataset = dataset.shuffle(seed=args.random_seed)
        dataset = dataset.select(range(1000))
        dataset = dataset.map(
            lambda sample: transform_sample(sample, prompt_len=args.prompt_len, continuation_len=args.continuation_len)
        )
        dataset = dataset.remove_columns(['token_ids'])

        # Store locally
        os.makedirs(args.data_path, exist_ok=True)
        dataset.to_json(local_path)
        print(f"{data_name} saved locally at {local_path}")
    
    dataset.set_format(type='torch', columns=['prompt', 'continuation'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataloader

