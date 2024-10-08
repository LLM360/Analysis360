import torch
import numpy as np

from utils.data_utils import load_dataset_from_text
from utils.model_utils import load_model

import argparse
import random
import os
from tqdm import tqdm
import math
import json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)

    # model & data args
    parser.add_argument('--model_name', type=str, default='LLM360/AmberChat', help='model name or path')
    parser.add_argument('--txt_path', type=str, default='./data/wikitext.txt', help='path to saved txt')
    parser.add_argument('--output_path', type=str, default='.', help='path to save results')

    # ppl args
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for model inference')

    args = parser.parse_args()

    return args

def fix_random(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    args = get_args()
    fix_random(args.random_seed)

    model, tokenizer = load_model(args.model_name)
    model_name = args.model_name.split('/')[-1]

    dataset = load_dataset_from_text(args.txt_path)
    txt_name = args.txt_path.split('/')[-1]

    # calculate ppl

    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(dataset), args.batch_size), desc=f'Caculating ppl for {model_name} on {txt_name}'):
        batch = dataset[i:i+args.batch_size]
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

        mask = inputs['attention_mask']
        total_loss += loss.item() * inputs['input_ids'].size(1)
        total_tokens += mask.sum().item() # 1s are valid tokens, 0s are padding
        

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    result = {
        'total_loss': total_loss,
        'total_tokens': total_tokens,
        'ppl': ppl
    }

    output_path = os.path.join(args.output_path, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f'Results saved to {output_path}')

    print(f'ppl for {model_name} on {txt_name}: {ppl:.4f}')

if __name__ == '__main__':
    main()

