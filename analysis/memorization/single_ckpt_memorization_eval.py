import torch
import numpy as np

from analysis.memorization.utils.data_utils import load_dataloader_with_id
from utils.model_utils import load_model

import argparse
import random
from tqdm import tqdm
import json
import os

def fix_random(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)

    # model & data args
    parser.add_argument('--ckpt_id', type=int, default=355, help='the checkpoint id')
    parser.add_argument('--data_id', type=int, default=355, help='the pretraining data chunk id')
    parser.add_argument('--data_path', type=str, default='data/', help='path to saved pretrain data chunks')
    parser.add_argument('--skip_local', action='store_true', help='skip local data chunk if exists')
    
    # memorization args
    parser.add_argument('--prompt_len', type=int, default=32, help='prompt length (k) for memorization experiment')
    parser.add_argument('--continuation_len', type=int, default=32, help='continuation length (l) for memorization experiment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    fix_random(args.random_seed)

    model = load_model(args.ckpt_id)
    dataloader = load_dataloader_with_id(args.data_id, args) 

    total_token_match = []
    all_match = []
    count = 0

    for batch in tqdm(dataloader, desc=f'Evaluating memorization of ckpt_{args.ckpt_id} on train_{args.data_id}'):
        prompt = batch['prompt'].cuda()
        continuation = batch['continuation'].cuda()

        generated = model.generate(prompt, min_length=args.prompt_len, max_new_tokens=args.continuation_len, num_beams=1, do_sample=False)[:, args.prompt_len:]
        
        total_token_match += (generated == continuation).sum(dim=1).tolist()
        all_match += (generated == continuation).int().tolist()
        count += prompt.size(0)
    
    full_match = sum(1 for x in total_token_match if x == args.continuation_len) / count
    token_match = sum(total_token_match) / args.continuation_len / count
    print(f'Full Match %: {full_match}')
    print(f'Memorization Score: {token_match}')

    save_path = f'result_ckpt-{args.ckpt_id}/data-{args.data_id}.json'
    os.makedirs(f'result_ckpt-{args.ckpt_id}', exist_ok=True)
    with open(save_path, 'w') as g:
        json.dump({
            'full_match': full_match,
            'token_match': token_match,
            'count': count,
            'match': total_token_match,
            'all_match': all_match,
        }, g)
    print(f'Results saved to {save_path}')


if __name__ == '__main__':
    main()