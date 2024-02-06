import json
import random
import os
from tqdm import tqdm, trange
from multiprocessing import Pool

prompt_length = 32
continuation_length = 32
data_folder = '/lustre/scratch/shared-folders/llm_project/bowen.tan/workspace/Nov19/llm_training/data'

os.makedirs('sampled_data', exist_ok=True)

pbar = tqdm(total=360)

def process(i):
    random.seed(f'memorization score sample {i}')
    with open(os.path.join(data_folder, f'train_{i}.jsonl')) as f:
        lines = list(f)
    random.shuffle(lines)
    lines = lines[:1000]
    with open(os.path.join('sampled_data', f'{i}.jsonl'), 'w') as f:
        for line in lines:
            line = json.loads(line)
            json.dump({
                'prompt': line['token_ids'][:prompt_length],
                'continuation': line['token_ids'][prompt_length:prompt_length+continuation_length],
                'source': line['source'],
            }, f)
            f.write('\n')

# with Pool(32) as pool:
#     for _ in pool.imap_unordered(process, range(241, 360)):
#         pbar.update(1)

for i in trange(253, 360, 2):
    process(i)