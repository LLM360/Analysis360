import json
import argparse
import evaluate
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    default="",
)
parser.add_argument(
    "--batch_size",
    default=1,
)
parser.add_argument(
    "--output_folder",
    default="",
)
args = parser.parse_args()

ckpt = int(args.model_id.split('/')[-1].replace('ckpt_', ''))
texts = []
types = []
for line in open('test.jsonl').readlines():
    data = json.loads(line)
    if len(data['text']) > 0:
        texts.append(data['text'])
        types.append(data['meta']['pile_set_name'])

perplexity = evaluate.load('my_perplexity.py', module_type='measurement')
results = perplexity.compute(model_id=args.model_id, batch_size=int(args.batch_size), data=texts, device='cuda')
ppls = results['perplexities']

type2ppl = defaultdict(list)
for idx, ppl in enumerate(ppls):
    type2ppl[types[idx]].append(ppl)

final_ppls = []
final_types = []
for key in type2ppl.keys():
    final_types.append(key)
    final_ppls.append(np.mean(type2ppl[key]))
final_types.append('average per doc')
final_ppls.append(np.mean(ppls))

df = pd.DataFrame()
df['type'] = final_types
df['ppl'] = final_ppls
df.to_csv(f'{args.output_folder}/{ckpt}.csv', index=False)
