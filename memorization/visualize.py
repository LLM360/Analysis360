from matplotlib import pyplot as plt
from glob import glob
import re
import json
from tqdm import tqdm
import numpy as np
from functools import partial

def visualize():
    ckpts = sorted(glob('result_*/'))
    all_data = []
    for ckpt in tqdm(ckpts):
        ckpt = ckpt[7:][:-1]
        files = glob(f'result_{ckpt}/*.json')
        files = sorted(files, key=lambda x: int(re.search(r'result_\d+/(\d+).json', x).group(1)))
        data = []
        for file in tqdm(files):
            with open(file) as f:
                result = json.load(f)
                data.append(result['token_match'])
                all_data.extend(result['match'])
        if ckpt in ['35', '107', '179', '251', '355']:
            plt.plot(data, label=f'ckpt-{ckpt}', linewidth=0.5)
    plt.legend()
    plt.title('Memorization Score')
    plt.xlabel('Sampled Data Partition')
    plt.savefig('token_match.png')
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title('Memoization Score Distribution')
    ax.hist(all_data, density=True, bins=33)
    ax.set_xlabel('Memorization Score * 32')
    # ax.set_yscale('function', functions=(partial(np.power, 10.0), np.log10))
    fig.savefig('token_match_hist.png')

visualize()
