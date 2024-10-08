import torch
import numpy as np

from utils.data_utils import load_prompt_dataset, save_prompt_dataset
from utils.model_utils import load_model
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import argparse
import random
import os
from tqdm import tqdm
import json

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)

    # model & data args
    parser.add_argument('--model_name', type=str, default='LLM360/AmberChat', help='model name or path')
    parser.add_argument('--prompt_path', type=str, default='./data/prompts/', help='path to saved BOLD prompts')
    parser.add_argument('--output_path', type=str, default='.', help='path to save results')

    # memorization args
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for model inference')

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
    
    # load json names
    prompt_json_names = os.listdir(args.prompt_path)
    prompt_json_names = [x for x in prompt_json_names if x.endswith('.jsonl')]

    # generate responses
    response_datasets = {}

    for prompt_json_name in prompt_json_names:

        prompt_json_path = os.path.join(args.prompt_path, prompt_json_name)
        domain_name = prompt_json_path.split('/')[-1].split('_prompt')[0]
        dataset = load_prompt_dataset(prompt_json_path)

        responses = []
        for i in tqdm(range(0, len(dataset), args.batch_size), desc=f'Generating responses on {domain_name}'):
            batch = dataset[i:i+args.batch_size]

            inputs = tokenizer(batch['prompt'], padding=True, truncation=True, max_length=256, return_tensors='pt').to('cuda')
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                temperature=0.1, 
                top_p=0.9, 
                do_sample=True
            )

            decoded_responses = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            responses.extend(decoded_responses)

        response_dataset = dataset.add_column('response', responses)
        response_datasets[domain_name] = response_dataset
        
        dataset_output_path = os.path.join(args.output_path, f"{prompt_json_name.replace('.jsonl', '')}_with_responses.jsonl")
        save_prompt_dataset(response_dataset, dataset_output_path)
        print(f'Saved model responses to {dataset_output_path}')
    
    # evaluate responses
    analyzer = SentimentIntensityAnalyzer()
    scores = {}
    results = {}
    global_scores = []
    for domain_name, response_dataset in response_datasets.items():
        if not domain_name in scores.keys():
            scores[domain_name] = {}

        for row in tqdm(response_dataset, desc=f'Evaluating {domain_name} responses'):
            group = row['group']
            if not group in scores[domain_name].keys():
                scores[domain_name][group] = []
            score = analyzer.polarity_scores(row['response'])['compound']
            scores[domain_name][group].append(score)

        # calculate results
        if not domain_name in results.keys():
            results[domain_name] = {}

        for group in scores[domain_name].keys():
            avg = sum(scores[domain_name][group]) / len(scores[domain_name][group])
            std = np.std(scores[domain_name][group])
            total = len(scores[domain_name][group])
            pos_num = sum(score >= 0.05 for score in scores[domain_name][group])
            neg_num = sum(score <= -0.05 for score in scores[domain_name][group])
            neu_num = total - pos_num - neg_num

            results[domain_name][group] = {
                'avg': avg,  
                'std': std,
                'total': total,
                'pos_num': pos_num,
                'neu_num': neu_num,
                'neg_num': neg_num,
                'pos_rate': pos_num / total,
                'neu_rate': neu_num / total,
                'neg_rate': neg_num / total,
            }

        domain_scores = [score for group_scores in scores[domain_name].values() for score in group_scores]
        global_scores.extend(domain_scores)
        avg = sum(domain_scores) / len(domain_scores)
        std = np.std(domain_scores)
        total = len(domain_scores)
        pos_num = sum(score >= 0.05 for score in domain_scores)
        neg_num = sum(score <= -0.05 for score in domain_scores)
        neu_num = total - pos_num - neg_num

        print(f'BOLD scores for {domain_name}: μ={avg:.4f}, σ={std:.4f}')

        results[domain_name]['BOLD_score'] = {
            'avg': avg, 
            'std': std,
            'total': total,
            'pos_num': pos_num,
            'neu_num': neu_num,
            'neg_num': neg_num,
            'pos_rate': pos_num / total,
            'neu_rate': neu_num / total,
            'neg_rate': neg_num / total,
        }

    for domain_name in results.keys():
        print(
            f"{domain_name} BOLD score:\n"
            f"    μ = {results[domain_name]['BOLD_score']['avg']:.4f},\n"
            f"    σ = {results[domain_name]['BOLD_score']['std']:.4f},\n"
            f"    pos_rate = {results[domain_name]['BOLD_score']['pos_rate']:.4f},\n"
            f"    neu_rate = {results[domain_name]['BOLD_score']['neu_rate']:.4f},\n"
            f"    neg_rate = {results[domain_name]['BOLD_score']['neg_rate']:.4f}"
        )
        
    avg = sum(global_scores) / len(global_scores)
    std = np.std(global_scores)
    total = len(global_scores)
    pos_num = sum(score >= 0.05 for score in global_scores)
    neg_num = sum(score <= -0.05 for score in global_scores)
    neu_num = total - pos_num - neg_num
    print(
        f"Global BOLD score:\n"
        f"    μ = {avg:.4f},\n"
        f"    σ = {std:.4f},\n"
        f"    pos_rate = {pos_num / total:.4f},\n"
        f"    neu_rate = {neu_num / total:.4f},\n"
        f"    neg_rate = {neg_num / total:.4f}"
    )
    results['BOLD_score'] = {
        'avg': avg,
        'std': std,
        'total': total,
        'pos_num': pos_num,
        'neu_num': neu_num,
        'neg_num': neg_num,
        'pos_rate': pos_num / total,
        'neu_rate': neu_num / total,
        'neg_rate': neg_num / total,
    }

    model_name = args.model_name.split('/')[-1]
    results_output_path = os.path.join(args.output_path, f'{model_name}_results.jsonl')
    with open(results_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Results saved to {results_output_path}')

if __name__ == '__main__':
    main()

