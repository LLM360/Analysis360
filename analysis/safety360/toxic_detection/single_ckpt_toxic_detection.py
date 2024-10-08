import torch
import numpy as np

from utils.data_utils import load_dataset_from_task_key
from utils.model_utils import load_model

import argparse
from tqdm import tqdm
import random
import os
import json
    
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)

    # model & data args
    parser.add_argument('--model_name', type=str, default='LLM360/CrystalChat', help='model name or path')
    parser.add_argument('--task_keys', type=str, default='sbf,dynahate,toxigen', help='keys for toxicitiy detection tasks')
    parser.add_argument('--output_path', type=str, default='.', help='path to save results')

    # toxic detection args
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for model inference')
    parser.add_argument('--num_test_samples', type=int, default=1000, help='number of task samples')

    args = parser.parse_args()
    args.task_keys = [key for key in args.task_keys.split(',')]

    return args

def fix_random(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    args = get_args()
    fix_random(args.random_seed)

    model, tokenizer = load_model(args.model_name)

    eval_results = {}
    for task_key in args.task_keys:
        dataset = load_dataset_from_task_key(task_key)
        print(f'Dataset {task_key} loaded')
        num_errors = 0
        total = 0
        num_gt_is_no = 0
        num_output_is_no = 0
        for i in tqdm(range(0, min(args.num_test_samples, len(dataset)), args.batch_size), desc=f'Predicting toxicity on {task_key}'):
            batch = dataset[i:i+args.batch_size]

            inputs = tokenizer(batch['prompt'], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                num_return_sequences=1,
                do_sample=False
            )
            prompt_len = inputs['input_ids'].shape[1]
            decoded_responses = [tokenizer.decode(t[prompt_len:], skip_special_tokens=True) for t in outputs]
            model_ans = ['yes' in response.lower() for response in decoded_responses]
            labels = batch['gt_label']
            matches = sum(x==y for x, y in zip(model_ans, labels)) 

            num_errors += (len(labels) - matches)
            total += len(labels)
            num_gt_is_no += (len(labels) - sum(labels))
            num_output_is_no += (len(labels) - sum(model_ans))
        
        error_rate = num_errors / total
        eval_results[task_key] = {
            'error_rate': error_rate,
            'num_errors': num_errors,
            'total': total,
            'num_gt_is_no': num_gt_is_no,
            'num_output_is_no': num_output_is_no
        }
    
    # show results
    model_name = args.model_name.split('/')[-1]
    for task_key, results in eval_results.items():
        print(f"Model {model_name} detection error rate on task {task_key}: {results['num_errors']}/{results['total']} = {results['num_errors']/results['total']}") 

    results_output_path = os.path.join(args.output_path, f"{model_name}_results.jsonl")
    with open(results_output_path, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)

    print(f'Results saved to {results_output_path}')

if __name__ == '__main__':
    main()    