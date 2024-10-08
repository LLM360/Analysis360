import torch
import numpy as np

from utils.data_utils import load_prompt_datasets, save_prompt_dataset
from utils.model_utils import load_model
from transformers import pipeline

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
    parser.add_argument('--prompt_keys', type=str, default='hate_black_1k,neutral_black_1k', help='toxigen prompt file names separated by ","; please check https://github.com/microsoft/TOXIGEN/tree/main/prompts for the full list')
    parser.add_argument('--output_path', type=str, default='.', help='path to save results')

    # toxic detection args
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for model inference')
    parser.add_argument('--num_generations_per_prompt', type=int, default=1, help='number of generations per prompt for toxigen response generation')

    args = parser.parse_args()
    args.prompt_keys = [key for key in args.prompt_keys.split(',')]
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

    datasets = load_prompt_datasets(args.prompt_keys)
    print(f'Toxigen prompt datasets for keys:{args.prompt_keys} loaded')

    # toxicitiy classifier
    classifier = pipeline("text-classification", model="tomh/toxigen_hatebert", tokenizer="bert-base-uncased")

    results = {}

    # generate model responses
    for idx, (key, dataset) in enumerate(zip(args.prompt_keys, datasets)):
        responses = []
        for i in tqdm(range(0, len(dataset), args.batch_size), desc=f'Generating responses for {key}'):
            batch = dataset[i:i+args.batch_size]
            
            inputs = tokenizer(batch['text'], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=args.num_generations_per_prompt,
                do_sample=True
            )
            prompt_len = inputs['input_ids'].shape[1]
            decoded_responses = [tokenizer.decode(t[prompt_len:], skip_special_tokens=True) for t in outputs]
            reshaped_responses = [decoded_responses[j:j+args.num_generations_per_prompt] for j in range(0, len(decoded_responses), args.num_generations_per_prompt)]
            
            responses.extend(reshaped_responses)
        
        response_dataset = dataset.add_column('response', responses)   
        dataset_output_path = os.path.join(args.output_path, f'{model_name}_{key}_responses.jsonl')
        save_prompt_dataset(response_dataset, dataset_output_path)
        print(f'Saved model responses to {dataset_output_path}')
    
        # Evaluate Model Responses
        toxic_num = 0
        for i in tqdm(range(0, len(response_dataset), args.batch_size), desc=f'Evaluating model responses for {key}'):
            batch = response_dataset[i:i+args.batch_size]
            flattened_response = [x for prompt in batch['response'] for x in prompt]
            outputs = classifier(flattened_response)
            toxic_num += sum(1 for row in outputs if row['label'] == 'LABEL_1')
    
        response_num = len(response_dataset)
        toxic_rate = toxic_num / response_num

        results[key] = {
            'toxic_rate': toxic_rate,
            'toxic_num': toxic_num,
            'response_num': response_num
        }

        print(f'Toxic rate for {model_name} responses on {key}: {toxic_num}/{response_num}={toxic_rate:.4f}')

    results_output_path = os.path.join(args.output_path, f"{model_name}_results.jsonl")
    with open(results_output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Results saved to {results_output_path}')

    print(f"\nSummary of Toxicity Rates for {model_name}:")
    for key, data in results.items():
        print(f"{key}: Toxic rate = {data['toxic_rate']:.4f} ({data['toxic_num']}/{data['response_num']})")

if __name__ == '__main__':
    main()    