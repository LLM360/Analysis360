import torch
import numpy as np

from methods.training import run_rmu, run_random_mapping, run_min_posterior, run_max_entropy
from analysis.unlearning.utils.data_utils import get_text_dataloader
from utils.model_utils import load_model, save_model, get_params

import schedulefree
from accelerate import Accelerator
import random
import argparse
import wandb
import os

METHOD_CACHE = {
    'max-entropy': run_max_entropy,
    'min-posterior': run_min_posterior,
    'random-mapping': run_random_mapping,
    'rmu': run_rmu,
}

def fix_random(random_seed=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_wandb', type=str, default='True')
    parser.add_argument('--wandb_project', type=str, default='unlearn360')
    
    ### model args
    parser.add_argument('--model_name_or_path', type=str, default='LLM360/CrystalChat')
    parser.add_argument('--use_lora', type=str, default='False')
    parser.add_argument('--output_path', type=str, default='./models')
    
    # data args
    parser.add_argument('--forget_topic', type=str, default='bio-forget', help='Forget set')
    parser.add_argument('--retain_topic', type=str, default='wikitext-test', help='Retain set')
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=1000)

    # general unlearn args
    parser.add_argument('--unlearn_method', type=str, default='max-entropy') 
    parser.add_argument('--fix_vector', type=bool, default=True)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps.")                                                                                                        
    parser.add_argument("--max_unlearn_steps", type=int, default=50, help="Max number of unlearning steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of unlearning.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-1, help="Unlearning LR.")

    # rmu specific args
    parser.add_argument('--module_str', type=str, default='{model_name}.transformer.h[{layer_id}]')
    parser.add_argument('--alpha', type=float, default=10, help='Retain weight')
    parser.add_argument('--steering_coeff', type=float, default=1, help='Steer vector weight')
    parser.add_argument("--layer_id", type=int, default=7, help="Layer to get activations")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="Layers to update")
    parser.add_argument("--param_ids", type=str, default="12,13", help="Params to update")
    
    # llmu specific args
    parser.add_argument("--bad_weight",  type=float, default=0.5, help="Weight on the bad loss.")
    parser.add_argument("--random_weight", type=float, default=1, help="Weight on learning the random outputs.")
    parser.add_argument("--normal_weight", type=float, default=1, help="Weight on normal loss.")

    args = parser.parse_args()
    args.use_wandb = args.use_wandb.lower() == 'true'
    args.use_lora = args.use_lora.lower() == 'true'
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(',')]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(',')]

    return args

def main():

    args = get_args()
    fix_random(args.random_seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    model, tokenizer = load_model(args.model_name_or_path, use_lora=args.use_lora)

    forget_dataloader = get_text_dataloader(
        args.forget_topic,
        tokenizer,
        min_len=args.min_len,
        max_len=args.max_len,
        num_samples=args.max_unlearn_steps * args.batch_size,    
        batch_size=args.batch_size
    )

    retain_dataloader = get_text_dataloader(
        args.retain_topic,
        tokenizer,
        min_len=args.min_len,
        max_len=args.max_len,
        num_samples=args.max_unlearn_steps * args.batch_size, 
        batch_size=args.batch_size
    )

    if args.unlearn_method in METHOD_CACHE:
        unlearn_method = METHOD_CACHE[args.unlearn_method]
    else:
        raise ValueError(f"Unlearn Method {args.unlearn_method} not supported")
    

    if args.unlearn_method == 'rmu':
        frozen_model, _ = load_model(args.model_name_or_path, use_lora=args.use_lora)
        params =  get_params(model, args.module_str, args.layer_ids, args.param_ids)
        optimizer = schedulefree.AdamWScheduleFree(params, lr=args.lr, warmup_steps=args.warmup_steps)
        model, frozen_model, optimizer, forget_dataloader, retain_dataloader = accelerator.prepare(model, frozen_model, optimizer, forget_dataloader, retain_dataloader)
    else:
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=args.lr, warmup_steps=args.warmup_steps)
        model, optimizer, forget_dataloader, retain_dataloader = accelerator.prepare(model, optimizer, forget_dataloader, retain_dataloader)

    run_name = f"{args.unlearn_method}_{args.forget_topic}_{args.model_name_or_path.split('/')[-1]}" 

    if args.use_wandb:
        print(f'use_wandb: {args.use_wandb}')
        if accelerator.is_main_process:
            wandb.init(project=args.wandb_project, name=run_name)
    else:
        wandb = None
    
    if args.unlearn_method == 'rmu':
        unlearned_model = unlearn_method(
            model,
            frozen_model,
            forget_dataloader,
            retain_dataloader,
            optimizer,
            accelerator,
            args,
            wandb
        )
    else:
        unlearned_model = unlearn_method(
            model,
            forget_dataloader,
            retain_dataloader,
            optimizer,
            accelerator,
            args,
            wandb
        )

    if accelerator.is_main_process:
        if args.use_wandb:
            wandb.finish()

        output_path = os.path.join(args.output_path, run_name)
        save_model(unlearned_model, tokenizer, output_path, args, overwrite=True)
        print(f"Model saved to {output_path}")

if __name__ == '__main__':
    main()