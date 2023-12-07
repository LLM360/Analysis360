import torch
import argparse
import tqdm
import json
import glob
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from toxigen.language_models import GPT3, ALICE
from toxigen.pretrained_classifiers import HateBERT, ToxDectRoBERTa


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # state_dict = torch.load("/rampart-stor/model_weights/pku-saferlhf-0.5k-dpo-ckpt355-sharegpt90k/LATEST/policy.pt", map_location='cpu')
    # model.load_state_dict(state_dict['state'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_files_to_check(args):
    files = glob.glob(f"{args.experiment_ckpt}/ckpt_*")
    files_to_check = []
    if args.ckpt_samples is None:
        files_to_check = files
    else:
        ckpt_samples = [int(ckpt) for ckpt in args.ckpt_samples]
        for file in files:
            ckpt = int(file.split('/')[-1].split('_')[1])
            if args.ckpt_samples and ckpt in ckpt_samples:
                files_to_check.append(file)
    return files_to_check


def generate(args):
    files_to_check = get_files_to_check(args)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(args.input_prompt_file, "r") as f:
        prompts = json.load(f)
    for file in files_to_check:
        model, tokenizer = load_model(file)
        seen = set()
        results = []
        for prompt in tqdm.tqdm(prompts):
            if prompt["text"] not in seen:
                seen.add(prompt["text"])
                for i in range(args.num_generations_per_prompt):
                    inputs = tokenizer(prompt["text"], return_tensors="pt").to("cuda")
                    tokens = model.generate(**inputs, max_new_tokens=20, temperature=0.1, top_p=0.9, do_sample=True)
                    response = tokenizer.decode(tokens[0], skip_special_tokens=True)
                    prompt["response"] = response
                    results.append(prompt)
        ckpt = int(file.split('/')[-1].split('_')[1])
        out_file = f'{args.output_folder}/{ckpt}_toxigen.json'
        with open(args.output_file, "w") as f:
            json.dump(results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_prompt_file", type=str, default="./prompts/toxiGen.json")
    # parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_generations_per_prompt", type=int, default=1)
    parser.add_argument("--experiment_ckpt", type=str)
    parser.add_argument("--ckpt_samples", nargs="+", default=None)
    # parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_folder", type=str, default="./output_360")
    parser.add_argument("--analysis_folder", type=str, default="./analysis_360")
    parser.add_argument("--mode", type=str, choices=['eval', 'analysis', 'eval_analysis'])

    args = parser.parse_args()
    if args.mode == "eval":
        generate(args)
    elif args.mode == "analysis":
        pass
    else:
        generate(args)
        pass

    

if __name__ == "__main__":
    main()
