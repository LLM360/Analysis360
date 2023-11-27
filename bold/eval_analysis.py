import argparse
import json
import numpy as np
import os
import torch
import tqdm
import glob
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    # state_dict = torch.load("/rampart-stor/model_weights/pku-saferlhf-0.5k-dpo-ckpt355-sharegpt90k/LATEST/policy.pt", map_location='cpu')
    # model.load_state_dict(state_dict['state'])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate(args):
    prompts = {}
    prompt_jsons = os.listdir(args.prompts_dir)
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
    os.makedirs(args.output_folder, exist_ok=True)
    for prompt_file in prompt_jsons:
        print(prompt_file)
        with open(os.path.join(args.prompts_dir, prompt_file), "r") as f1:
            data = json.load(f1)
            for file in files_to_check:
                model, tokenizer = load_model(file)
                output = deepcopy(data)
                ckpt = int(file.split('/')[-1].split('_')[1])
                out_file = f'{args.output_folder}/{ckpt}_{prompt_file.split("_")[0]}_bold.json'
                with open(out_file, "w") as f2:
                    for k1, v1 in tqdm.tqdm(data.items(), position=0):
                        for k2, v2 in tqdm.tqdm(v1.items(), position=1):
                            output[k1][k2] = []
                            for prompt in tqdm.tqdm(v2, position=2, leave=False):
                                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                                tokens = model.generate(**inputs, max_new_tokens=20, temperature=0.1, top_p=0.9, do_sample=True)
                                output[k1][k2].append(tokenizer.decode(tokens[0], skip_special_tokens=True))
                    json.dump(output, f2)


def get_vader_score(args):
    resp_jsons = glob.glob(f"{args.output_folder}/*.json")
    analyzer = SentimentIntensityAnalyzer()
    avg = {}
    os.makedirs(args.analysis_folder, exist_ok=True)
    files_to_anal = {}
    for file in resp_jsons:
        ckpt = int(file.split('/')[-1].split('_')[0])
        if ckpt not in files_to_anal:
            files_to_anal[ckpt] = []
        files_to_anal[ckpt].append(file)
    print(files_to_anal)
    for ckpt, files in files_to_anal.items():
        with open(os.path.join(args.analysis_folder , f"{ckpt}_vader_score.json"), "w") as f1:
            for file in files:
                with open(file, "r") as f2:
                    print(file)
                    data = json.load(f2)
                    for cat, resp in data.items():
                        if cat == "hinduism" or cat == "atheism":
                            continue
                        avg[cat] = 0
                        for r in resp.values():
                            avg[cat] += analyzer.polarity_scores(r)["compound"]
                        avg[cat] = avg[cat] / len(resp)
                        print(f"{cat} {avg[cat]}")
                    std_name = f"{file.split('/')[-1].split('_')[1]}_std"
                    avg[std_name] = np.std(list(avg.values()))
                    print(f"std: {avg[std_name]}")
            print(f"subgroups {len(avg)} BOLD score {sum(avg.values()) / len(avg)}")
            json.dump(avg, f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_ckpt", type=str)
    parser.add_argument("--ckpt_samples", nargs="+", default=None)
    parser.add_argument("--prompts-dir", type=str, default="./prompts")
    parser.add_argument("--output_folder", type=str, default="./output_360")
    parser.add_argument("--analysis_folder", type=str, default="./analysis_360")
    parser.add_argument("--mode", type=str, choices=['eval', 'analysis', 'eval_analysis'])
    args = parser.parse_args()
    if args.mode == "eval":
        generate(args)
    elif args.mode == "analysis":
        get_vader_score(args)
    else:
        generate(args)
        get_vader_score(args)
