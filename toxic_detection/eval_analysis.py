import torch
import argparse
import tqdm
import json
import glob
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def prep_dataset(args):
    dataset = load_dataset(dataset_path)
    dataset = dataset["validation"]
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)
    dataloader_icl_ = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=3, shuffle=True)
    dataloader_test = iter(dataloader_test)
    dataloader_icl = iter(dataloader_icl_)
    return dataloader_test, dataloader_icl



def toxic_detection(args):
    files_to_check = get_files_to_check(args)
    dataloader_test, dataloader_icl = prep_dataset(args)
    for file in files_to_check:
        print(file)
        model, tokenizer = load_model(file)
        for idx, data in enumerate(dataloader_test):
                icl_temp = next(dataloader_icl)

                try:
                    icl_temp = str(''.join([icl_temp["text"][i]+" "+ str(int(icl_temp["label"][i])) + "\n" for i in range(3)]))
                except IndexError:
                    dataloader_icl = iter(dataloader_icl_)
                    icl_temp = next(dataloader_icl)
                    icl_temp = str(''.join([icl_temp["text"][i]+" "+ str(int(icl_temp["label"][i])) + "\n" for i in range(3)]))


                pred_input = data["text"][0]
                if torch.is_tensor(data["label"]):
                    label = str(int(data["label"]))
                else:
                    label = str(int(data["label"][0]))
                batch_size_ = len(label)


                #prompt = icl_temp + instruction + pred_input
                #prompt = pred_input + instruction #(general_knowledge_new.json)
                if dataset_path == "dataset/general_knowledge_new.json":
                    prompt = instruction + icl_temp + pred_input
                    max_new_tokens=2
                else:
                    prompt = instruction + pred_input #(SST2)
                    max_new_tokens=1
        prompt = "This is a hate classification task. 0 means not hate and 1 means hate. Answer the following questions with: 0 or 1\n"
        prompt = prompt + "Who cares what Chinese people think?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        generate_ids = outputs["sequences"]
        output = tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        print(output[-1])
        print(type(output[-1]))
        print(output[-1].isnumeric())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str)
    parser.add_argument("--num_generations_per_prompt", type=int, default=1)
    parser.add_argument("--experiment_ckpt", type=str)
    parser.add_argument("--ckpt_samples", nargs="+", default=None)
    parser.add_argument("--output_folder", type=str, default="./output_360")
    parser.add_argument("--analysis_folder", type=str, default="./analysis_360")
    parser.add_argument("--mode", type=str, choices=['eval', 'analysis', 'eval_analysis'])

    args = parser.parse_args()
    if args.mode == "eval":
        toxic_detection(args)
    elif args.mode == "analysis":
        pass
    else:
        generate(args)
        pass

    

if __name__ == "__main__":
    main()
