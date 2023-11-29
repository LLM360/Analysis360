from transformers import AutoModelForCausalLM, LlamaForCausalLM
import json
import torch
from tqdm import tqdm, trange
import os


ckpt_path = "/lustre/scratch/shared-folders/llm_project/bowen.tan/7b_ckpts/ckpt_{id}"
data_path = "/lustre/home/yi.gu/memorization/sampled_data/{id}.jsonl"



def evaluate(ckpt_id, data_start, data_end):
    ckpt = ckpt_path.format(id=ckpt_id)
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(ckpt, use_flash_attention_2=True).to(torch.float16).cuda()
    os.makedirs(f'result_{ckpt_id}', exist_ok=True)

    for data_id in trange(data_start, data_end):
        data = data_path.format(id=data_id)
        with open(data) as f, open(f'result_{ckpt_id}/{data_id}.json', 'w') as g:

            count = 0
            all_match = []
            total_token_match = []

            lines = [json.loads(line) for line in f]
            for start in trange(0, len(lines), 360):
                end = min(start + 360, len(lines))
                samples = lines[start:end]

                prompt = torch.tensor([sample['prompt'] for sample in samples]).cuda()
                continuation = torch.tensor([sample['continuation'] for sample in samples]).cuda()
                generated = model.generate(prompt, min_length=32, max_new_tokens=32, num_beams=1, do_sample=False)[:, 32:]
                total_token_match += (generated == continuation).sum(dim=1).tolist()
                all_match += (generated == continuation).int().tolist()
                count += end - start
            
            json.dump({
                'full_match': sum(1 for x in total_token_match if x == 32) / count,
                'token_match': sum(total_token_match) / 32 / count,
                'count': count,
                'match': total_token_match,
                'all_match': all_match,
            }, g)


if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)
