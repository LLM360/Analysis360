from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                 trust_remote_code=True).to(torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer