import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig

import os
import shutil

def load_model(model_name_or_path, use_lora=False, lora_rank=32):

    model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )
    if use_lora:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            target_modules=["c_attn"],
            r=lora_rank,
            lora_alpha=16,
        )
        model.add_adapter(peft_config)
        model.enable_adapters()
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def save_model(model, tokenizer, path, args, overwrite=False):

    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(f"The directory {path} already exists. Use overwrite=True to overwrite it.")

    if args.use_lora:
        merged_model = model.merge_and_unload()
        merged_model.base_model.save_pretrained(path)
    else:
        model.save_pretrained(path)

    tokenizer.save_pretrained(path)

def get_params(model, module_str, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        layer = eval(module_str.format(model_name='model', layer_id=layer_id))
        for i, p in enumerate(layer.parameters()):
            if i in param_ids:
                params.append(p)
    return params

def forward_with_cache(model, inputs, module, no_grad=True):
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    hook_handle.remove()
    return cache[0]
