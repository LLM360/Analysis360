import torch
from transformers import AutoModelForCausalLM

ckpt_template = "ckpt_{id}"

def load_model(model_id):
    # NOTE: only support Amber ckpts for now
    model_id = str(model_id).zfill(3)
    model_id = ckpt_template.format(id=model_id)
    model = AutoModelForCausalLM.from_pretrained("LLM360/Amber", revision=model_id).to(torch.float16).cuda()

    return model