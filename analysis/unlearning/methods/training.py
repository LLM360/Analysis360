import torch

from methods.utils import max_entropy_loss, lm_loss, log_p_loss, log_1_minus_p_loss, cos_sim_loss, mse_loss
from utils.model_utils import forward_with_cache

from tqdm import tqdm
import json
import math

def run_llmu(
        updated_model,
        frozen_model,
        forget_dataloader,
        retain_dataloader,
        optimizer,
        accelerator,
        args,
        wandb=None,
):
    
    updated_model.train()
    updated_model.zero_grad()

    local_log = {
        "total_loss": [],
        "rand_loss": [],
        "normal_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_rand_loss = 0.0
    total_normal_loss = 0.0
    total_forget_loss = 0.0

    retain_ans = [x['text'][:args.max_len] for x in retain_dataloader.dataset if args.min_len <= len(x['text'])]

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc='Training', leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx > args.max_unlearn_steps:
                break

            # forget loss
            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = updated_model(**forget_batch_squeezed, output_hidden_states=True)
            forget_loss = (
                -1.0 * lm_loss(outputs.logits, forget_batch_squeezed["labels"], updated_model.config.vocab_size) / args.gradient_accumulation_steps
            )
            
            # rand loss
            




            # total loss
            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                updated_model.zero_grad()
                if wandb:
                    wandb.log(
                        {
                            "train/loss": total_loss, 
                            "train/retain_loss": total_retain_loss, 
                            "train/forget_loss": total_forget_loss
                        },
                        step=idx+1
                    )

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_postfix({
                    "Total Loss": f"{total_loss:.4f}",
                    "Retain Loss": f"{total_retain_loss:.4f}",
                    "Forget Loss": f"{total_forget_loss:.4f}"
                })
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")    
    
    return updated_model
    

def run_rmu(
        updated_model,
        frozen_model,
        forget_dataloader,
        retain_dataloader,
        optimizer,
        accelerator,
        args,
        wandb=None,
):
    
    updated_model.train()
    updated_model.zero_grad()

    local_log = {
        "total_loss": [],
        "retain_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    frozen_module = eval(
        args.module_str.format(model_name='frozen_model',layer_id=args.layer_id)
    )
    updated_module = eval(
        args.module_str.format(model_name='updated_model',layer_id=args.layer_id)
    )

    g = None
    if args.fix_vector:
        g = torch.Generator().manual_seed(42)

    random_vector = torch.rand(1,1, updated_model.config.hidden_size, 
                                    dtype=updated_model.dtype,  
                                    generator=g).to(accelerator.device)
    control_vector = random_vector / torch.norm(random_vector) * args.steering_coeff

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc='Training', leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx > args.max_unlearn_steps:
                break

            # retain loss
            retain_batch_squeezed = {
                key: value.squeeze()
                for key, value in retain_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            updated_retain_activations = forward_with_cache(
                        updated_model, retain_batch_squeezed, module=updated_module, no_grad=False
                    ).to(accelerator.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, retain_batch_squeezed, module=frozen_module, no_grad=True
            ).to(accelerator.device)

            retain_loss = (
                mse_loss(updated_retain_activations, frozen_retain_activations) / args.gradient_accumulation_steps
            )
            retain_loss *= args.alpha
            accelerator.backward(retain_loss)

            # forget loss
            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            updated_forget_activations = forward_with_cache(
                updated_model, forget_batch_squeezed, module=updated_module, no_grad=False
            ).to(accelerator.device)

            forget_loss = (
                mse_loss(updated_forget_activations, control_vector) / args.gradient_accumulation_steps
            )
            accelerator.backward(forget_loss)

            # total loss
            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                updated_model.zero_grad()
                if wandb:
                    wandb.log(
                        {
                            "train/loss": total_loss, 
                            "train/retain_loss": total_retain_loss, 
                            "train/forget_loss": total_forget_loss
                        },
                        step=idx+1
                    )

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_postfix({
                    "Total Loss": f"{total_loss:.4f}",
                    "Retain Loss": f"{total_retain_loss:.4f}",
                    "Forget Loss": f"{total_forget_loss:.4f}"
                })
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")    
    
    return updated_model

def run_random_mapping(
        model,
        forget_dataloader,
        retain_dataloader,
        optimizer,
        accelerator,
        args,
        wandb=None,
):
    
    model.train()
    model.zero_grad()

    local_log = {
        "total_loss": [],
        "retain_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    g = None
    if args.fix_vector:
        g = torch.Generator().manual_seed(42)

    stream_hash_table = torch.randn(
        model.config.vocab_size,
        model.config.hidden_size,
        requires_grad=False,
        generator=g
    ).to(accelerator.device)

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc='Training', leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx > args.max_unlearn_steps:
                break

            # retain loss
            retain_batch_squeezed = {
                key: value.squeeze()
                for key, value in retain_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**retain_batch_squeezed, output_hidden_states=False)
            retain_loss = (
                lm_loss(outputs.logits, retain_batch_squeezed["labels"], model.config.vocab_size) / args.gradient_accumulation_steps
            )
            accelerator.backward(retain_loss)

            # forget loss
            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**forget_batch_squeezed, output_hidden_states=True)
            forget_loss = (
                cos_sim_loss(outputs.hidden_states, forget_batch_squeezed['input_ids'], stream_hash_table, model.config.hidden_size) / args.gradient_accumulation_steps
            )
            accelerator.backward(forget_loss)

            # total loss
            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                if wandb:
                    wandb.log(
                        {
                            "train/loss": total_loss, 
                            "train/retain_loss": total_retain_loss, 
                            "train/forget_loss": total_forget_loss
                        },
                        step=idx+1
                    )

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_postfix({
                    "Total Loss": f"{total_loss:.4f}",
                    "Retain Loss": f"{total_retain_loss:.4f}",
                    "Forget Loss": f"{total_forget_loss:.4f}"
                })
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")    
    
    return model

def run_min_posterior(
        model, 
        forget_dataloader, 
        retain_dataloader, 
        optimizer, 
        accelerator, 
        args,
        wandb=None,
):
    
    model.train()
    model.zero_grad()

    local_log = {
        "total_loss": [],
        "retain_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc="Training", leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx > args.max_unlearn_steps:
                break
            
            # retain loss
            retain_batch_squeezed = {
                key: value.squeeze()
                for key, value in retain_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**retain_batch_squeezed, output_hidden_states=False)
            retain_loss = (
                log_p_loss(outputs.logits, retain_batch_squeezed["labels"], model.config.vocab_size) / args.gradient_accumulation_steps
            )
            accelerator.backward(retain_loss)

            # forget loss
            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**forget_batch_squeezed, output_hidden_states=False)
            forget_loss = (
                log_1_minus_p_loss(outputs.logits, forget_batch_squeezed["labels"], model.config.vocab_size
                ) / args.gradient_accumulation_steps
            )
            accelerator.backward(forget_loss)
            
            # total loss
            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                if wandb:
                    wandb.log(
                        {
                            "train/loss": total_loss, 
                            "train/retain_loss": total_retain_loss, 
                            "train/forget_loss": total_forget_loss
                        },
                        step=idx+1
                    )

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_postfix({
                    "Total Loss": f"{total_loss:.4f}",
                    "Retain Loss": f"{total_retain_loss:.4f}",
                    "Forget Loss": f"{total_forget_loss:.4f}"
                })
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")    
    
    return model

def run_max_entropy(
      model,
      forget_dataloader,
      retain_dataloader,
      optimizer,
      accelerator,
      args,  
      wandb=None,
):
    model.train()
    model.zero_grad()

    local_log = {
        "total_loss": [],
        "retain_loss": [],
        "forget_loss": []
    }

    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0

    total_steps = math.ceil(args.max_unlearn_steps / args.gradient_accumulation_steps)

    with tqdm(total=total_steps, desc="Training", leave=True) as pbar:
        for idx, (forget_batch, retain_batch) in enumerate(zip(forget_dataloader, retain_dataloader)):
            if idx >= args.max_unlearn_steps: 
                break

            retain_batch_squeezed = {
                key: value.squeeze() 
                for key, value in retain_batch.items() 
                if key in {"input_ids", "labels", "attention_mask"}
            }
            outputs = model(**retain_batch_squeezed, output_hidden_states=False)
            retain_loss = (
                lm_loss(outputs.logits, retain_batch_squeezed["labels"], model.config.vocab_size) / args.gradient_accumulation_steps
            )
            accelerator.backward(retain_loss)

            forget_batch_squeezed = {
                key: value.squeeze()
                for key, value in forget_batch.items()
                if key in ["input_ids", "labels", "attention_mask"]
            }
            outputs = model(**forget_batch_squeezed, output_hidden_states=False)
            forget_loss = (
                max_entropy_loss(outputs.logits) / args.gradient_accumulation_steps
            )
            accelerator.backward(forget_loss)

            total_retain_loss += retain_loss.item()
            total_forget_loss += forget_loss.item()
            total_loss += retain_loss.item() + forget_loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

                if wandb:
                    wandb.log(
                        {
                            "train/loss": total_loss, 
                            "train/retain_loss": total_retain_loss, 
                            "train/forget_loss": total_forget_loss
                        },
                        step=idx+1
                    )

                local_log["total_loss"].append(total_loss)
                local_log["retain_loss"].append(total_retain_loss)
                local_log["forget_loss"].append(total_forget_loss)

                pbar.set_postfix({
                    "Total Loss": f"{total_loss:.4f}",
                    "Retain Loss": f"{total_retain_loss:.4f}",
                    "Forget Loss": f"{total_forget_loss:.4f}"
                })
                pbar.update(1)

                total_loss = 0.0
                total_retain_loss = 0.0
                total_forget_loss = 0.0

    with open('log.json', "w") as f:
        json.dump(local_log, f, indent=4)
        print(f"Log saved to log.json")

    return model