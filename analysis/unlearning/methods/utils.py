import torch
from torch.nn import CrossEntropyLoss

def cos_sim_loss(hidden_states, input_ids, stream_hash_table, hidden_size,):
    f_stream = [stream.view(-1, hidden_size) for stream in hidden_states]
    input_ids = input_ids.view(-1)
    rand_stream = [stream_hash_table[input_ids] for _ in f_stream]

    cos_sim_loss = (
        torch.stack(
            [
                (1 - torch.abs(torch.nn.functional.cosine_similarity(f, r, dim=-1))).mean()
                for f, r in zip(f_stream, rand_stream)
            ]
        ).mean()
    )

    return cos_sim_loss

def mse_loss(activations_1, activations_2):
    return torch.nn.functional.mse_loss(activations_1, activations_2)

def max_entropy_loss(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return -1 * entropy.mean()

def lm_loss(logits, labels, vocab_size,):
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1)

    loss_f = torch.nn.CrossEntropyLoss()
    loss = loss_f(shift_logits, shift_labels.to(shift_logits.device))
    return loss

def log_p_loss(logits, labels, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def log_1_minus_p_loss(logits, labels, vocab_size, threshold: float = -15.0):
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)

    log_sum_exp_all = torch.logsumexp(logits, dim=-1)

    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0

    logits_for_labels = torch.gather(logits, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)

    log_p = logits_for_labels - log_sum_exp_all

    masks = torch.zeros_like(logits).scatter(-1, gather_labels.unsqueeze(-1), 1.0)

    masked_logits = logits * (1 - masks) + masks * (-1e10) 
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)

    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all

    ignored_values = labels == -100
    log_1_minus_p[ignored_values] = 0

    below_threshold = log_p < threshold
    log_1_minus_p[below_threshold] = 0

    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()

    return loss