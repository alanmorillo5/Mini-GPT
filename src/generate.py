import torch
import os
import torch.nn.functional as F
from src.model import GPT, ModelArgs
from src.tokenizer_utils import get_tokenizer

def load_model(checkpoint_path="checkpoints/ckpt.pt", device="cpu"):
    args = ModelArgs()
    model = GPT(args)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")
    model.to(device)
    model.eval()
    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

@torch.no_grad()
def generate(model, prompt, max_new_tokens=50, device="cpu", temperature=1.0, top_k=50, top_p=0.9):
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= model.args.max_seq_len else x[:, -model.args.max_seq_len:]
        logits = model(x_cond)
        next_token_logits = logits[0, -1, :] 
        
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        
        x = torch.cat((x, next_token), dim=1)
    
    return tokenizer.decode(x[0].tolist())

@torch.no_grad()
def generate_stream(model, prompt, max_new_tokens=50, device="cpu", temperature=1.0, top_k=50, top_p=0.9):
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= model.args.max_seq_len else x[:, -model.args.max_seq_len:]
        logits = model(x_cond)
        next_token_logits = logits[0, -1, :] 
        
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
            
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        x = torch.cat((x, next_token.unsqueeze(0)), dim=1)
        
        yield tokenizer.decode([next_token.item()])
