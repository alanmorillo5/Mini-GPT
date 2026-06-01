import torch
import os
from model import GPT, ModelArgs
from tokenizer_utils import get_tokenizer

def load_model(checkpoint_path="checkpoints/ckpt.pt", device="cpu"):
    args = ModelArgs()
    model = GPT(args)
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def generate(model, prompt, max_new_tokens=50, device="cpu"):
    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= model.args.max_seq_len else x[:, -model.args.max_seq_len:]
        logits = model(x_cond)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        x = torch.cat((x, next_token), dim=1)
    
    return tokenizer.decode(x[0].tolist())
