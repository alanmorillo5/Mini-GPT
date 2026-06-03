import os
import math
import torch
import torch.nn as nn
import numpy as np
from model import GPT, ModelArgs

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def get_batch(split, seq_len, batch_size, device="cpu"):
    data_path = f"data/{split}.bin"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file '{data_path}' not found! "
            f"Run 'python src/dataset.py' first to tokenize your text data."
        )
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    if len(data) <= seq_len:
        raise ValueError(
            f"Data file '{data_path}' has only {len(data)} tokens, "
            f"need at least {seq_len + 1} for seq_len={seq_len}."
        )
        
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, eval_iters, seq_len, batch_size, device):
    out = {}
    model.eval()
    criterion = nn.CrossEntropyLoss()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, seq_len, batch_size, device)
            logits = model(X)
            loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 32
    seq_len = 512
    max_steps = 50000
    warmup_steps = 700
    max_lr = 6e-4
    min_lr = 6e-5
    
    args = ModelArgs(max_batch_size=batch_size, max_seq_len=seq_len)
    model = GPT(args)
    model.to(device)
    
    # Count and report parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-1)
    
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    eval_interval = 500
    eval_iters = 50
    patience = 30          # stop after this many evals with no improvement
    steps_since_improvement = 0
    
    model.train()
    for step in range(max_steps):
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model, eval_iters, seq_len, batch_size, device)
            print(f"Step {step} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                steps_since_improvement = 0
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, "checkpoints/ckpt.pt")
                print(f"Saved new best checkpoint with Val Loss: {best_val_loss:.4f}")
            else:
                steps_since_improvement += 1
                print(f"No improvement for {steps_since_improvement}/{patience} evals")
                if steps_since_improvement >= patience:
                    print(f"Early stopping triggered at step {step} — no improvement for {patience * eval_interval} steps.")
                    break
        
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        xb, yb = get_batch('train', seq_len, batch_size, device)
        
        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

if __name__ == "__main__":
    main()
