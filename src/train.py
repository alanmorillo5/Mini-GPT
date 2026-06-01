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
        x = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
        y = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
        return x.to(device), y.to(device)
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    if len(data) <= seq_len:
        x = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
        y = torch.randint(0, 50257, (batch_size, seq_len), dtype=torch.long)
        return x.to(device), y.to(device)
        
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 4
    seq_len = 64
    max_steps = 1000
    warmup_steps = 100
    max_lr = 6e-4
    min_lr = 6e-5
    
    args = ModelArgs(max_batch_size=batch_size, max_seq_len=seq_len)
    model = GPT(args)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-1)
    
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for step in range(max_steps):
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
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

if __name__ == "__main__":
    main()
