import torch
import torch.nn as nn
from model import GPT, ModelArgs

def test_overfit():
    # Micro-dataset (5 sentences)
    vocab_size = 50
    seq_len = 16
    batch_size = 5
    
    # Fake tokenizer mapping
    # Just generating random data for 5 sentences
    torch.manual_seed(42)
    X = torch.randint(0, vocab_size, (batch_size, seq_len))
    Y = X.clone()
    Y[:, :-1] = X[:, 1:]  # Shift left by 1 token
    Y[:, -1] = 0          # Pad the final token
    
    args = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=2,
        vocab_size=vocab_size,
        max_batch_size=batch_size,
        max_seq_len=seq_len
    )
    
    model = GPT(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    print("Starting overfit test on 5 sentences...")
    
    initial_loss = None
    final_loss = None
    
    for step in range(100):
        logits = model(X)
        loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        if step == 99:
            final_loss = loss.item()
            
        if step % 20 == 0 or step == 99:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < 0.1 or final_loss < initial_loss * 0.1, "Loss did not drop sufficiently"
    print("Overfit test passed! Loss dropped towards zero.")

if __name__ == "__main__":
    test_overfit()
