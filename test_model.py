import torch
from src.model import GPT, ModelArgs

def test():
    args = ModelArgs()
    model = GPT(args)
    bsz = 2
    seqlen = 1024
    
    # Dummy tokens
    tokens = torch.randint(0, args.vocab_size, (bsz, seqlen))
    
    out = model(tokens)
    
    expected_shape = (bsz, seqlen, args.vocab_size)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"Success! Output shape matches expected: {expected_shape}")

if __name__ == "__main__":
    test()
