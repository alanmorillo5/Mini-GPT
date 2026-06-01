import numpy as np
from tokenizer_utils import get_tokenizer

def test_validation():
    enc = get_tokenizer("gpt2")
    train_arr = np.memmap("data/train.bin", dtype=np.uint16, mode='r')
    tokens = train_arr[:10].tolist()  # Sample batch of 10 tokens
    decoded_text = enc.decode(tokens)
    print(f"Sample Tokens: {tokens}")
    print(f"Decoded text:\n{decoded_text}")
    assert len(decoded_text) > 0, "Failed to decode text from train.bin"
    print("Validation passed.")

if __name__ == "__main__":
    test_validation()
