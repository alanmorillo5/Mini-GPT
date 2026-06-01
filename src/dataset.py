import os
import glob
import numpy as np
from tokenizer_utils import get_tokenizer

def process_data(data_dir="data", out_dir="data", val_ratio=0.1):
    print(f"Reading from {data_dir}...")
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not txt_files:
        print("No .txt files found in", data_dir)
        return
        
    all_text = ""
    for fpath in txt_files:
        with open(fpath, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
            
    print("Tokenizing...")
    enc = get_tokenizer("gpt2")
    tokens = enc.encode(all_text)
    print(f"Total tokens: {len(tokens)}")
    
    val_count = int(len(tokens) * val_ratio)
    train_count = len(tokens) - val_count
    
    train_tokens = tokens[:train_count]
    val_tokens = tokens[train_count:]
    
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    
    print("Writing train.bin...")
    train_arr = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(len(train_tokens),))
    train_arr[:] = train_tokens
    train_arr.flush()
    
    print("Writing val.bin...")
    val_arr = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(len(val_tokens),))
    val_arr[:] = val_tokens
    val_arr.flush()
    
    print(f"Saved {len(train_tokens)} train tokens and {len(val_tokens)} val tokens to {out_dir}")

if __name__ == "__main__":
    process_data()
