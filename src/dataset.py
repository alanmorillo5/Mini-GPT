import os
import glob
import numpy as np
from tokenizer_utils import get_tokenizer

CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per chunk to avoid OOM

def process_data(data_dir="data", out_dir="data", val_ratio=0.1):
    print(f"Reading from {data_dir}...")
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not txt_files:
        print("No .txt files found in", data_dir)
        return

    enc = get_tokenizer("gpt2")

    # Phase 1: Tokenize in chunks and collect all tokens
    all_tokens = []
    for fpath in txt_files:
        file_size = os.path.getsize(fpath)
        print(f"Tokenizing {os.path.basename(fpath)} ({file_size / 1e6:.1f} MB)...")
        with open(fpath, "r", encoding="utf-8") as f:
            remainder = ""
            chunks_done = 0
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    # Encode any leftover text
                    if remainder:
                        tokens = enc.encode(remainder, allowed_special="all")
                        all_tokens.extend(tokens)
                    break
                text = remainder + chunk
                # Split on last newline so we don't cut mid-word
                last_nl = text.rfind("\n")
                if last_nl == -1:
                    remainder = text
                    continue
                to_encode = text[: last_nl + 1]
                remainder = text[last_nl + 1 :]
                tokens = enc.encode(to_encode, allowed_special="all")
                all_tokens.extend(tokens)
                chunks_done += 1
                if chunks_done % 10 == 0:
                    mb_done = chunks_done * CHUNK_SIZE / 1e6
                    print(f"  ...processed ~{mb_done:.0f} MB, {len(all_tokens):,} tokens so far")

    total = len(all_tokens)
    print(f"Total tokens: {total:,}")

    # Phase 2: Split and write binary shards
    val_count = int(total * val_ratio)
    train_count = total - val_count

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")

    print(f"Writing train.bin ({train_count:,} tokens)...")
    train_arr = np.memmap(train_path, dtype=np.uint16, mode='w+', shape=(train_count,))
    train_arr[:] = all_tokens[:train_count]
    train_arr.flush()

    print(f"Writing val.bin ({val_count:,} tokens)...")
    val_arr = np.memmap(val_path, dtype=np.uint16, mode='w+', shape=(val_count,))
    val_arr[:] = all_tokens[train_count:]
    val_arr.flush()

    print(f"Done! Saved {train_count:,} train tokens and {val_count:,} val tokens to {out_dir}/")

if __name__ == "__main__":
    process_data()
