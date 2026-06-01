# Mini-GPT Local Trainer

A scratch-built, lightweight GPT (Decoder-Only Transformer) model optimized for training and inference locally on Apple Silicon (M-Series) utilizing PyTorch's `mps` backend. 

## Features
- **Architecture**: Decoder-Only Transformer (~125 Million Parameters, 1024 Token Context Length).
- **Modern Components**: Includes RMSNorm, Rotary Position Embeddings (RoPE), Multi-Query Attention (MQA), and SwiGLU MLP.
- **Fast Local Training**: AdamW optimizer with Cosine Annealing learning rate schedule, built for PyTorch `mps` hardware routing.
- **Data Ingestion**: Custom Byte-Pair Encoding (BPE) integration with `tiktoken` (GPT-2 encoding) and efficient `np.memmap` binary sharding.
- **Inference Engine**: Streaming generation loop with dynamic Top-K, Top-P, and Temperature sampling controls.

## Quick Start

### 1. Prerequisites
Ensure you have a Python environment set up with PyTorch and `tiktoken` installed.
```bash
pip install torch numpy tiktoken
```

### 2. Prepare Data
Place your raw text data `.txt` files into the `data/` directory. The data pipeline will automatically aggregate, tokenize, and shard them into efficient `train.bin` and `val.bin` files.

### 3. Training the Model
To start training the model from scratch, simply use the `main.py` wrapper in `--train` mode:
```bash
python main.py --train
```
This will run the training loop, track validation loss, and automatically save the best-performing model weights to `checkpoints/ckpt.pt`.

### 4. Interactive Chat (Inference)
Once the model is trained and a checkpoint is saved, you can chat with it! The generation logic streams tokens directly to the terminal.
```bash
python main.py --chat
```
This drops you into an interactive session where you can prompt the model and watch it generate responses.

## Codebase Structure
- `src/model.py`: The neural network graph (`GPT` module, MQA, SwiGLU, RMSNorm, RoPE).
- `src/train.py`: Training loop, optimizer setup, learning rate scheduling, and checkpoint logic.
- `src/dataset.py`: Memory-mapped binary sharding (`train.bin`, `val.bin`) logic.
- `src/generate.py`: Inference engine, checkpoint loader, and Top-K/Top-P/Temperature sampling logic.
- `src/tokenizer_utils.py`: Text encoding and decoding wrapper leveraging `tiktoken`.
- `main.py`: The primary CLI wrapper routing to the respective train and chat modes.
