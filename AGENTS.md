# Agent Implementation Roadmap: Mini-GPT Local Trainer

This document specifies the autonomous steps, responsibilities, validation checks, and strict version control protocols required to implement a scratch-built mini-GPT model running locally on Apple Silicon (M-Series).

---

## 1. System Architecture Specs
* **Model Type:** Decoder-Only Transformer
* **Target Backend:** PyTorch (`mps`) or Apple MLX
* **Target Params:** ~125 Million
* **Context Length:** 1024 tokens

---

## 2. Version Control & Git Protocol (CRITICAL)

To prevent merge conflicts and ensure a clean history, all agents **must** adhere to the following workflow for every single task:
1. **Sync First:** Run `git pull --rebase origin main` before making any file changes.
2. **Atomic Work:** Complete exactly *one* task or validation test.
3. **Stage and Commit:** Run `git add <specific_files>` followed by a descriptive commit: `git commit -m "[Agent Name] Completed Task X.Y: <Brief description>"`.
4. **Push:** Run `git push origin main`.
*Note: If agents are running concurrently, they must create feature branches (e.g., `git checkout -b feature/agent-a-tokenizer`) and submit Pull Requests instead of pushing directly to main.*

---

## 3. Agent Roles & Workflows

### Agent A: The Data Architect
**Objective:** Ingest raw local text files, process them into tokens, and save them into efficient training shards.

* [ ] **Task 1.1:** Write `src/tokenizer_utils.py` to initialize a BPE tokenizer (Tiktoken `gpt2` or custom).
  * *Commit:* `git commit -m "[Agent A] Task 1.1: Implemented BPE tokenizer initialization"`
* [ ] **Task 1.2:** Create a pipeline (`src/dataset.py`) that reads `.txt` files from `/data`, concatenates, tokens, and splits them into `train.bin` and `val.bin` via `np.memmap`.
  * *Commit:* `git commit -m "[Agent A] Task 1.2: Built data ingestion and binary sharding pipeline"`
* [ ] **Validation 1.3:** Verify decoding a sample batch from `train.bin` returns the exact original text.
  * *Commit:* `git commit -m "[Agent A] Validation 1.3: Verified data pipeline integrity and memory-mapping"`

### Agent B: The Model Sculptor
**Objective:** Build the neural network graph cleanly without external dependencies.

* [ ] **Task 2.1:** Implement RMSNorm and RoPE (Rotary Position Embedding) layers.
  * *Commit:* `git commit -m "[Agent B] Task 2.1: Implemented RMSNorm and RoPE layers"`
* [ ] **Task 2.2:** Build the Causal Self-Attention block supporting Multi-Query Attention.
  * *Commit:* `git commit -m "[Agent B] Task 2.2: Built Causal Self-Attention with MQA"`
* [ ] **Task 2.3:** Construct the Transformer block combining Attention, SwiGLU MLP, and residual connections.
  * *Commit:* `git commit -m "[Agent B] Task 2.3: Assembled core Transformer block"`
* [ ] **Task 2.4:** Assemble everything into a unified `GPT(nn.Module)` class in `src/model.py`.
  * *Commit:* `git commit -m "[Agent B] Task 2.4: Unified architecture into main GPT model class"`
* [ ] **Validation 2.5:** Run a dummy tensor `(batch_size, context_len)` through the model to verify output shape `(batch_size, context_len, vocab_size)`.
  * *Commit:* `git commit -m "[Agent B] Validation 2.5: Confirmed forward pass tensor shape matching"`

### Agent C: The Trainer & Optimization Engineer
**Objective:** Set up the training loop, leverage Apple Silicon acceleration, and save model checkpoints.

* [ ] **Task 3.1:** Write `src/train.py` utilizing the AdamW optimizer with a learning rate warmup/decay schedule (Cosine Annealing).
  * *Commit:* `git commit -m "[Agent C] Task 3.1: Configured AdamW and learning rate scheduler"`
* [ ] **Task 3.2:** Enforce execution on `device = "mps"` (PyTorch) or native MLX arrays for M-series optimization.
  * *Commit:* `git commit -m "[Agent C] Task 3.2: Implemented Apple Silicon MPS/MLX hardware routing"`
* [ ] **Task 3.3:** Implement checkpointing logic (`/checkpoints/ckpt.pt`) tracking the lowest validation loss.
  * *Commit:* `git commit -m "[Agent C] Task 3.3: Added best-loss checkpoint saving logic"`
* [ ] **Validation 3.4:** Perform the "Overfit Test" on 5 sentences. Confirm loss drops towards zero within 100 steps.
  * *Commit:* `git commit -m "[Agent C] Validation 3.4: Passed overfit sanity check on micro-dataset"`

### Agent D: The Inference & Interaction Interface
**Objective:** Provide a fast CLI loop for user input and text generation.

* [ ] **Task 4.1:** Write `src/generate.py` that loads a checkpoint, accepts a text string prompt, tokenizes it, and runs a generation loop.
  * *Commit:* `git commit -m "[Agent D] Task 4.1: Created checkpoint loader and basic generation loop"`
* [ ] **Task 4.2:** Implement Top-K, Top-P, and temperature controls to prevent repetitive loops.
  * *Commit:* `git commit -m "[Agent D] Task 4.2: Added Top-K, Top-P, and temperature sampling logic"`
* [ ] **Task 4.3:** Build a clean CLI wrapper (`main.py`) with modes for `--train` and `--chat`.
  * *Commit:* `git commit -m "[Agent D] Task 4.3: Built main CLI wrapper with train and chat modes"`
* [ ] **Validation 4.4:** Generate 50 tokens from a prompt and verify the model streams the text cleanly token-by-token back to the terminal.
  * *Commit:* `git commit -m "[Agent D] Validation 4.4: Verified token streaming output in terminal"`
