# Data Directory

This directory is intended to store the raw `.txt` files that make up your training dataset. 

## How the Dataset is Compiled
The data ingestion pipeline (located in `src/dataset.py`) looks for any raw text files within this `data/` directory, concatenates them, and tokenizes them using the GPT-2 Byte-Pair Encoding (`tiktoken`). It then splits the tokenized data into a 90/10 train and validation split, saving them to efficient memory-mapped binaries (`train.bin` and `val.bin`) which the model reads during training. This happens automatically when you run `python main.py --train`.

## Where to get Raw Text Data
To train a generalized language model, you need a substantial corpus of text data. Here are a few great open-source datasets you can use:

1. **TinyShakespeare**: A popular beginner dataset comprising the works of William Shakespeare. It's excellent for small, character-level or mini-GPT models.
   - **Download:** [tinyshakespeare.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

2. **OpenWebText**: A much larger, web-crawled dataset similar to what was used for OpenAI's GPT-2. You can download a small sample or the full corpus from HuggingFace.
   - **Download:** [OpenWebText on HuggingFace](https://huggingface.co/datasets/openwebtext)

3. **Project Gutenberg**: A library of over 60,000 free eBooks. You can download plain `.txt` files of classic literature.
   - **Download:** [Gutenberg Text Dump](https://www.gutenberg.org/)

## Instructions
1. Download any of the text datasets mentioned above (or bring your own custom text).
2. Save the plain text files with a `.txt` extension directly into this `data/` directory.
3. Run the training script: `python ../main.py --train`. The pipeline will automatically build `train.bin` and `val.bin`!
