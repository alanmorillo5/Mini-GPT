import tiktoken

def get_tokenizer(encoding_name="gpt2"):
    """
    Initializes and returns a BPE tokenizer.
    By default, uses the 'gpt2' encoding via tiktoken.
    """
    return tiktoken.get_encoding(encoding_name)
