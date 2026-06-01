import argparse
import sys
import os

# Add src to sys.path so we can import from it easily without breaking existing scripts
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="Mini-GPT CLI")
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--chat", action="store_true", help="Run chat generation mode")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for chat mode")
    parser.add_argument("--tokens", type=int, default=50, help="Number of tokens to generate")
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training mode...")
        import train
        train.main()
    elif args.chat:
        print("Starting chat mode...")
        import torch
        from generate import load_model, generate_stream
        
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(device=device)
        
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        print(args.prompt, end="", flush=True)
        
        # Stream output token-by-token
        for token_text in generate_stream(model, args.prompt, max_new_tokens=args.tokens, device=device):
            print(token_text, end="", flush=True)
        print("\n" + "-" * 40)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
