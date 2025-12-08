import torch
import json
from tokenizer import Tokenizer
from transformerlm import TransformerLM
import torch.nn.functional as F
import os

def decode(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text from a language model."""
    model.eval()
    model.to(device)
    
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Generate tokens one by one
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            logits = model(tokens)
            next_logits = logits[0, -1, :]
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Sample next token
            if top_p is not None:
                next_token = _top_p_sample(next_logits, top_p)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we hit end-of-text
            if next_token.item() == tokenizer.special_token_ids.get("<|endoftext|>"):
                break
            
            # Add to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode and return (remove the prompt part)
    prompt_len = len(tokenizer.encode(prompt))
    generated_tokens = tokens[0, prompt_len:].tolist()
    
    return tokenizer.decode(generated_tokens)


def _top_p_sample(logits, top_p):
    """Top-p (nucleus) sampling"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    
    mask = cumsum > top_p
    mask[0] = False
    sorted_logits[mask] = float('-inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    
    return sorted_indices[idx].unsqueeze(0)


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., 'final.pt')
        config_path: Path to config.json (optional, will look in checkpoint dir)
    
    Returns:
        model, tokenizer, config_dict
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config
    if config_path is None:
        # Try to find config.json in the same directory as checkpoint
        
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config from your training script
        print("Config file not found, using default values...")
        config = {
            'vocab_size': 10000,
            'num_layers': 4,
            'context_length': 256,
            'd_model': 512,
            'd_ff': 1344,
            'num_heads': 16,
            'theta': 10000.0
        }
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        'vocab.json',
        'merges.json',
        special_tokens=["<|endoftext|>"]
    )
    
    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        theta=config['theta']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['params_state'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    return model, tokenizer, config


if __name__ == "__main__":
    # Load model
    model, tokenizer, config = load_model_from_checkpoint(
        checkpoint_path='checkpoints/final.pt',
        config_path='checkpoints/config.json'  # optional
    )
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a forest far away",
    ]
    
    print("\n" + "="*80)
    print("GENERATING TEXT")
    print("="*80 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        # Generate with different settings
        text = decode(model, tokenizer, prompt, max_tokens=50, temperature=0.8)
        print(f"Generated: {prompt}{text}")
        print()
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        # You can adjust these parameters
        text = decode(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=100,
            temperature=0.8,
            # top_p=0.9  # Uncomment for nucleus sampling
        )
        
        print(f"\n{prompt}{text}\n")