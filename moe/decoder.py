import torch
import json
import torch.nn.functional as F
import os
import sys

# Add the parent directory (root) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basic_transformer.tokenizer import Tokenizer
# CHANGED: Import MoeLM instead of TransformerLM
from moelm import MoeLM 

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
            # CHANGED: Unpack the tuple! MoE returns (logits, aux_loss)
            logits, _ = model(tokens)
            
            next_logits = logits[0, -1, :]
            
            # Apply temperature
            next_logits = next_logits / temperature
            
            # Sample next token
            if top_p > 0.0:
                next_token = _top_p_sample(next_logits, top_p)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = next_token.unsqueeze(0)
            
            
            # Stop if we hit end-of-text
            # (Assuming you have a way to access special token IDs, or hardcode 50256 for GPT2)
            if hasattr(tokenizer, 'special_token_ids'):
                eos_id = tokenizer.special_token_ids.get("<|endoftext|>")
                if eos_id is not None and next_token.item() == eos_id:
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
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumsum > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def load_model_from_checkpoint(checkpoint_path, config_path=None):
    """
    Load model from checkpoint.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load config
    if config_path is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config updated with MoE params
        print("Config file not found, using default values...")
        config = {
            'vocab_size': 50257,
            'num_layers': 4,
            'context_length': 256,
            'd_model': 512,
            'd_ff': 2048,
            'num_heads': 8,
            'theta': 10000.0,
            # CHANGED: Added MoE Defaults
            'num_experts': 8,
            'top_k': 2
        }
    
    # Load tokenizer
    print("Loading tokenizer...")
    # Adjust path if your vocab files are elsewhere
    tokenizer = Tokenizer.from_files(
        'vocab.json',
        'merges.json',
        special_tokens=["<|endoftext|>"]
    )
    
    # Create model
    print("Creating MoE model...")
    # CHANGED: Instantiate MoeLM
    model = MoeLM(
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        theta=config['theta'],
        # Pass MoE params
        num_experts=config.get('num_experts', 8),
        top_k=config.get('top_k', 2)
    )
    
    # Load weights
    # Note: If you saved the whole model state dict under 'model_state_dict', change 'params_state' below
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('params_state', checkpoint))
    
    # Strict=False allows loading if there are minor mismatches (useful during dev)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    return model, tokenizer, config


if __name__ == "__main__":
    # Ensure paths are correct
    model, tokenizer, config = load_model_from_checkpoint(
        checkpoint_path='checkpoints_moe/final.pt',
        config_path='checkpoints_moe/config.json'
    )
    
    # Example prompts
    prompts = [
        "Once upon a time",
        "The artificial intelligence",
    ]
    
    print("\n" + "="*80)
    print("GENERATING TEXT")
    print("="*80 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print("-" * 40)
        
        # Generate
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
        
        text = decode(
            model, 
            tokenizer, 
            prompt, 
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"\n{prompt}{text}\n")


        