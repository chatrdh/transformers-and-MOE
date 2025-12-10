import torch
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basic_transformer.tokenizer import Tokenizer
from decoder import load_model_from_checkpoint

from moelm import MOELayer

# --- 1. THE SPY (Hook Mechanism) ---
# This dictionary will store the routing choices
routing_data = {}

def get_router_choices(layer_idx):
    """
    Creates a hook that captures the output of the router.
    The router returns: (weights, indices, P_i)
    We only care about 'indices'.
    """
    def hook(module, input, output):
        # output[1] contains the 'indices' (Batch, Seq, Top_K)
        # We detach it from the graph and move to CPU
        indices = output[1].detach().cpu()
        routing_data[f"layer_{layer_idx}"] = indices
    return hook

# --- 2. TERMINAL COLORING UTILS ---
# ANSI escape codes for coloring text in terminal
COLORS = [
    '\033[91m', # Red
    '\033[92m', # Green
    '\033[93m', # Yellow
    '\033[94m', # Blue
    '\033[95m', # Magenta
    '\033[96m', # Cyan
    '\033[97m', # White
    '\033[90m', # Grey
]
RESET = '\033[0m'

def print_colored_text(tokens, expert_indices):
    """
    Prints tokens colored by their primary expert choice.
    expert_indices shape: (Seq_Len, Top_K)
    """
    output_str = ""
    for token_str, experts in zip(tokens, expert_indices):
        # We only color based on the Primary Expert (Top-1)
        primary_expert = experts[0].item()
        
        # Pick a color (modulo 8 in case you have >8 experts)
        color = COLORS[primary_expert % len(COLORS)]
        
        output_str += f"{color}{token_str}{RESET}"
    print(output_str)
    print(RESET)

# --- 3. MAIN ANALYSIS ---
def analyze_text(model, tokenizer, text, device='cuda'):
    # Clear previous data
    routing_data.clear()
    
    # Prepare input
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device) # (1, Seq)
    
    # Run model (Forward pass only)
    with torch.no_grad():
        model(input_tensor)
        
    # Convert token IDs back to strings for visualization
    # (We assume the tokenizer has a decode method, or we decode one by one)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    print(f"\nAnalyzing: '{text[:50]}...'")
    print("-" * 60)
    
    # Visualize Layer by Layer
    for layer_name, indices in routing_data.items():
        # indices shape is (Batch=1, Seq, TopK) -> flatten batch
        layer_indices = indices[0] 
        
        print(f"\n{layer_name.upper()} (Color = Expert ID):")
        print_colored_text(token_strs, layer_indices)
        
        # Optional: Print distribution stats
        counts = torch.bincount(layer_indices.flatten(), minlength=8).float()
        total = counts.sum()
        dist = (counts / total).numpy()
        print(f"Expert Usage: {np.round(dist, 2)}")

def main():
    # Load Model
    checkpoint_path = 'checkpoints_moe/final.pt' # Check your path
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found! Please check path.")
        return

    model, tokenizer, config = load_model_from_checkpoint(checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # REGISTER HOOKS
    # We attach our spy to every transformer block's router
    print("\nAttaching hooks to routers...")
    for i, block in enumerate(model.transformer_blocks):
            # Safety Check: Ensure this block actually has an 'ffn' that is an MOELayer
            if hasattr(block, 'ffn') and isinstance(block.ffn, MOELayer):
                # Now Python knows it's an MOELayer, so .router is valid
                block.ffn.router.register_forward_hook(get_router_choices(i))
                print(f" - Hooked Layer {i}")
            else:
                print(f" ! Warning: Layer {i} does not look like a standard MOE block. Skipped.")
        
    # TEST CASES
    # 1. English Prose
    analyze_text(model, tokenizer, "The quick brown fox jumps over the lazy dog.", device)
    
    # 2. Code / Technical (If you trained on code, this will look different)
    analyze_text(model, tokenizer, "def train_model(x): return x * 2", device)
    
    # 3. Repetitive pattern
    analyze_text(model, tokenizer, "1 2 3 4 1 2 3 4 1 2 3 4", device)

if __name__ == "__main__":
    main()