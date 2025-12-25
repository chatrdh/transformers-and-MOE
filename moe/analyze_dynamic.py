import torch
import os
import sys
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basic_transformer.tokenizer import Tokenizer
from moelm import DynamicMOELM, DynamicKMOELayer

# --- 1. DATA STORAGE ---
router_logs = {}

def get_router_stats(layer_idx):
    """Hooks into the DynamicConfidenceRouter."""
    def hook(module, input, output):
        weights, probs, counts = output
        router_logs[f"layer_{layer_idx}"] = {
            "counts": counts.detach().cpu(),
            "weights": weights.detach().cpu(),
            "probs": probs.detach().cpu()  # Store probabilities too
        }
    return hook

# --- 2. VISUALIZATION UTILS ---
RESET = '\033[0m'

# Cost palette (number of experts)
COST_COLORS = {
    1: '\033[92m',  # Green
    2: '\033[96m',  # Cyan
    3: '\033[93m',  # Yellow
    4: '\033[91m',  # Red
}

# Expert identity palette
EXPERT_COLORS = [
    '\033[91m',  # Red
    '\033[92m',  # Green
    '\033[93m',  # Yellow
    '\033[94m',  # Blue
    '\033[95m',  # Magenta
    '\033[96m',  # Cyan
    '\033[97m',  # White
    '\033[90m',  # Gray
]

def print_cost_view(tokens, counts):
    """Prints text colored by compute cost (difficulty)."""
    print("\n🎨 VIEW: COGNITIVE COST (Green=Easy, Red=Hard)")
    output = ""
    avg_experts = counts.float().mean().item()
    
    for t_str, c in zip(tokens, counts):
        count_val = int(c.item())
        color_key = min(count_val, 4)
        color = COST_COLORS.get(color_key, '\033[97m')
        output += f"{color}{t_str}{RESET}"
    
    print(output)
    print(f"📊 Avg Experts/Token: {avg_experts:.2f}")
    
    # Distribution histogram
    unique, counts_unique = counts.unique(return_counts=True)
    print(f"📈 Distribution: ", end="")
    for u, c_u in zip(unique, counts_unique):
        print(f"{int(u.item())}exp→{int(c_u.item())}tok ", end="")
    print()

def print_expert_view(tokens, weights):
    """Prints text colored by primary expert ID."""
    print("\n🎨 VIEW: PRIMARY EXPERT (Color = Expert ID)")
    output = ""
    expert_usage = torch.zeros(weights.shape[-1])
    
    for t_str, w in zip(tokens, weights):
        primary_expert = w.argmax().item()
        expert_usage[primary_expert] += 1
        color = EXPERT_COLORS[primary_expert % len(EXPERT_COLORS)]
        output += f"{color}{t_str}{RESET}"
    
    print(output)
    
    # Expert usage statistics
    print(f"📊 Expert Usage: ", end="")
    for i, usage in enumerate(expert_usage):
        if usage > 0:
            pct = (usage / len(tokens)) * 100
            print(f"E{i}→{int(usage)}tok({pct:.1f}%) ", end="")
    print()

def print_confidence_view(tokens, weights):
    """Shows confidence/entropy of routing decisions."""
    print("\n🎨 VIEW: ROUTING CONFIDENCE")
    output = ""
    confidences = []
    
    for t_str, w in zip(tokens, weights):
        # Calculate entropy (lower = more confident)
        probs = torch.softmax(w, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        confidences.append(entropy)
        
        # Color by confidence (low entropy = high confidence)
        max_entropy = np.log(len(w))  # Max possible entropy
        norm_entropy = entropy / max_entropy
        
        if norm_entropy < 0.25:
            color = '\033[92m'  # Green (very confident)
        elif norm_entropy < 0.5:
            color = '\033[96m'  # Cyan
        elif norm_entropy < 0.75:
            color = '\033[93m'  # Yellow
        else:
            color = '\033[91m'  # Red (uncertain)
        
        output += f"{color}{t_str}{RESET}"
    
    print(output)
    avg_confidence = np.mean(confidences)
    print(f"📊 Avg Routing Entropy: {avg_confidence:.3f}")

def print_layer_summary(layer_stats):
    """Aggregate statistics across all layers."""
    print("\n" + "="*60)
    print("📋 CROSS-LAYER SUMMARY")
    print("="*60)
    
    for layer_name in sorted(layer_stats.keys()):
        counts = layer_stats[layer_name]["counts"][0]
        weights = layer_stats[layer_name]["weights"][0]
        
        avg_experts = counts.float().mean().item()
        hard_tokens = (counts >= 3).sum().item()
        easy_tokens = (counts == 1).sum().item()
        
        print(f"{layer_name}: Avg={avg_experts:.2f} | Easy={easy_tokens} | Hard={hard_tokens}")

# --- 3. MAIN ANALYSIS ---
def analyze_text(model, tokenizer, text, device='cuda', verbose=True):
    """Analyze routing behavior for given text."""
    router_logs.clear()
    
    # Encode
    tokens = tokenizer.encode(text)
    if len(tokens) == 0:
        print(f"⚠️  Warning: Empty tokens for text '{text}'")
        return
    
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    print(f"\n{'='*60}")
    print(f"🔍 Analyzing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"{'='*60}")
    print(f"📝 Token Count: {len(tokens)}")
    
    # Analyze each layer
    if verbose:
        for layer_name in sorted(router_logs.keys()):
            print(f"\n{'─'*60}")
            print(f"📍 {layer_name.upper()}")
            print(f"{'─'*60}")
            
            counts = router_logs[layer_name]["counts"][0]
            weights = router_logs[layer_name]["weights"][0]
            
            # Show all views
            print_cost_view(token_strs, counts)
            print_expert_view(token_strs, weights)
            print_confidence_view(token_strs, weights)
    
    # Summary
    print_layer_summary(router_logs)
    
    return router_logs

def load_dynamic_model(checkpoint_path, device):
    """Load model with proper checkpoint handling."""
    config = {
        'vocab_size': 10000,
        'num_layers': 4,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 2048,
        'num_heads': 8,
        'num_experts': 4,
        'confidence_threshold': 0.8,
        'theta': 10000.0  # Add theta to enable RoPE
    }
    
    print(f"🔧 Initializing model with config: {config}")
    model = DynamicMOELM(**config)
    
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading weights from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Try different checkpoint formats
            if 'params_state' in checkpoint:
                print("✓ Found 'params_state' key")
                state_dict = checkpoint['params_state']
            elif 'model_state_dict' in checkpoint:
                print("✓ Found 'model_state_dict' key")
                state_dict = checkpoint['model_state_dict']
            else:
                print("✓ Using raw state_dict")
                state_dict = checkpoint
            
            # Clean DDP prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith('module.') else k
                new_state_dict[key] = v
            
            # Load with strict=False to handle any remaining mismatches
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            # Categorize missing/unexpected keys
            rope_missing = [k for k in missing if 'rope' in k]
            other_missing = [k for k in missing if 'rope' not in k]
            rope_unexpected = [k for k in unexpected if 'rope' in k]
            other_unexpected = [k for k in unexpected if 'rope' not in k]
            
            if rope_missing:
                print(f"ℹ️  RoPE buffers missing (will be auto-generated): {len(rope_missing)} keys")
            if other_missing:
                print(f"⚠️  Other missing keys: {other_missing[:3]}...")
            if rope_unexpected:
                print(f"ℹ️  Extra RoPE buffers in checkpoint (ignored): {len(rope_unexpected)} keys")
            if other_unexpected:
                print(f"⚠️  Other unexpected keys: {other_unexpected[:3]}...")
            
            total_params = sum(p.numel() for p in model.parameters())
            loaded_params = total_params - sum(model.state_dict()[k].numel() for k in other_missing if k in model.state_dict())
            load_pct = (loaded_params / total_params) * 100
            
            print(f"✅ Loaded {load_pct:.1f}% of model parameters")
            
        except Exception as e:
            print(f"❌ ERROR loading checkpoint: {e}")
            print("⚠️  Running with RANDOM weights")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("⚠️  Using RANDOM initialized weights")
    
    return model.eval()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load tokenizer
    try:
        tokenizer = Tokenizer.from_files(
            'vocab.json', 
            'merges.json', 
            special_tokens=["<|endoftext|>"]
        )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️  Tokenizer loading failed: {e}")
        print("⚠️  Using fallback tokenizer")
        tokenizer = Tokenizer.from_files('vocab.json', 'merges.json')
    
    # Load model - try fixed checkpoint first, fallback to original
    fixed_checkpoint = ""
    original_checkpoint = "checkpoints_dynamic_moe/final_checkpoint.pt"
    
    if os.path.exists(fixed_checkpoint):
        print(f"🔧 Using fixed checkpoint: {fixed_checkpoint}")
        checkpoint_path = fixed_checkpoint
    elif os.path.exists(original_checkpoint):
        print(f"📦 Using original checkpoint: {original_checkpoint}")
        checkpoint_path = original_checkpoint
    else:
        print(f"⚠️  No checkpoint found, using random weights")
        checkpoint_path = None
    
    model = load_dynamic_model(checkpoint_path, device) if checkpoint_path else DynamicMOELM(**{
        'vocab_size': 10000,
        'num_layers': 4,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 2048,
        'num_heads': 8,
        'num_experts': 4,
        'confidence_threshold': 0.8,
        'theta': 10000.0
    }).eval()
    model.to(device)
    
    # Attach hooks
    print("\n🔗 Attaching hooks to Dynamic Routers...")
    hook_count = 0
    for i, block in enumerate(model.transformer_blocks):
        if hasattr(block, 'ffn') and isinstance(block.ffn, DynamicKMOELayer):
            block.ffn.router.register_forward_hook(get_router_stats(i))
            hook_count += 1
    print(f"✅ Hooked {hook_count} layers")
    
    # --- TEST CASES ---
    test_prompts = [
        ("Simple sentence", "The cat sat on the mat."),
        ("Complex technical", "The quantum mechanical interpretation of gravity is complex."),
        ("Code snippet", "def train_model(x): return x * 2"),
        ("Mixed difficulty", "AI models use transformers with self-attention mechanisms."),
    ]
    
    for name, prompt in test_prompts:
        print(f"\n\n{'#'*60}")
        print(f"# TEST: {name}")
        print(f"{'#'*60}")
        analyze_text(model, tokenizer, prompt, device, verbose=True)
    
    print("\n\n✨ Analysis complete!")