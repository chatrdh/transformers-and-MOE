import torch
import numpy as np
import argparse
import json
import os
import sys
import time
from torch.cuda.amp import GradScaler, autocast

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basic_transformer.transformerlm import AdamW, cross_entropy_loss, gradient_clipping, learning_rate_schedule, save_checkpoint, load_checkpoint
# CHANGED: Import the Dynamic Model
from moelm import DynamicMOELM
from basic_transformer.tokenizer import Tokenizer

class TrainingConfig:
    def __init__(self, **kwargs):
        # Model Params
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.num_layers = kwargs.get('num_layers', 4)
        self.context_length = kwargs.get('context_length', 256)
        self.d_model = kwargs.get('d_model', 512)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.num_heads = kwargs.get('num_heads', 8)
        self.theta = kwargs.get('theta', 10000.0)
        
        # Phase 2: Dynamic MoE Params
        self.num_experts = kwargs.get('num_experts', 8)
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.8) # Tau
        
        # Loss Weights
        self.balance_loss_weight = kwargs.get('balance_loss_weight', 0.01) # Beta
        self.entropy_loss_weight = kwargs.get('entropy_loss_weight', 0.001) # Alpha (Sparsity)
        
        # Training Params
        self.batch_size = kwargs.get('batch_size', 8)
        self.grad_accum_steps = kwargs.get('grad_accum_steps', 1)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.max_steps = kwargs.get('max_steps', 20000)
        self.warmup_steps = kwargs.get('warmup_steps', 2000)
        
        # System
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints_dynamic_moe')
        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'dynamic-moe-phase2')

# ... (MemoryMappedDataset class remains the same as before) ...
class MemoryMappedDataset:
    def __init__(self, data_path: str, context_length: int):
        self.data_path = data_path
        self.context_length = context_length
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.length = len(self.data)
    
    def get_batch(self, batch_size: int):
        max_start_idx = self.length - self.context_length - 1
        start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
        inputs = np.array([self.data[i:i + self.context_length] for i in start_indices])
        targets = np.array([self.data[i + 1:i + self.context_length + 1] for i in start_indices])
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def train(config: TrainingConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=config.__dict__)

    # 1. Init Model
    print(f"\nInitializing Dynamic MoE (Threshold={config.confidence_threshold})...")
    model = DynamicMOELM(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        context_length=config.context_length,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        num_experts=config.num_experts,
        confidence_threshold=config.confidence_threshold, # Dynamic!
        theta=config.theta
    ).to(config.device)
    
    # 2. Init Optimizer & Scaler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    
    # 3. Load Data
    train_dataset = MemoryMappedDataset('data/train.bin', config.context_length) # Update path
    
    print("Starting training...")
    model.train()
    start_time = time.time()
    
    for step in range(config.max_steps):
        # LR Schedule
        lr = learning_rate_schedule(step, config.learning_rate, 0, config.warmup_steps, config.max_steps)
        for g in optimizer.param_groups: g['lr'] = lr
        
        # Get Batch (Move to GPU async)
        inputs, targets = train_dataset.get_batch(config.batch_size)
        inputs, targets = inputs.to(config.device, non_blocking=True), targets.to(config.device, non_blocking=True)
        
        # --- FORWARD PASS (Mixed Precision) ---
        with autocast():
            # UNPACK 4 VALUES
            logits, loss_bal, loss_ent, avg_experts = model(inputs)
            
            # Main Loss
            ce_loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # TOTAL OBJECTIVE
            # L_total = CE + (Beta * Balance) + (Alpha * Entropy)
            total_loss = ce_loss + \
                         (config.balance_loss_weight * loss_bal) + \
                         (config.entropy_loss_weight * loss_ent)

        # --- BACKWARD PASS ---
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        gradient_clipping(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # --- LOGGING ---
        if step % 100 == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | "
                  f"Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}) | "
                  f"Bal: {loss_bal.item():.4f} | Ent: {loss_ent.item():.4f} | "
                  f"Exp/Tok: {avg_experts.item():.2f}") # <-- The Critical Stat
            
            if config.use_wandb:
                wandb.log({
                    "total_loss": total_loss.item(),
                    "ce_loss": ce_loss.item(),
                    "balance_loss": loss_bal.item(),
                    "entropy_loss": loss_ent.item(),
                    "avg_experts_per_token": avg_experts.item(),
                    "lr": lr
                })

    print("Training Complete.")

if __name__ == '__main__':
    # Simple CLI for quick testing
    # Use argparse for full implementation
    config = TrainingConfig(
        batch_size=8,
        num_experts=8,
        confidence_threshold=0.8, # Try 0.6 or 0.9 to see how Exp/Tok changes
        entropy_loss_weight=0.001,
        balance_loss_weight=0.01
    )
    train(config)