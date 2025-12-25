import torch
import numpy as np
import argparse
import json
import os
import sys
import time
import wandb
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
        self.vocab_size = kwargs.get('vocab_size', 10000)
        self.num_layers = kwargs.get('num_layers', 4)
        self.context_length = kwargs.get('context_length', 256)
        self.d_model = kwargs.get('d_model', 512)
        self.d_ff = kwargs.get('d_ff', 2048)
        self.num_heads = kwargs.get('num_heads', 8)
        self.theta = kwargs.get('theta', 10000.0)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.95)
        self.eps = kwargs.get('eps', 1e-8)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
        # Phase 2: Dynamic MoE Params
        self.num_experts = kwargs.get('num_experts', 4)
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.8) # Tau
        
        # Loss Weights
        self.balance_loss_weight = kwargs.get('balance_loss_weight', 0.01) # Beta
        self.entropy_loss_weight = kwargs.get('entropy_loss_weight', 0.001) # Alpha (Sparsity)
        
        # Training Params
        self.batch_size = kwargs.get('batch_size', 4)
        self.grad_accum_steps = kwargs.get('grad_accum_steps', 1)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.max_steps = kwargs.get('max_steps', 20000)
        self.warmup_steps = kwargs.get('warmup_steps', 2000)
        
        # System
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints_dynamic_moe')
        self.use_wandb = kwargs.get('use_wandb', True)
        self.wandb_project = kwargs.get('wandb_project', 'dynamic-moe-phase2')

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
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    scaler = GradScaler()
    
    # 3. Load Data
    train_dataset = MemoryMappedDataset('data/train.txt', config.context_length)
    val_dataset = MemoryMappedDataset('data/test.txt', config.context_length)
    
    print("Starting training...")
    model.train()
    start_time = time.time()
    
    for step in range(config.max_steps):
        # LR Schedule
        lr = learning_rate_schedule(step, config.learning_rate, 0, config.warmup_steps, config.max_steps)
        for g in optimizer.param_groups: 
            g['lr'] = lr
        
        # Get Batch (Move to GPU async)
        inputs, targets = train_dataset.get_batch(config.batch_size)
        inputs = inputs.to(config.device, non_blocking=True)
        targets = targets.to(config.device, non_blocking=True)
        
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
        gradient_clipping(model.parameters(), config.max_grad_norm)
        
        scaler.step(optimizer)
        scaler.update()
        
        # --- LOGGING ---
        if step % 100 == 0:
            dt = time.time() - start_time
            print(f"Step {step:5d} | "
                  f"Loss: {total_loss.item():.4f} (CE: {ce_loss.item():.4f}) | "
                  f"Bal: {loss_bal.item():.4f} | Ent: {loss_ent.item():.4f} | "
                  f"Exp/Tok: {avg_experts.item():.2f}")
            
            if config.use_wandb:
                wandb.log({
                    "train/total_loss": total_loss.item(),
                    "train/ce_loss": ce_loss.item(),
                    "train/balance_loss": loss_bal.item(),
                    "train/entropy_loss": loss_ent.item(),
                    "train/avg_experts_per_token": avg_experts.item(),
                    "lr": lr,
                    "step": step
                })
        
        # --- VALIDATION ---
        if step % 500 == 0 and step > 0:
            model.eval()
            val_losses = []
            val_ce_losses = []
            val_bal_losses = []
            val_ent_losses = []
            val_avg_experts = []
            
            num_val_batches = 10
            with torch.no_grad():
                for _ in range(num_val_batches):
                    val_inputs, val_targets = val_dataset.get_batch(config.batch_size)
                    val_inputs = val_inputs.to(config.device, non_blocking=True)
                    val_targets = val_targets.to(config.device, non_blocking=True)
                    
                    with autocast():
                        val_logits, val_loss_bal, val_loss_ent, val_avg_exp = model(val_inputs)
                        val_ce_loss = cross_entropy_loss(
                            val_logits.view(-1, val_logits.size(-1)), 
                            val_targets.view(-1)
                        )
                        val_total_loss = val_ce_loss + \
                                       (config.balance_loss_weight * val_loss_bal) + \
                                       (config.entropy_loss_weight * val_loss_ent)
                    
                    val_losses.append(val_total_loss.item())
                    val_ce_losses.append(val_ce_loss.item())
                    val_bal_losses.append(val_loss_bal.item())
                    val_ent_losses.append(val_loss_ent.item())
                    val_avg_experts.append(val_avg_exp.item())
            
            avg_val_loss = np.mean(val_losses)
            avg_val_ce = np.mean(val_ce_losses)
            avg_val_bal = np.mean(val_bal_losses)
            avg_val_ent = np.mean(val_ent_losses)
            avg_val_exp = np.mean(val_avg_experts)
            
            print(f"\n{'='*60}")
            print(f"VALIDATION @ Step {step}")
            print(f"Val Loss: {avg_val_loss:.4f} (CE: {avg_val_ce:.4f})")
            print(f"Val Bal: {avg_val_bal:.4f} | Val Ent: {avg_val_ent:.4f}")
            print(f"Val Exp/Tok: {avg_val_exp:.2f}")
            print(f"{'='*60}\n")
            
            if config.use_wandb:
                wandb.log({
                    "val/total_loss": avg_val_loss,
                    "val/ce_loss": avg_val_ce,
                    "val/balance_loss": avg_val_bal,
                    "val/entropy_loss": avg_val_ent,
                    "val/avg_experts_per_token": avg_val_exp,
                    "step": step
                })
            
            model.train()
        
        # --- CHECKPOINTING ---
        if step % 2000 == 0 and step > 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print("Training Complete.")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(config.checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(model, optimizer, config.max_steps, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    
    if config.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    # Simple CLI for quick testing
    # Use argparse for full implementation
    config = TrainingConfig(
        batch_size=4,
        num_experts=4,
        confidence_threshold=0.8, # Try 0.6 or 0.9 to see how Exp/Tok changes
        entropy_loss_weight=0.001,
        balance_loss_weight=0.01
    )
    train(config)