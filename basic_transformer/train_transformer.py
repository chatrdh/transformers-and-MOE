import torch
import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import Optional
import time
from datetime import datetime

# Import your model components
from transformerlm import (
    TransformerLM,
    AdamW,
    cross_entropy_loss,
    gradient_clipping,
    learning_rate_schedule,
    save_checkpoint,
    load_checkpoint
)
from tokenizer import Tokenizer


class TrainingConfig:
    """Configuration class for training hyperparameters."""
    def __init__(self, **kwargs):
        # Model hyperparameters
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.num_layers = kwargs.get('num_layers', 12)
        self.context_length = kwargs.get('context_length', 1024)
        self.d_model = kwargs.get('d_model', 768)
        self.d_ff = kwargs.get('d_ff', 3072)
        self.num_heads = kwargs.get('num_heads', 12)
        self.theta = kwargs.get('theta', 10000.0)
        
        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 8)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.alpha_min = kwargs.get('alpha_min', 3e-5)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.95)
        self.eps = kwargs.get('eps', 1e-8)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
        # Learning rate schedule
        self.warmup_steps = kwargs.get('warmup_steps', 2000)
        self.max_steps = kwargs.get('max_steps', 100000)
        
        # Training control
        self.eval_interval = kwargs.get('eval_interval', 500)
        self.log_interval = kwargs.get('log_interval', 100)
        self.save_interval = kwargs.get('save_interval', 5000)
        self.eval_steps = kwargs.get('eval_steps', 100)
        
        # Paths
        self.train_data_path = kwargs.get('train_data_path', 'data/train.bin')
        self.val_data_path = kwargs.get('val_data_path', 'data/val.bin')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.resume_from = kwargs.get('resume_from', None)
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Weights & Biases
        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'transformer-lm')
        self.wandb_run_name = kwargs.get('wandb_run_name', None)
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class MemoryMappedDataset:
    """Memory-efficient dataset using np.memmap for large files."""
    def __init__(self, data_path: str, context_length: int):
        self.data_path = data_path
        self.context_length = context_length
        
        # Memory-map the data file
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.length = len(self.data)
        
        print(f"Loaded dataset from {data_path}")
        print(f"Dataset size: {self.length:,} tokens")
    
    def get_batch(self, batch_size: int, device: str = 'cpu'):
        """Generate a random batch of data."""
        max_start_idx = self.length - self.context_length - 1
        start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
        
        # Create batch of input sequences
        inputs = np.array([self.data[i:i + self.context_length] for i in start_indices])
        
        # Create batch of target sequences (next tokens)
        targets = np.array([self.data[i + 1:i + self.context_length + 1] for i in start_indices])
        
        # Convert to PyTorch tensors and move to device
        inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
        targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        
        return inputs_tensor, targets_tensor


def estimate_loss(model, dataset, config, num_batches=None):
    """Estimate loss on the dataset."""
    model.eval()
    
    if num_batches is None:
        num_batches = config.eval_steps
    
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = dataset.get_batch(config.batch_size, config.device)
            logits = model(inputs)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = cross_entropy_loss(logits_flat, targets_flat)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def train(config: TrainingConfig):
    """Main training loop."""
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save config
    config.to_json(os.path.join(config.checkpoint_dir, 'config.json'))
    
    # Initialize Weights & Biases if requested
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
        except ImportError:
            print("Warning: wandb not installed. Logging to console only.")
            config.use_wandb = False
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MemoryMappedDataset(config.train_data_path, config.context_length)
    val_dataset = MemoryMappedDataset(config.val_data_path, config.context_length)
    
    # Initialize model
    print("\nInitializing model...")
    model = TransformerLM(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        context_length=config.context_length,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        theta=config.theta
    ).to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        print(f"Resumed from step {start_step}")
    
    # Training loop
    print(f"\nStarting training on {config.device}...")
    print(f"Training from step {start_step} to {config.max_steps}")
    print("-" * 80)
    
    model.train()
    start_time = time.time()
    
    for step in range(start_step, config.max_steps):
        # Get learning rate for this step
        lr = learning_rate_schedule(
            t=step,
            alpha_max=config.learning_rate,
            alpha_min=config.alpha_min,
            T_w=config.warmup_steps,
            T_c=config.max_steps
        )
        
        # Update learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        inputs, targets = train_dataset.get_batch(config.batch_size, config.device)
        
        # Forward pass
        logits = model(inputs)
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = cross_entropy_loss(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        # Logging
        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step - start_step + 1) * config.batch_size * config.context_length / elapsed
            
            log_msg = (
                f"Step {step:6d} | Loss: {loss.item():.4f} | "
                f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}"
            )
            print(log_msg)
            
            if config.use_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                    'step': step
                })
        
        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            print("\nEvaluating...")
            train_loss = estimate_loss(model, train_dataset, config)
            val_loss = estimate_loss(model, val_dataset, config)
            
            print(f"Step {step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("-" * 80)
            
            if config.use_wandb:
                wandb.log({
                    'eval/train_loss': train_loss,
                    'eval/val_loss': val_loss,
                    'step': step
                })
        
        # Save checkpoint
        if step % config.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_step_{step}.pt'
            )
            print(f"\nSaving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            
            # Also save as latest
            latest_path = os.path.join(config.checkpoint_dir, 'latest.pt')
            save_checkpoint(model, optimizer, step, latest_path)
    
    # Final checkpoint
    final_path = os.path.join(config.checkpoint_dir, 'final.pt')
    print(f"\nSaving final checkpoint to {final_path}")
    save_checkpoint(model, optimizer, config.max_steps, final_path)
    
    # Final evaluation
    print("\nFinal evaluation...")
    train_loss = estimate_loss(model, train_dataset, config)
    val_loss = estimate_loss(model, val_dataset, config)
    print(f"Final | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if config.use_wandb:
        wandb.finish()
    
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    
    # Config file
    parser.add_argument('--config', type=str, help='Path to JSON config file')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--context_length', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--theta', type=float, default=10000.0)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha_min', type=float, default=3e-5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Learning rate schedule
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    
    # Training control
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=100)
    
    # Paths
    parser.add_argument('--train_data_path', type=str, default='data/train.bin')
    parser.add_argument('--val_data_path', type=str, default='data/val.bin')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Weights & Biases
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='transformer-lm')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config from file if provided, otherwise use command line args
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        config = TrainingConfig(**vars(args))
    
    # Print configuration
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    for key, value in config.__dict__.items():
        print(f"{key:25s}: {value}")
    print("=" * 80)
    
    # Start training
    train(config)


if __name__ == '__main__':
    main()