# Mixture-of-Experts Language Model

A PyTorch implementation exploring conditional computation in Transformers through Mixture-of-Experts.

## Overview

Standard Transformers apply the same compute to every token. This project explores allocating compute dynamically:

- **Phase 1 (Sparse MoE)**: Routes tokens to top-k experts, increasing capacity without proportionally increasing FLOPs
- **Phase 2 (Adaptive MoE)**: Dynamically adjusts experts per token based on confidence—"easy" tokens use fewer experts

## Architecture

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Router | Top-k (fixed k=2) | Confidence threshold (τ) |
| Selection | Always 2 experts | 1 to N experts |
| Loss | CE + Load Balance | CE + Balance + Entropy |

## Quick Start

```bash
# Train Phase 1 (Sparse MoE)
python moe/trainmoe.py

# Train Phase 2 (Dynamic MoE)  
python moe/traindmoe.py

# Analyze expert utilization
python moe/analyze_experts.py
```

## Key Files

- `moe/moelm.py` - MoE model implementations
- `moe/trainmoe.py` - Phase 1 training
- `moe/traindmoe.py` - Phase 2 training
- `basic_transformer/` - Base transformer components

## Reference

Based on ["Harder Tasks Need More Experts: Dynamic Routing in MoE Models"](https://arxiv.org/abs/2403.07652)
