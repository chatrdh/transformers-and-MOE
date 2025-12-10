# Mixture-of-Experts (MoE) Language Model: From Sparse to Adaptive

This repository contains a PyTorch implementation of a Transformer-based Mixture-of-Experts (MoE) Language Model. The project evolves in two distinct phases: building a standard Top-k Sparse MoE (Phase 1) and advancing to a Dynamic Adaptive Compute MoE (Phase 2), inspired by research into token difficulty and computational efficiency.

---

## 🧠 Project Overview

Standard Transformers apply the same computational budget (FLOPs) to every token. However, predicting a function word like *"the"* is computationally trivial compared to predicting a complex concept like *"quantum"*.

This project explores **conditional computation**:
1.  **Phase 1 (Sparse MoE):** Increases model capacity without increasing inference cost by routing tokens to a fixed subset of experts (Top-k).
2.  **Phase 2 (Adaptive MoE):** Dynamically adjusts the computational budget per token based on confidence, allowing the model to "think harder" on difficult tasks.

---

## Phase 1: Sparse Mixture-of-Experts
**Goal:** Scale model parameters while keeping inference FLOPs constant.

### Architecture
* **Routing Mechanism:** Top-k Gating.
* **Experts:** 4 Independent Feed-Forward Networks (SwiGLU).
* **Selection:** Every token is forced to select exactly $k=2$ experts.

### Mathematical Formulation
Given a token representation $x$ and a router weight matrix $W_r$:

1.  **Logits:** $h(x) = x \cdot W_r$
2.  **Probabilities:** $P(x) = \text{Softmax}(h(x))$
3.  **Selection:** Select indices of top $k$ values.
4.  **Output:** Weighted sum of expert outputs.
    $$y = \sum_{i \in \text{Top-}k} P(x)_i \cdot E_i(x)$$

### Expert Collapse
A naive MoE router often "collapses," sending all tokens to a single expert (e.g., Expert 0) while ignoring the rest. To prevent this, we implemented an **Auxiliary Load Balancing Loss**:

$$L_{balance} = N \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
* $f_i$: Fraction of tokens assigned to Expert $i$.
* $P_i$: Average probability assigned to Expert $i$.
This penalizes the model if distribution $f_i$ deviates from uniformity.

---

## Phase 2: Adaptive Compute (Dynamic Routing)
**Goal:** Allocate variable compute based on token difficulty.

Inspired by the paper *"[Harder Tasks Need More Experts: Dynamic Routing in MoE Models" (2024)](https://arxiv.org/abs/2403.07652)"*, we move away from fixed Top-k routing.

### Key Hypothesis
* **Easy Tokens:** Have sharp probability distributions (High Confidence). One expert is sufficient.
* **Hard Tokens:** Have flat distributions (Low Confidence). Multiple experts are needed to resolve ambiguity.

### Architecture Changes
| Component | Phase 1 (Standard) | Phase 2 (Adaptive) |
| :--- | :--- | :--- |
| **Router** | Top-k Router | **Dynamic Confidence Router** |
| **Selection Rule** | Fixed $k$ (e.g., 2) | Cumulative Probability Threshold ($\tau$) |
| **Budget** | Constant per token | Variable (1 to $N$ experts) |
| **Loss** | $L_{CE} + L_{balance}$ | $L_{CE} + L_{balance} + L_{entropy}$ |

### 1. The Dynamic Router
Instead of picking the top 2 experts, we sort the expert probabilities and select the smallest set needed to exceed a confidence threshold $\tau$ (e.g., 0.8).

$$k^* = \min \{ k \mid \sum_{i=1}^{k} p_{(i)} \ge \tau \}$$

* If the model is 90% sure of Expert A, it selects **only** Expert A ($k=1$).
* If the model is unsure (30% A, 30% B, 25% C...), it selects all three ($k=3$).

### 2. The New Objective Function
To prevent the model from becoming "lazy" (activating all experts to minimize loss easily), we introduce a **Sparsity Penalty** via Entropy Minimization.

**Total Loss:**
$$L_{total} = L_{CE} + \beta \cdot L_{balance} + \alpha \cdot L_{dynamic}$$

* **$L_{dynamic}$ (Entropy):** $-\sum P \cdot \log(P)$
    * Minimizing entropy forces the router to be confident (sharp peaks).
    * Sharper peaks mean fewer experts are needed to hit the threshold $\tau$.
    * Therefore, **Minimizing Entropy $\approx$ Minimizing Computational Cost.**

---
