import torch
from basic_transformer.transformerlm import Linear,softmax,SwiGLU
import einops

class topk(torch.nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.W_gate = Linear(self.d_model,self.num_experts)

    def forward(self,x:torch.Tensor):
        raw_logits = self.W_gate(x)
        soft_max_logits = softmax(raw_logits,-1)
        top_k_logits ,top_k_indices = torch.topk(soft_max_logits,self.top_k)
        return top_k_logits, top_k_indices


class MOELayer(torch.nn.Module):
    def __init__(self,d_model: int, d_ff: int, num_experts: int, top_k: int ):
        super().__init()
        self.router = top_k(d_model,num_experts,top_k)
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(num_experts)
        ])
    def forward(self, x: torch.Tensor):
        weights , indices = self.router(x)
        x_flat = einops.rearrange(x, 'b s d -> (b s) d')
        weights_flat = einops.rearrange(weights, '... k -> (...) k')
        indices_flat = einops.rearrange(indices, '... k -> (...) k')


        final_output = torch.zeros_like(x_flat)

        for i in range(self.num_experts):
            # A token might select expert 'i' as its 1st choice or 2nd choice.
            # We create a mask where (indices == i) is True.
            # Shape: (total_tokens, top_k)
            expert_mask = (indices_flat == i)
            
            # Collapse to see which tokens use this expert *at all*
            token_mask = expert_mask.any(dim=-1)  # Shape: (total_tokens,)
            
            # Optimization: Skip expert if no tokens assigned
            if not token_mask.any():
                continue
            
            # Extract inputs for this expert
            active_inputs = x_flat[token_mask]
            
            # Forward pass through the specific expert
            expert_out = self.experts[i](active_inputs)
            
            # Get the weight for this expert for these tokens.
            # We select the specific weight from the top_k weights using the mask.
            active_weights = weights_flat[token_mask][expert_mask[token_mask]]
            
            # Reshape for broadcasting using rearrange
            active_weights = einops.rearrange(active_weights, 'n -> n 1')
            
            # Accumulate results
            final_output[token_mask] += expert_out * active_weights
        
        # Reshape back to original dimensions using rearrange
        return einops.rearrange(final_output, '(b s) d -> b s d')
            


