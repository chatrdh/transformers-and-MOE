import torch
from basic_transformer.transformerlm import Linear, softmax, SwiGLU ,RMSNorm ,multihead_self_attention
import einops

class topk(torch.nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()  # Fixed: Added parentheses
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.W_gate = Linear(self.d_model, self.num_experts)

    def forward(self, x: torch.Tensor):
        raw_logits = self.W_gate(x)
        
        # Softmax over expert dimension
        soft_max_logits = softmax(raw_logits, -1)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(soft_max_logits, self.top_k)
        
        # Calculate P_i: average probabilities across all tokens
        # Shape: (batch, seq, num_experts) -> (num_experts,)
        P_i = soft_max_logits.mean(dim=(0, 1))
        
        return top_k_logits, top_k_indices, P_i


class MOELayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int):
        super().__init__()  # Fixed: Added parentheses
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = topk(d_model, num_experts, top_k)
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor):
        # Get routing weights, indices, and average probabilities
        weights, indices, P_i = self.router(x)
        
        # Flatten inputs for processing
        x_flat = einops.rearrange(x, 'b s d -> (b s) d')
        weights_flat = einops.rearrange(weights, '... k -> (...) k')
        indices_flat = einops.rearrange(indices, '... k -> (...) k')
        
        # Calculate f_i (fraction of selections for each expert)
        # Flatten all selected indices
        all_indices = indices_flat.flatten()  # Shape: (total_tokens * top_k,)
        
        # Count how many times each expert was selected
        expert_counts = torch.bincount(
            all_indices, 
            minlength=self.num_experts
        ).float()
        
        # Normalize to get fraction
        f_i = expert_counts / all_indices.numel()  # Shape: (num_experts,)
        
        # Compute auxiliary load balancing loss
        # L_aux = N * dot(f_i, P_i)
        aux_loss = self.num_experts * torch.dot(f_i, P_i)
        
        # Process tokens through experts
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            # Create mask for tokens assigned to this expert
            expert_mask = (indices_flat == i)
            token_mask = expert_mask.any(dim=-1)
            
            # Skip if no tokens assigned
            if not token_mask.any():
                continue
            
            # Extract inputs for this expert
            active_inputs = x_flat[token_mask]
            
            # Forward pass through expert
            expert_out = self.experts[i](active_inputs)
            
            # Get weights for this expert
            active_weights = weights_flat[token_mask][expert_mask[token_mask]]
            active_weights = einops.rearrange(active_weights, 'n -> n 1')
            
            # Accumulate weighted results
            final_output[token_mask] += expert_out * active_weights
        
        # Reshape back to original dimensions
        output = einops.rearrange(final_output, '(b s) d -> b s d', b=x.shape[0])
        
        return output, aux_loss
    


class MOETransformerBlock(torch.nn.Module):
     def __init__(self,d_model : int, num_heads:int , d_ff :int ,max_seq_len :int | None = None,theta : float | None =None):
          super().__init__()
          self.d_model = d_model
          self.num_heads = num_heads
          self.d_ff = d_ff
          self.theta = theta
          self.max_seq_len = max_seq_len
          self.rmsnorm1 = RMSNorm(self.d_model )
          self.rmsnorm2 = RMSNorm(self.d_model )

          self.multihead  = multihead_self_attention(self.d_model,self.num_heads,self.max_seq_len,self.theta)
          self.ffn = MOELayer(d_model,d_ff,4,2)
         

     def forward (self, x:torch.Tensor,token_positions: torch.Tensor | None = None):
          y = x + self.multihead(self.rmsnorm1(x),token_positions)
          result,auxloss = y + self.ffn(self.rmsnorm2(y))

          return result,auxloss
