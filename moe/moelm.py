import torch
from basic_transformer.transformerlm import Linear,Embedding, softmax, SwiGLU ,RMSNorm ,multihead_self_attention
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
        P_i = einops.reduce(soft_max_logits, 'B S E -> E', 'mean')
        
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
        x_flat = einops.rearrange(x, 'b s d -> (b s) d') #(output = (total_tokens,d_model))
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
     def __init__(self,d_model : int, num_heads:int , d_ff :int ,num_experts: int, top_k: int,max_seq_len :int | None = None,theta : float | None =None):
          super().__init__()
          self.d_model = d_model
          self.num_heads = num_heads
          self.d_ff = d_ff
          self.theta = theta
          self.max_seq_len = max_seq_len
          self.rmsnorm1 = RMSNorm(self.d_model )
          self.rmsnorm2 = RMSNorm(self.d_model )

          self.multihead  = multihead_self_attention(self.d_model,self.num_heads,self.max_seq_len,self.theta)
          self.ffn = MOELayer(d_model,d_ff,num_experts,top_k)
         

     def forward (self, x:torch.Tensor,token_positions: torch.Tensor | None = None):
          y = x + self.multihead(self.rmsnorm1(x),token_positions)
          moe_out, aux_loss = self.ffn(self.rmsnorm2(y))
          result = y + moe_out

          return result,aux_loss



class MoeLM(torch.nn.Module):
     def __init__(self, vocab_size :int, num_layers :int , context_length : int,d_model : int,d_ff:int,num_heads :int,num_experts: int,top_k: int,theta : float | None =None):
          super().__init__()
          self.vocab_size = vocab_size
          self.num_layers = num_layers
          self.context_length = context_length
          self.d_model = d_model
          self.num_heads = num_heads
          self.d_ff = d_ff
          self.theta = theta
          self.embedding = Embedding(self.vocab_size,d_model)
          self.transformer_blocks = torch.nn.ModuleList([
            MOETransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,num_experts =num_experts,top_k=top_k, theta=theta,max_seq_len=context_length)
            for _ in range(num_layers) ])

          self.norm = RMSNorm(self.d_model)
          self.linear = Linear(self.d_model, self.vocab_size)
          
     def forward(self, in_indices):
          x = self.embedding(in_indices)     

          for block in self.transformer_blocks:
               x, layer_loss = block(x)
               total_aux_loss += layer_loss
          x = self.norm(x)
          logits = self.linear(x)
          return logits,total_aux_loss  
     


     



### Dynamic -k Routing Code ....

class DynamicK(torch.nn.Module):
    def __init__ (self,d_model :int,num_experts:int, confidence_threshold :float):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.confidence_threshold = confidence_threshold
        self.gate = Linear(d_model,num_experts)

    def forward(self,x:torch.Tensor):
        logits = self.gate(x)
        probs = softmax(logits,-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
         #Cumulative Sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # We select experts until the sum exceeds the threshold.
        # CRITICAL LOGIC: We want to include the first expert that crosses the line.
        # We do this by checking if the *cumulative sum excluding the current expert* # is less than the threshold.
        # shifted_cumsum: The sum of probabilities *before* adding the current expert.
        shifted_cumsum = cumulative_probs - sorted_probs
        
        # Create boolean mask
        # Example (Threshold 0.8):
        # Shifted: [0.0, 0.6, 0.85]
        # Check:   [T,   T,    F  ] -> Keeps top 2 experts
        is_active = shifted_cumsum < self.confidence_threshold
        
        # 3. Safety Net (Top-1 Guarantee)
        # We enforce that the highest-ranked expert is ALWAYS active.
        # This handles the edge case where expert_1 > threshold (e.g., 0.9 > 0.8).
        # Without this, the mask might be all False for very confident tokens.
        is_active[..., 0] = True
# --- Step 5: Re-normalization ---
        
        # Zero out the non-selected experts
        # We multiply the sorted probabilities by the boolean mask (True=1, False=0)
        active_probs = sorted_probs * is_active.float()
        
        # Re-normalize so the selected experts sum to 1.0
        # active_probs.sum(dim=-1) gives the total mass of selected experts
        total_mass = active_probs.sum(dim=-1, keepdim=True)
        
        # Safety: Add epsilon to avoid division by zero (though Top-1 guarantee makes this rare)
        active_weights = active_probs / (total_mass + 1e-6)
        
        
        #  Un-sorting (Scatter back to original indices) 
        
        # We currently have weights in "Rank Order" (Best, 2nd Best...)
        # We need to put them back into "Expert Order" (Expert 0, Expert 1...)
        
        # Initialize an empty tensor of zeros with the original shape
        # Shape: (Batch, Seq, Num_Experts)
        routing_weights = torch.zeros_like(probs)
        
        # Scatter the values back to their original positions
        # dim=-1: Scatter along the expert dimension
        # index=sorted_indices: The map of where each rank came from
        # src=active_weights: The values to place
        routing_weights.scatter_(dim=-1, index=sorted_indices, src=active_weights)
        
        
        # --- Final Returns ---
        
        # 1. routing_weights: Sparse weights for the MoE layer (0.0 for inactive experts)
        # 2. probs: Full raw probabilities (needed for Entropy Loss and Load Balance Loss)
        # 3. active_count: How many experts were active (needed for Logging/Analysis)
        active_count = is_active.sum(dim=-1)
        
        return routing_weights, probs, active_count




class DynamicKMOELayer(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int, confidence_threshold: float = 0.8,):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. The New Dynamic Router
        # This handles the "Cumulative Confidence" logic
        self.router = DynamicK(d_model, num_experts, confidence_threshold)
        
        # 2. The Experts
        # Independent neural networks. 
        # (Make sure SwiGLU is defined/imported from your transformerlm file)
        self.experts = torch.nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self,x : torch.Tensor):
        batch_size, seq_len, _ = x.shape
        

        #get the router
        weights, probs, active_counts = self.router(x)
        # --- 2. Sparse Dispatch ---
        final_output = torch.zeros_like(x)
        
        # Flatten for easy indexing: (Total_Tokens, d_model)
        x_flat = einops.rearrange(x, 'b s d -> (b s) d')
        weights_flat = einops.rearrange(weights, '... k -> (...) k')
        for i in range(self.num_experts):
            # Check which tokens selected Expert i
            # We look for non-zero weights
            active_mask = weights_flat[:, i] > 0.0
            
            # OPTIMIZATION: Early Exit
            # If the mask is all False (no token picked this expert), skip it.
            if not active_mask.any():
                continue
            
            # A. Gather the specific tokens that need this expert
            active_inputs = x_flat[active_mask]
            
            # B. Process them
            expert_out = self.experts[i](active_inputs)
            
            # C. Weight the output by the router's confidence
            # (e.g., if weight is 0.5, we only take half the signal)
            scaling_factor = einops.rearrange(weights_flat[active_mask, i], 'n -> n 1')
            weighted_out = expert_out * scaling_factor
            
            # D. Scatter (Add) back to the final output
            final_output.view(-1, x.shape[-1])[active_mask] += weighted_out


            # --- 3. Loss Calculation ---
        
            # A. Load Balancing Loss
            # Get fraction of tokens assigned to each expert
            # (Convert boolean mask to float, sum up, divide by total tokens)
            tokens_per_expert = (weights_flat > 0).float().sum(dim=0)
            f_i = tokens_per_expert / (batch_size * seq_len)
            
            # Get average probability per expert
            P_i = probs.view(-1, self.num_experts).mean(dim=0)
            
            # Standard auxiliary loss formula
            loss_balance = self.num_experts * torch.sum(f_i * P_i)
            
            # B. Dynamic Sparsity Loss (Entropy)
            # We calculate entropy per token, then average over the batch
            # Add 1e-6 to avoid log(0) error
            entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
            loss_entropy = entropy_per_token.mean()

            # Return everything needed
            return final_output, loss_balance, loss_entropy, active_counts


class DynamicMOETransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 num_experts: int, confidence_threshold: float = 0.8,
                 max_seq_len: int | None = None, theta: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Norms (Same as before)
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)

        # Attention (Same as before)
        self.multihead = multihead_self_attention(d_model, num_heads, max_seq_len, theta)
        
        # MOE Layer (UPDATED)
        # We now initialize the DynamicMOELayer
        self.ffn = DynamicKMOELayer(d_model, d_ff, num_experts, confidence_threshold)
    
    def forward(self, x: torch.Tensor):
        # --- 1. Attention Sub-layer ---
        # (Standard Residual Connection)
        h = self.multihead(self.rmsnorm1(x))
        x = x + h 

        # --- 2. Dynamic MoE Sub-layer ---
        # Apply Norm
        normed_x = self.rmsnorm2(x)
        
        # UNPACKING 4 VALUES:
        # moe_out: The tensor output
        # l_bal:   Load balancing loss (scalar)
        # l_ent:   Entropy/Sparsity loss (scalar)
        # counts:  Tensor of active experts per token (for analysis)
        moe_out, l_bal, l_ent, counts = self.ffn(normed_x)
        
        # Residual Add (only the tensor)
        result = x + moe_out

        # Return EVERYTHING
        return result, l_bal, l_ent, counts
    

class DynamicMOELM(torch.nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, context_length: int, 
                 d_model: int, d_ff: int, num_heads: int, 
                 num_experts: int, confidence_threshold: float = 0.8, 
                 theta: float | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        
        self.embedding = Embedding(vocab_size, d_model)
        
        # Stack of Dynamic Blocks
        self.transformer_blocks = torch.nn.ModuleList([
            DynamicMOETransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                num_experts=num_experts, 
                confidence_threshold=confidence_threshold, # Dynamic Parameter
                theta=theta, 
                max_seq_len=context_length
            )
            for _ in range(num_layers) 
        ])

        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)


    def forward(self, in_indices):
        # x shape: (Batch, Seq_Len)
        x = self.embedding(in_indices)     

        # Initialize Accumulators
        total_balance_loss = 0.0
        total_entropy_loss = 0.0
        total_active_experts = 0.0 # Just for tracking stats

        for block in self.transformer_blocks:
            # UNPACK 4 VALUES
            x, l_bal, l_ent, counts = block(x)
            
            # Accumulate
            total_balance_loss += l_bal
            total_entropy_loss += l_ent
            
            # For logging: Track average active experts
            # counts is (Batch, Seq), so we take the mean
            total_active_experts += counts.float().mean()
        
        # Final Norm and Projection
        x = self.norm(x)
        logits = self.linear(x)
        
        # Average the expert count across layers for a global view
        avg_active_experts = total_active_experts / len(self.transformer_blocks)
        
        # Return Logits + The 2 Loss Components + The Stat
        return logits, total_balance_loss, total_entropy_loss, avg_active_experts
    

    