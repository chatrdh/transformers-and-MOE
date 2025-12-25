import torch
import einops
import math
import numpy as np
from numpy.typing import NDArray as npt
from collections.abc import Callable, Iterable
from typing import Optional
import typing
import os

class Linear(torch.nn.Module):

     def __init__(self, in_features, out_features, device=None, dtype=None):
          super().__init__()
          self.in_features = in_features
          self.out_features = out_features
          self.device = device
          self.dtype = dtype

          self.W = torch.nn.Parameter(
            torch.empty(out_features,in_features, device=device, dtype=dtype)
        )
          
          var = 2 / (in_features + out_features)
          torch.nn.init.trunc_normal_(
               self.W,
               mean=0.0,
               std=var**0.5,
               a= - 3 * var**0.5,
               b = 3 * var**0.5
        )

     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
       ## print("W.shape:", self.W.shape)
       ## print("x.shape:", x.shape)
        result = einops.einsum(self.W , x , "out_dim in_dim, ... in_dim -> ... out_dim")  
        return result
     


class Embedding(torch.nn.Module):
     def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
          super().__init__()
          self.num_embeddings = num_embeddings
          self.embedding_dim = embedding_dim
          self.device = device
          self.dtype = dtype
          self.E = torch.nn.Parameter(
          torch.empty(num_embeddings, embedding_dim, device=device, dtype=torch.float32)
                )

          # in-place initializer (note trailing underscore)
          torch.nn.init.trunc_normal_(self.E, mean=0.0, std=1.0, a=-3.0, b=3.0)
         
               
     def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
           
          token_ids = token_ids.to( device=self.E.device)
          return self.E[token_ids]   # shape: (batch, seq, d_model)

        
     

class RMSNorm(torch.nn.Module):
     def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
          super().__init__()
          self.d_model = d_model
          self.eps =eps
          self.device = device
          self.dtype = dtype
          self.gain = torch.nn.Parameter(torch.ones(d_model))


     def forward(self, x: torch.Tensor) -> torch.Tensor :
          in_dtype = x.dtype
          x_float = x.to(torch.float32)
          rms = torch.sqrt(x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps))
          x_norm = x_float / rms                    
          out = x_norm * self.gain               
          return out.to(in_dtype)


class SwiGLU(torch.nn.Module):
     def __init__(self,d_model: int ,d_ff: int, device=None, dtype=None ):
          super().__init__()
          self.d_model = d_model
          self.d_ff = d_ff
          self.device = device
          self.dtype =dtype
          self.W1 = Linear(self.d_model,self.d_ff)
          self.W2 = Linear(self.d_ff,self.d_model)
          self.W3 = Linear(self.d_model,self.d_ff)

     def forward(self, x: torch.Tensor) -> torch.Tensor :
          w1_x = self.W1(x)
          w3_x = self.W3(x)
          silu = w1_x * torch.sigmoid(w1_x)
          #gate = silu * w3_x
          output = self.W2(silu*w3_x)
          return output
     


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
     
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Compute frequency bands: theta^(-2i/d_k) for i = 0, 1, ..., d_k/2 - 1
        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = theta ** (-dim_indices / d_k)  # Shape: (d_k // 2,)
        
        # Precompute angles for all positions: position * frequency
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.einsum('i,j->ij', positions, freqs)  # Shape: (max_seq_len, d_k // 2)
        
        # Compute cos and sin
        cos = torch.cos(angles)  # Shape: (max_seq_len, d_k // 2)
        sin = torch.sin(angles)  # Shape: (max_seq_len, d_k // 2)
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        # Register as buffers so they're moved with the model but not trained
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) specifying token positions
            
        Returns:
            Tensor of the same shape as x with RoPE applied
        """
        # Rearrange x to separate pairs: (..., seq_len, d_k) -> (..., seq_len, d_k//2, 2)
        x = einops.rearrange(x, '... s (d two) -> ... s d two', two=2)
        
        # Index cos and sin by token positions
        # cos/sin: (max_seq_len, d_k//2), token_positions: (..., seq_len)
        cos = self.cos[token_positions]  # (..., seq_len, d_k//2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k//2)
        
        # Apply rotation using einsum
        # x[..., 0] is x0, x[..., 1] is x1
        # Rotation: [x0*cos - x1*sin, x1*cos + x0*sin]
        x0 = x[..., 0]
        x1 = x[..., 1]
        heads = x0.shape[1]
        cos = einops.repeat(cos, 'b s h -> b heads s h', heads=heads)
        sin = einops.repeat(sin, 'b s h -> b heads s h', heads=heads)
        x_rotated = torch.stack([
            x0 * cos - x1 * sin,
            x1 * cos + x0 * sin
        ], dim=-1)
        
        # Rearrange back to original shape: (..., seq_len, d_k//2, 2) -> (..., seq_len, d_k)
        x_rotated = einops.rearrange(x_rotated, '... s d two -> ... s (d two)', two=2)
        
        return x_rotated
    






def softmax(v : torch.Tensor ,i : int ):
     v_max = torch.max(v,dim=i,keepdim=True).values
     v_exp = torch.exp(v - v_max)
     v_sum = v_exp.sum(dim=i, keepdim=True)
     return v_exp/v_sum
     
def scaled_dot_product_attention(
    Q: torch.tensor,
    K: torch.tensor,
    V: torch.tensor,
    mask: torch.tensor,
) -> torch.tensor:
     d_k = Q.shape[-1]
     pre_soft_attn = (einops.einsum(Q,K,"... queries d_k , ... keys d_k -> ... queries keys")/(d_k**0.5))
     pre_soft_attn = torch.where(mask, pre_soft_attn, torch.tensor(float('-inf')))
     soft_attn = softmax(pre_soft_attn,-1)
     attention =soft_attn @ V
     return attention


class multihead_self_attention(torch.nn.Module):
     def __init__(self,d_model: int , num_heads :int, max_seq_len :int | None = None,theta : float | None =None):
          super().__init__()  
          self.d_model = d_model
          self.num_heads = num_heads
          self.d_k = d_model//num_heads
          self.d_v = d_model//num_heads
          #these are Linear projections. Refer to the paper for details
          self.W_q = Linear(d_model , d_model,device=None,dtype=None )
          self.W_k = Linear(d_model , d_model,device=None,dtype=None )
          self.W_v = (Linear(d_model , d_model,device=None,dtype=None ))
          self.W_o = (Linear( d_model,d_model,device=None,dtype=None ))
          # Always create RoPE if parameters are provided
          self.rope = None
          if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)

     def forward(self, x: torch.tensor , token_positions :torch.Tensor| None =None):
          b, s, _ = x.shape
          Q = self.W_q(x)
          K = self.W_k (x)
          V = self.W_v (x)
          #Rearranging for multihead form
          Q = einops.rearrange(Q, 'b s (h d_k) -> b h s d_k', h=self.num_heads)
          K = einops.rearrange(K, 'b s (h d_k) -> b h s d_k', h=self.num_heads)
          V = einops.rearrange(V, 'b s (h d_k) -> b h s d_k', h=self.num_heads)

          #Using RoPE
          #if token_positions is not None:
          if self.rope is not None:
        # Create default positions if not provided
               if token_positions is None:
                    token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
               Q = self.rope(Q, token_positions)
               K = self.rope(K, token_positions)
          
          
            # Use scaled_dot_product_attention  function from above
          mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2], dtype=torch.bool, device=Q.device), diagonal=0)
          attn_output = scaled_dot_product_attention(Q, K, V, mask)
          
        
           # Concatenate heads: (b, h, s, d_k) -> (b, s, d)
          attn_output = einops.rearrange(attn_output, 'b h s d_k -> b s (h d_k)')
        
          # Final linear projection
          output = self.W_o(attn_output)
        
          return output
             

class TransformerBlock(torch.nn.Module):
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
          self.ffn = SwiGLU(self.d_model, self.d_ff)
         

     def forward (self, x:torch.Tensor,token_positions: torch.Tensor | None = None):
          y = x + self.multihead(self.rmsnorm1(x),token_positions)
          result = y + self.ffn(self.rmsnorm2(y))
          return result



class TransformerLM(torch.nn.Module):
     def __init__(self, vocab_size :int, num_layers :int , context_length : int,d_model : int,d_ff:int,num_heads :int,theta : float | None =None):
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
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=theta,max_seq_len=context_length)
            for _ in range(num_layers) ])

          self.norm = RMSNorm(self.d_model)
          self.linear = Linear(self.d_model, self.vocab_size)
          
     def forward(self, in_indices):
          x = self.embedding(in_indices)     

          for block in self.transformer_blocks:
               x = block(x)
          x = self.norm(x)
          logits = self.linear(x)
          return logits     

          


          

          

## Components for training such as loss functions, optimizers, and learning rate schedulers would typically go here

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
     max_logits = torch.max(logits, dim=-1, keepdim=True).values
     logits_new = (logits - max_logits)
     logits_new_sum = torch.sum(torch.exp(logits_new), dim=-1, keepdim=True)
     log_probs = logits_new - torch.log(logits_new_sum)     
     loss = -torch.mean(torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)))
   
     return loss




class AdamW(torch.optim.Optimizer):
     def __init__(self, params, lr ,betas,eps ,weight_decay=0.01 ):
          defaults = {'lr':lr,'betas':betas,'eps':eps,'weight_decay':weight_decay}
          super().__init__(params, defaults)
     
     def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            lamda = group["weight_decay"]

            
        # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                     continue
                state = self.state[p]
                if len(state)==0:
                     state["t"] = 1
                     state["m"] = torch.zeros_like(p.data)
                     state["v"] = torch.zeros_like(p.data)
                
                t = state.get("t")
                m = state.get("m")
                v = state.get("v")
                grad = p.grad.data

                #Update moment 1
                m = beta_1 * m + (1-beta_1)*grad
                #Update moment 2
                v = beta_2 * v + (1-beta_2)*(grad**2)
                alpha_t = (alpha)*(((1-beta_2**t)**0.5)/(1-beta_1**t))
                # Update params
                p.data = p.data - alpha_t * (m/(v**0.5 + eps))
                p.data = p.data - alpha * lamda * p.data
                # Increment iteration number.
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                
          
        return loss



def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c):
     if t< T_w :
          alpha_t = (t/T_w)*alpha_max
     if T_w <= t<= T_c :
          alpha_t = alpha_min +  0.5*(1 + math.cos(((t-T_w)/(T_c-T_w))*torch.pi) )*(alpha_max-alpha_min)
     if t> T_c :
          alpha_t = alpha_min
     return alpha_t




def gradient_clipping(params , max_l2_norm : float,eps  = 1e-6):
     total_norm = 0.0
     for p in params:
          if p.grad is not None:
               total_norm += (p.grad.norm(2) ** 2)
     total_norm = total_norm ** 0.5
     
     # Clip if needed
     clip_coef = max_l2_norm / (total_norm + 1e-6)
     if clip_coef < 1:
          for p in params:
               if p.grad is not None:
                    p.grad.mul_(clip_coef)



def data_loading(x,batch_size: int, context_length: int, device='cpu')-> tuple[torch.Tensor, torch.Tensor]:
     max_start_idx = len(x) - context_length - 1
     start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
     
     # Create batch of input sequences
     inputs = np.array([x[i:i + context_length] for i in start_indices])
     
     # Create batch of target sequences (next tokens)
     targets = np.array([x[i + 1:i + context_length + 1] for i in start_indices])
     
     # Convert to PyTorch tensors and move to device
     inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
     targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)
     
     return inputs_tensor, targets_tensor



def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
     checkpoint = {
         'params_state': model.state_dict(),
         'optim_state': optimizer.state_dict(),
         'iteration': iteration
     }
     torch.save(checkpoint, out)
     

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
     checkpoint = torch.load(src)
     model.load_state_dict(checkpoint['params_state'])
     optimizer.load_state_dict(checkpoint['optim_state'])
     return checkpoint['iteration']




          
          
          

