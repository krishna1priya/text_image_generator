import torch
from torch import nn
from torch.nn import functional as F
import math

class selfAttention(nn.Module):
    def __init__(self, num_heads, vec_size):
        super().__init__()
        self.vec_size = vec_size        #embedding dimension
        self.num_heads = num_heads      #number of heads
        self.head_dim = vec_size // num_heads
        self.qkv_layer = nn.Linear(vec_size , 3 * vec_size, bias = True)
        self.linear_layer = nn.Linear(vec_size, vec_size, bias = True)

    def forward(self, x, mask=False):
        batch_size, sequence_length, vec_size = x.size()
        qkv = self.qkv_layer(x)

        req_shape = (batch_size, sequence_length, self.num_heads, self.head_dim) 
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(req_shape).transpose(1, 2)
        k = k.view(req_shape).transpose(1, 2)
        v = v.view(req_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        
        if mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 
        output = weight @ v
        output = output.transpose(1, 2) 
        output = output.reshape(x.size()) 
        output = self.linear_layer(output) 
    
        return output

class crossAttention(nn.Module):
    def __init__(self, num_heads, vec_size, d_cross):
        super().__init__()
        self.q_layer   = nn.Linear(vec_size, vec_size, bias=True)
        self.k_layer   = nn.Linear(d_cross, vec_size, bias=True)
        self.v_layer   = nn.Linear(d_cross, vec_size, bias=True)
        self.out_proj = nn.Linear(vec_size, vec_size, bias=True)
        self.num_heads = num_heads
        self.head_dim = vec_size // num_heads
    
    def forward(self, x, y):        #x - latent (batch_size, seq_len_q, vec_size), y - prompt(batch_size, seq_len_kv, vec_size)

        input_shape = x.shape
        batch_size, sequence_length, vec_size = input_shape
        req_shape = (batch_size, -1, self.num_heads, self.head_dim)
             
        q = self.q_layer(x)
        k = self.k_layer(y)
        v = self.v_layer(y)

        q = q.view(req_shape).transpose(1, 2)       # (batch_size, num_heads, seq_len_q, vec_size/num_heads)
        k = k.view(req_shape).transpose(1, 2)        # (batch_size, num_heads, seq_len_kv, vec_size/num_heads)
        v = v.view(req_shape).transpose(1, 2)        # (batch_size, num_heads, seq_len_kv, vec_size/num_heads)
        
        # (batch_size, num_heads, seq_len_q, vec_size/num_heads) *  (batch_size, num_heads, vec_size/num_heads, seq_len_kv) = (batch_size, num_heads, seq_len_q, seq_len_kv) 
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        # (batch_size, num_heads, seq_len_q, seq_len_kv) * (batch_size, num_heads, seq_len_kv, vec_size/num_heads) -> (batch_size, num_heads, seq_len_q,vec_size/num_heads)
        output = weight @ v
        output = output.transpose(1, 2).contiguous()       # (batch_size, seq_len_q,  num_heads, vec_size/num_heads)
        output = output.view(input_shape)                   # (batch_size, seq_len_q, vec_size)
        output = self.out_proj(output)
        return output