import torch
from torch import nn
from torch.nn import functional as F
from attention import selfAttention

class clipEmbedding(nn.Module):
    def __init__(self, num_vocab: int, num_embed: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(num_vocab, num_embed)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, num_embed)))
    
    def forward(self, tokens):
        x = self.token_embedding(tokens)     #  (batch_size, sequence length, dimension)
        x += self.position_embedding        
        return x

class clipLayer(nn.Module):
    def __init__(self, n_head: int, num_embed: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(num_embed)
        self.attention = selfAttention(n_head, num_embed)
        self.layernorm_2 = nn.LayerNorm(num_embed)
        self.linear_1 = nn.Linear(num_embed, 4 * num_embed)
        self.linear_2 = nn.Linear(4 * num_embed, num_embed)

    def forward(self, x):
        residue = x          #  (batch_size, sequence length, dimension)
        
        x = self.layernorm_1(x)
        x = self.attention(x, mask=True)

        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)        #  (batch_size, sequence length, dimension*4)
    
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation  ##in practice works better
        x = self.linear_2(x)          #  (batch_size, sequence length, dimension)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue                  #  (batch_size, sequence length, dimension)

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = clipEmbedding(49408, 768, 77)          #vocab_size, embed_size, seq_length

        self.layers = nn.ModuleList([clipLayer(12, 768) for i in range(12)]) #num_heads in multihead attention = 12, vocab_size = 768

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)    #  (batch_size, sequence length)
        
        state = self.embedding(tokens)      #  (batch_size, sequence length, dimension), dimension = 768
  
        for layer in self.layers: 
            state = layer(state)
        output = self.layernorm(state)
        return output