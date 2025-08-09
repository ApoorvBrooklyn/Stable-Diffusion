import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention
class CLIPEmbedding(nn.Module):

    def __init__(self,  n_vocab: int, n_embd: int, n_tokens:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens: torch.Tensor)-> torch.Tensor:
        x = self.token_embedding(tokens)

        x+= self.position_embedding

        return x
    

class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, n_embed: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_heads, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        residue = x

        x = self.layernorm_1(x)
        x = self.attention(x, casual_mask=True)

        x+= residue

        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x) # QuickGELU function. It performs better than the standard AF.
        x = self.linear_2(x)

        x+= residue
        
        return x


class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([ CLIPLayer(12, 768) for i in range(12) ])

        self.layernorms = nn.LayerNorm(768)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        

        output = self.layernorms(state)

        return output