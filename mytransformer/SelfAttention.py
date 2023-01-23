import torch 
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
    
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, values, keys, query, mask): #input_size -> (N, value_len, embed_size)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size) -> learnable value matrix
        keys = self.keys(keys) 
        queries = self.queries(query) 

        values = values.reshape(N, value_len, self.heads, self.head_dim) # Split the embedding into self.heads different pieces
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e9"))

        attention = torch.softmax(energy / (self.embed_size ** (0.5)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
                                                                        N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out