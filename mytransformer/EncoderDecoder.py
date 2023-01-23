import torch 
import torch.nn as nn
from TransformerBlock import PositionalEncoding, TransformerBlock, DecoderBlock


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList([TransformerBlock(
                                                        embed_size,
                                                        heads,
                                                        dropout=dropout,
                                                        forward_expansion=forward_expansion,) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        out = self.dropout((self.word_embedding(x) + self.position_embedding(x)))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out




class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(    
                                    [DecoderBlock(  
                                                embed_size, 
                                                heads, 
                                                forward_expansion, 
                                                dropout, 
                                                device)
                                    for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        x = self.dropout((self.word_embedding(x) + self.position_embedding(x))) # x->query
        for layer in self.layers:
                x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        
        return out