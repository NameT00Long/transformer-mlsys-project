import torch
import torch.nn as nn
import math
from .layers import PositionalEncoding, MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        for layer in self.layers:
            src = layer(src, mask)
        
        return src