import torch
import torch.nn as nn
import math
from .layers import PositionalEncoding, MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 三个子层：掩码自注意力、编码器-解码器注意力、前馈网络
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # 三个归一化层和dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        # 第一个子层：掩码多头自注意力
        attn1 = self.masked_self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn1))
        
        # 第二个子层：编码器-解码器注意力
        attn2 = self.enc_dec_attn(tgt, memory, memory, mask=src_mask)
        tgt = self.norm2(tgt + self.dropout2(attn2))
        
        # 第三个子层：前馈网络
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        
        return tgt

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, src_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, src_mask)
        
        return tgt