import torch
import torch.nn as nn
import math
from encoder import Encoder  # 假设您有encoder.py
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, max_len=512, dropout=0.1):
        super(Transformer, self).__init__()
        
        # 编码器和解码器
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        """生成下三角掩码，防止解码器看到未来信息"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt):
        # 生成目标序列掩码
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # 编码器前向传播
        memory = self.encoder(src)
        
        # 解码器前向传播
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 输出投影
        output = self.output_projection(output)
        return output