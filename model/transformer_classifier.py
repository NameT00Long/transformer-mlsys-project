import torch
import torch.nn as nn
import math
from .encoder import Encoder

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, d_ff=512, num_layers=4, 
                 num_classes=2, max_len=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # 使用自己实现的编码器
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 编码器前向传播
        encoded = self.encoder(src)
        
        # 平均池化获取句子表示
        pooled = torch.mean(encoded, dim=1)
        
        # 分类
        output = self.dropout(pooled)
        output = self.classifier(output)
        return output