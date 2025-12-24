import torch
import torch.nn as nn
import math
from model.layers import PositionalEncoding, MultiHeadAttention


class FeedForward(nn.Module):
    """
    前馈网络：两层线性变换 + ReLU
    """
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    单个 Transformer 编码器层：
    MultiHeadAttention → Add & Norm → FeedForward → Add & Norm
    """
    def __init__(self, d_model, num_heads, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        
        # 两个 LayerNorm（标准做法）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        src: [batch_size, seq_len, d_model]
        """
        # 自注意力 + 残差连接 + LayerNorm
        attn_output = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

class TransformoerEncoder(nn.Module):
    """
    堆叠 N 个 Transformer 编码器层
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformoerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output

class TransformerClassifier(nn.Module):
    """
    完整的文本分类模型
    """
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, 
                 dim_feedforward=512, num_classes=2, max_len=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        
        # 1. 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 3. Transformer 编码器（手动堆叠多层）
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

        # 5. Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        """
        src: [batch_size, seq_len]
        输出：[batch_size, num_classes]
        """
        # 获取词向量并缩放
        src = self.embedding(src) * math.sqrt(self.d_model)

        # 添加位置编码
        src = self.pos_encoding(src)

        # 通过所有编码层
        for layers in self.encoder_layers:
            src = layers(src)

        # 全局平均池化
        # 序列维度求平均
        src = torch.mean(src, dim=1)

        src = self.dropout(src)
        output = self.classifier(src)

        return output