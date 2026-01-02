import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置编码：为输入序列添加位置信息
    输入：[batch_size, seq_len, d_model]
    输出：[batch_size, seq_len, d_model] (加上了位置编码)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)

        # 生成位置索引：[0, 1, 2, ..., max_len-1] -> 形状=[max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))

        # 偶数维度->sin, 奇数维度->cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # bath维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        # 只取前 seq_len 个位置编码
        return x + self.pe[:, :x.size(1), :]
    

class FeedForward(nn.Module):
    """前馈网络，用于编码器和解码器层中"""
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


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    输入 q, k, v: [batch_size, seq_len, d_model]
    输出: [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 定义线性变换层（可学习参数）
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len] 或 None（分类任务通常不需要）
        """
        batch_size = q.size(0)
        
        # 1. 线性变换
        q = self.q_linear(q)  # [B, L, D]
        k = self.k_linear(k)  # [B, L, D]
        v = self.v_linear(v)  # [B, L, D]
        
        # 2. 拆分成多头: [B, L, D] -> [B, L, H, d_k] -> [B, H, L, d_k]
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]
        
        # 4. 应用 mask（如果有的话）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Softmax 得到注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 6. 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 7. 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 8. 最后的线性变换
        output = self.out_linear(output)
        
        return output