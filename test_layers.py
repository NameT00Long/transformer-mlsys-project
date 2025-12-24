# test_layers.py
import torch
from model.layers import PositionalEncoding, MultiHeadAttention

def test_positional_encoding():
    print("🔍 测试 PositionalEncoding...")
    d_model = 256
    seq_len = 10
    batch_size = 2
    
    pe = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe(x)
    
    assert out.shape == x.shape, f"形状错误！期望 {x.shape}, 得到 {out.shape}"
    print("✅ PositionalEncoding 测试通过！\n")

def test_multi_head_attention():
    print("🔍 测试 MultiHeadAttention...")
    d_model = 256
    num_heads = 4
    seq_len = 10
    batch_size = 2
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 分类任务：q=k=v=x
    out = mha(x, x, x)
    
    assert out.shape == x.shape, f"形状错误！期望 {x.shape}, 得到 {out.shape}"
    print("✅ MultiHeadAttention 测试通过！\n")

if __name__ == "__main__":
    test_positional_encoding()
    test_multi_head_attention()
    print("🎉 所有测试通过！layers.py 已准备好！")