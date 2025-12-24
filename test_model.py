# test_model.py
import torch
from model.transformer import TransformerClassifier

def test_transformer_classifier():
    print("🔍 测试 TransformerClassifier...")
    
    # 超参数
    vocab_size = 10000   # IMDb 通常用 10k 词表
    d_model = 256
    batch_size = 4
    seq_len = 256
    
    # 创建模型
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=4,
        num_classes=2
    )
    
    # 模拟输入：token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # [4, 256]
    
    # 前向传播
    with torch.no_grad():  # 测试时不计算梯度
        logits = model(input_ids)
    
    # 检查输出形状
    assert logits.shape == (batch_size, 2), f"期望 (4, 2)，得到 {logits.shape}"
    print("✅ 模型前向传播成功！")
    print(f"  输入形状: {input_ids.shape}")
    print(f"  输出形状: {logits.shape}")
    print(f"  输出示例: {logits[0].tolist()}")

if __name__ == "__main__":
    test_transformer_classifier()