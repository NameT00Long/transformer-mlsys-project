"""
翻译任务训练模块
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.full_transformer import Transformer
from data.translation_data import get_dataloaders
import matplotlib.pyplot as plt

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if project_root not in sys.path:
    sys.path.append(project_root)

def prepare_vocab_for_saving(vocab, vocab_name):
    """
    准备词表以供保存，确保词表格式正确
    (已修复：支持 token_to_idx 属性名)
    """
    # 检查词表是否为空
    if vocab is None:
        raise ValueError(f"{vocab_name}为空")
    
    token2idx = None
    idx2token = None

    # 1. 检查字典格式
    if isinstance(vocab, dict):
        token2idx = vocab
        idx2token = {v: k for k, v in vocab.items()}
    
    # 2. 检查常见对象格式 (token2idx)
    elif hasattr(vocab, 'token2idx') and hasattr(vocab, 'idx2token'):
        token2idx = vocab.token2idx
        idx2token = vocab.idx2token
        
    # 3. [新增] 检查你的 Vocab 类格式 (token_to_idx)
    elif hasattr(vocab, 'token_to_idx') and hasattr(vocab, 'idx_to_token'):
        print(f"✅ 识别到 {vocab_name} 为 token_to_idx 格式")
        token2idx = vocab.token_to_idx
        # 注意：你的 idx_to_token 是列表，为了统一格式，我们在保存时把它转为字典
        if isinstance(vocab.idx_to_token, list):
            idx2token = {i: t for i, t in enumerate(vocab.idx_to_token)}
        else:
            idx2token = vocab.idx_to_token

    # 4. 检查 torchtext 旧版本格式
    elif hasattr(vocab, 'get_stoi') and hasattr(vocab, 'get_itos'):
        token2idx = vocab.get_stoi()
        idx2token = {i: s for s, i in token2idx.items()}
        
    # 5. 列表格式
    elif isinstance(vocab, list):
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
    
    # 6. 兜底尝试
    else:
        try:
            token2idx = dict(vocab)
            idx2token = {v: k for k, v in token2idx.items()}
        except (TypeError, ValueError):
            print(f"词表类型: {type(vocab)}")
            raise ValueError(f"{vocab_name}格式不支持: {type(vocab)}")
    
    # 验证提取结果
    if not token2idx or not idx2token:
        raise ValueError(f"{vocab_name}内容提取失败或为空")
    
    # 检查特殊token
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    missing_tokens = [token for token in special_tokens if token not in token2idx]
    if missing_tokens:
        print(f"⚠️ 警告: {vocab_name}缺少特殊token: {missing_tokens}")
    
    # 返回标准格式
    return {'token2idx': token2idx, 'idx2token': idx2token}

def calculate_accuracy(outputs, targets):
    """计算预测准确率"""
    _, predicted = torch.max(outputs, dim=2)
    correct = (predicted == targets).float()
    mask = (targets != 0).float()  # 忽略padding
    return (correct * mask).sum() / mask.sum()

def train_translation_epoch(model, dataloader, optimizer, criterion, device):
    """训练翻译模型一个epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # 目标输入和目标输出
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # 计算损失
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        accuracy = calculate_accuracy(
            output.reshape(src.size(0), tgt_input.size(1), -1),
            tgt_output.reshape(src.size(0), tgt_input.size(1))
        )
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def evaluate_translation(model, dataloader, criterion, device):
    """评估翻译模型"""
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # 目标输入和目标输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 前向传播
            output = model(src, tgt_input)
            
            # 计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
            
            # 计算准确率
            accuracy = calculate_accuracy(
                output.reshape(src.size(0), tgt_input.size(1), -1),
                tgt_output.reshape(src.size(0), tgt_input.size(1))
            )
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
    
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

def train_translation_model(params):
    """训练翻译模型主函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 检查数据目录是否存在
    if not os.path.exists(params['data_dir']):
        print(f"❌ 错误: 未找到数据目录 {params['data_dir']}")
        print("请确保翻译数据集已下载并放置在正确目录中")
        # 创建示例数据
        from data.download_sample_data import prepare_data_for_training
        prepare_data_for_training()
        params['data_dir'] = 'data/multi30k'  # 使用新的数据目录

    # 加载翻译数据
    print("📥 加载翻译数据...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        params['data_dir'],
        batch_size=params['batch_size'],
        max_len=params['max_seq_len'],
        src_lang=params['src_lang'],
        tgt_lang=params['tgt_lang']
    )
    
    if train_loader is None:
        print("❌ 无法加载数据")
        return float('inf')
    
    print(f"🔤 源语言词汇表大小: {len(src_vocab)}")
    print(f"🔤 目标语言词汇表大小: {len(tgt_vocab)}")
    print(f"📊 训练集大小: {len(train_loader.dataset)}")
    print(f"📊 验证集大小: {len(val_loader.dataset)}")
    print(f"📊 测试集大小: {len(test_loader.dataset)}")

    # 初始化完整Transformer模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        d_ff=params['d_model']*4,  # 前馈网络维度通常是d_model的4倍
        num_layers=params['num_layers'],
        max_len=params['max_seq_len'],
        dropout=params['dropout']
    ).to(device)

    print(f"📊 翻译模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器 & 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充值

    # 记录训练过程中的指标
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 训练循环
    best_loss = float('inf')
    for epoch in range(1, params['epochs'] + 1):
        print(f"\n🚀 Epoch {epoch}/{params['epochs']}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = train_translation_epoch(model, train_loader, optimizer, criterion, device)
        print(f"📈 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证
        val_loss, val_acc = evaluate_translation(model, val_loader, criterion, device)
        print(f"📉 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 保存最佳模型和词汇表
        if val_loss < best_loss:
            best_loss = val_loss
            # 保存模型
            torch.save(model.state_dict(), params['model_save_path'])
            
            # 准备词表以供保存
        try:
            src_vocab_dict = prepare_vocab_for_saving(src_vocab, "源语言词表")
            tgt_vocab_dict = prepare_vocab_for_saving(tgt_vocab, "目标语言词表")
            
            # 保存词表到项目根目录
            src_vocab_path = os.path.join(project_root, 'src_vocab.pkl')
            tgt_vocab_path = os.path.join(project_root, 'tgt_vocab.pkl')
            
            with open(src_vocab_path, 'wb') as f:
                pickle.dump(src_vocab_dict, f)
            with open(tgt_vocab_path, 'wb') as f:
                pickle.dump(tgt_vocab_dict, f)
            
            print(f"✨ 翻译模型和词汇表已保存 (Val Loss: {val_loss:.4f})")
            print(f"📁 模型路径: {params['model_save_path']}")
            print(f"📁 源语言词表路径: {src_vocab_path}")
            print(f"📁 目标语言词表路径: {tgt_vocab_path}")
        except Exception as e:
            print(f"❌ 保存词表时出错: {e}")
            # 尝试保存原始词表作为备份
            with open('src_vocab_backup.pkl', 'wb') as f:
                pickle.dump(src_vocab, f)
            with open('tgt_vocab_backup.pkl', 'wb') as f:
                pickle.dump(tgt_vocab, f)
            print("⚠️ 已保存原始词表备份")


    # 绘制训练过程中的准确率变化曲线
    epochs_range = range(1, len(train_accuracies) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'training_curves.png'), dpi=300, bbox_inches='tight')

    print(f"\n✅ 翻译训练完成！最佳验证损失: {best_loss:.4f}")
    return best_loss
