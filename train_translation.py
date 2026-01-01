"""
翻译任务训练模块
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.full_transformer import Transformer
from data.translation_data import get_dataloaders
import pickle
import matplotlib.pyplot as plt


def calculate_accuracy(outputs, targets):
    """计算预测准确率"""
    with torch.no_grad():
        # 获取预测结果
        predicted = outputs.argmax(dim=-1)
        # 创建掩码，忽略填充值（假设填充值为0）
        mask = targets != 0
        # 计算准确率
        correct = (predicted == targets) & mask
        accuracy = correct.sum().item() / mask.sum().item()
        return accuracy


def train_translation_epoch(model, dataloader, optimizer, criterion, device):
    """训练翻译模型一个epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        
        # 在训练时，目标序列需要移位作为输入，预测下一个词
        tgt_input = tgt[:, :-1]  # 除最后一个词
        tgt_output = tgt[:, 1:]  # 从第二个词开始
        
        outputs = model(src, tgt_input)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.contiguous().view(-1))
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # 计算准确率
        acc = calculate_accuracy(outputs, tgt_output)
        total_acc += acc
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc


def evaluate_translation(model, dataloader, criterion, device):
    """评估翻译模型"""
    model.eval()
    total_loss = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            outputs = model(src, tgt_input)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_output.contiguous().view(-1))

            total_loss += loss.item()
            # 计算准确率
            acc = calculate_accuracy(outputs, tgt_output)
            total_acc += acc
            num_batches += 1

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc / num_batches if num_batches > 0 else 0
    return avg_loss, avg_acc


def train_translation_model(params):
    """训练翻译模型主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

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
    print(f"📊 训练集大小: {len(train_loader.dataset) if hasattr(train_loader, 'dataset') else len(train_loader) * params['batch_size']}")
    print(f"📊 验证集大小: {len(val_loader.dataset) if hasattr(val_loader, 'dataset') else len(val_loader) * params['batch_size']}")
    print(f"📊 测试集大小: {len(test_loader.dataset) if hasattr(test_loader, 'dataset') else len(test_loader) * params['batch_size']}")

    # 初始化完整Transformer模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        d_ff=params['d_model']*4,  # 前馈网络维度通常是d_model的4倍
        num_layers=params['num_layers'],
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
            # 同时保存词汇表
            with open('src_vocab.pkl', 'wb') as f:
                pickle.dump(src_vocab, f)
            with open('tgt_vocab.pkl', 'wb') as f:
                pickle.dump(tgt_vocab, f)
            print(f"✨ 翻译模型和词汇表已保存 (Val Loss: {val_loss:.4f})")

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
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')

    print(f"\n✅ 翻译训练完成！最佳验证损失: {best_loss:.4f}")
    return best_loss