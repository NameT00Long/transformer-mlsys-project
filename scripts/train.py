import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import re
import random
from model.transformer_classifier import TransformerClassifier


def preprocess_text(text):
    """文本预处理函数"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.strip()


def basic_tokenize(text):
    """基础分词函数"""
    return text.split()


def load_imdb_from_local(data_dir="aclImdb"):
    """从本地目录加载IMDb数据"""
    def read_split(split):
        data = []
        for label_name, label in [("pos", 1), ("neg", 0)]:
            path = os.path.join(data_dir, split, label_name)
            for fname in os.listdir(path):
                if fname.endswith(".txt"):
                    full_path = os.path.join(path, fname)
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    data.append((text, label))
        return data

    print(f"📂 从 {data_dir} 加载 IMDb...")
    train_data = read_split("train")
    test_data = read_split("test")
    random.shuffle(train_data)
    print(f"✅ 训练样本数: {len(train_data)}, 测试样本数: {len(test_data)}")
    return train_data, test_data


def build_vocab(train_data, min_freq=5, max_tokens=10000):
    """构建词汇表"""
    counter = Counter()
    for text, _ in train_data:
        tokens = basic_tokenize(preprocess_text(text))
        counter.update(tokens)
    
    most_common = counter.most_common(max_tokens)
    vocab = {"<unk>": 0, "<pad>": 1}
    for word, freq in most_common:
        if freq >= min_freq and len(vocab) < max_tokens + 2:
            vocab[word] = len(vocab)
    return vocab


def collate_batch(batch, vocab, max_seq_len=256):
    """批处理函数"""
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    
    sequences = []
    for text in texts:
        tokens = basic_tokenize(preprocess_text(text))
        ids = [vocab.get(t, 0) for t in tokens]
        if len(ids) > max_seq_len:
            ids = ids[:max_seq_len]
        else:
            ids += [1] * (max_seq_len - len(ids))
        sequences.append(ids)
    
    input_ids = torch.tensor(sequences, dtype=torch.long)
    return input_ids, labels


def load_imdb_data(batch_size=32, max_seq_len=256, max_tokens=10000, data_dir="aclImdb"):
    """主数据加载函数"""
    train_data, test_data = load_imdb_from_local(data_dir)
    vocab = build_vocab(train_data, min_freq=5, max_tokens=max_tokens)
    
    def collate_fn(batch):
        return collate_batch(batch, vocab, max_seq_len)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, test_loader, vocab


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def train_model(params):
    """训练模型主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    if not os.path.exists(params['data_dir']):
        print(f"❌ 错误: 未找到 {params['data_dir']} 目录！")
        exit(1)

    # 加载数据
    print("📥 加载数据...")
    train_loader, test_loader, vocab = load_imdb_data(
        batch_size=params['batch_size'],
        max_seq_len=params['max_seq_len'],
        max_tokens=params['max_tokens'],
        data_dir=params['data_dir']
    )
    vocab_size = len(vocab)
    print(f"🔤 词汇表大小: {vocab_size}")

    # 初始化模型
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=params['d_model'],
        num_heads=params['num_heads'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        dropout=params['dropout']
    ).to(device)

    print(f"📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器 & 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    best_acc = 0.0
    for epoch in range(1, params['epochs'] + 1):
        print(f"\n🚀 Epoch {epoch}/{params['epochs']}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"📈 Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

        # 验证
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"📉 Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), params['model_save_path'])
            print(f"✨ 模型已保存 (Val Acc: {val_acc:.2f}%)")

    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.2f}%")
    return best_acc