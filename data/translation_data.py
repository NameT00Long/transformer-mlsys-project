"""
翻译数据处理模块
"""
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import torch.nn.functional as F

def simple_tokenizer(text):
    """
    统一的简单的分词器：
    1. 转小写
    2. 将标点符号与单词分开
    3. 按空格切分
    """
    if not isinstance(text, str):
        return []
    
    text = text.lower().strip()
    
    # 在标点符号前后加空格 (包括 ?.!,)
    # 这样 "hello." 就会变成 "hello ."
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    
    # 将多余的空格合并为一个
    text = re.sub(r'[" "]+', " ", text)
    
    return text.split()

class TranslationDataset(Dataset):
    def __init__(self, dataset, src_lang, tgt_lang, src_vocab, tgt_vocab, max_len=100):
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 提取文本
        src_text = self._extract_text(item, self.src_lang)
        tgt_text = self._extract_text(item, self.tgt_lang)

        # --- 修改处：使用统一的分词器 ---
        src_word_list = simple_tokenizer(src_text)
        tgt_word_list = simple_tokenizer(tgt_text)

        # 编码 (截断长度，留出SOS和EOS的位置)
        src_tokens = [self.src_vocab.get('<sos>', 1)] + \
                     [self.src_vocab.get(token, 3) for token in src_word_list[:self.max_len-2]] + \
                     [self.src_vocab.get('<eos>', 2)]
        
        tgt_tokens = [self.tgt_vocab.get('<sos>', 1)] + \
                     [self.tgt_vocab.get(token, 3) for token in tgt_word_list[:self.max_len-2]] + \
                     [self.tgt_vocab.get('<eos>', 2)]

        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)
    
    def _extract_text(self, item, lang):
        # ... (保持原来的提取逻辑不变) ...
        if isinstance(item, dict):
            if lang in item: return item[lang]
            elif 'translation' in item and lang in item['translation']: return item['translation'][lang]
        return ""



class Vocab:
    """词汇表类"""
    def __init__(self, tokens=None, max_tokens=None):
        self.token_to_idx = {}
        self.idx_to_token = []
        # 添加特殊标记
        self.pad_token = '<pad>'  # 索引0
        self.sos_token = '<sos>'  # 索引1
        self.eos_token = '<eos>'  # 索引2
        self.unk_token = '<unk>'  # 索引3

        # 添加特殊标记
        special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        for token in special_tokens:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token.append(token)

        if tokens:
            self.build_vocab(tokens, max_tokens)

    def build_vocab(self, tokens, max_tokens=None):
        """构建词汇表"""
        counter = Counter(tokens)
        sorted_tokens = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        for token, _ in sorted_tokens:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
                if max_tokens and len(self.idx_to_token) >= max_tokens:
                    break

    def __len__(self):
        return len(self.idx_to_token)

    def get(self, token, default=None):
        """获取token的索引"""
        if default is None:
            return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])
        else:
            return self.token_to_idx.get(token, default)

    def to_tokens(self, indices):
        """将索引转换为token"""
        return [self.idx_to_token[idx] for idx in indices]


def build_vocab_from_dataset(dataset, src_lang, tgt_lang, max_tokens=30000):
    """从数据集构建词汇表"""
    print("🔄 正在构建词汇表 (使用 simple_tokenizer)...")
    src_counter = Counter()
    tgt_counter = Counter()
    
    # 遍历数据集 (IWSLT不算太大，建议遍历全部或至少5万条)
    # 如果为了速度，可以只遍历前 30000 条
    limit = min(30000, len(dataset)) 
    
    for i in range(limit):
        item = dataset[i]
        src_text = extract_text_from_item(item, src_lang)
        tgt_text = extract_text_from_item(item, tgt_lang)
        
        # --- 修改处：使用统一的分词器进行统计 ---
        src_counter.update(simple_tokenizer(src_text))
        tgt_counter.update(simple_tokenizer(tgt_text))
    
    # 提取最常见的单词列表
    src_tokens_list = [token for token, count in src_counter.most_common(max_tokens)]
    tgt_tokens_list = [token for token, count in tgt_counter.most_common(max_tokens)]

    # 构建
    src_vocab = Vocab(src_tokens_list, max_tokens=max_tokens)
    tgt_vocab = Vocab(tgt_tokens_list, max_tokens=max_tokens)
    
    return src_vocab, tgt_vocab


def extract_text_from_item(item, lang):
    """从数据项中提取文本，处理不同的数据格式"""
    # 如果是简单的键值对格式
    if isinstance(item, dict):
        if lang in item:
            return item[lang]
        # 如果是translation格式
        elif 'translation' in item and lang in item['translation']:
            return item['translation'][lang]
        # 如果是sentence格式
        elif 'sentence' in item and lang == 'sentence':
            return item['sentence']
    
    # 如果没有找到对应语言的文本，返回空字符串
    return ""


def collate_fn(batch):
    """批处理函数，用于处理不同长度的序列"""
    src_batch, tgt_batch = zip(*batch)
    
    # 找到批次中最长的序列长度
    src_max_len = max([len(seq) for seq in src_batch])
    tgt_max_len = max([len(seq) for seq in tgt_batch])
    
    # 填充到相同长度
    src_padded = [F.pad(seq, (0, src_max_len - len(seq)), value=0) for seq in src_batch]
    tgt_padded = [F.pad(seq, (0, tgt_max_len - len(seq)), value=0) for seq in tgt_batch]
    
    return torch.stack(src_padded), torch.stack(tgt_padded)


def get_dataloaders(data_dir, batch_size=64, max_len=100, src_lang='en', tgt_lang='de', max_tokens=30000):
    """
    使用 Opus Books 数据集 (小说/短句)
    优点：下载极其稳定，无需脚本，适合课程作业
    """
    print(f"📥 正在加载 Opus Books (en-de) 数据集...")
    from datasets import load_dataset
    
    try:
        # 加载 Opus Books
        # 这是一个纯数据文件，镜像站下载非常快，且不需要 trust_remote_code
        dataset = load_dataset("opus_books", "de-en")
        
        print(f"💡 数据集加载成功 (大小: {len(dataset['train'])})，正在构建词表...")
        
        # Opus Books 只有 'train' 分割
        # 我们需要手动切分出 验证集 和 测试集
        # 90% 训练, 5% 验证, 5% 测试
        full_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_set = full_dataset['train']
        
        test_val_split = full_dataset['test'].train_test_split(test_size=0.5, seed=42)
        val_set = test_val_split['train']
        test_set = test_val_split['test']
        
        # 构建词表 (使用之前定义的 build_vocab_from_dataset)
        # 注意：Opus 的数据结构是 item['translation']['en']
        src_vocab, tgt_vocab = build_vocab_from_dataset(
            train_set, 'en', 'de', max_tokens=max_tokens
        )
        
        # 创建 Dataset 对象
        train_dataset = TranslationDataset(train_set, 'en', 'de', src_vocab, tgt_vocab, max_len)
        val_dataset = TranslationDataset(val_set, 'en', 'de', src_vocab, tgt_vocab, max_len)
        test_dataset = TranslationDataset(test_set, 'en', 'de', src_vocab, tgt_vocab, max_len)
        
        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        return train_loader, val_loader, test_loader, src_vocab, tgt_vocab

    except Exception as e:
        print(f"❌ 加载数据集出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None