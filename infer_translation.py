import torch
import pickle
import sys
import os

# --- 新增/修改的部分：自动获取项目根目录并添加到搜索路径 ---
# 获取当前文件（比如 inference.py）的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 model 文件夹和当前脚本在同一个目录下，那么项目根目录就是 current_dir
project_root = current_dir 

# 将 project_root 加入系统路径，解决 ModuleNotFoundError
if project_root not in sys.path:
    sys.path.append(project_root)
# -------------------------------------------------------

# 现在这行就不会报错了
from model.full_transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载词表（假设是 dict: token -> idx，并保存了 idx->token 反向表或可反向）
try:
    with open(os.path.join(project_root, 'src_vocab.pkl'), 'rb') as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(project_root, 'tgt_vocab.pkl'), 'rb') as f:
        tgt_vocab = pickle.load(f)
except FileNotFoundError:
    print("❌ 词汇表文件未找到。请先运行训练脚本生成词汇表。")
    exit(1)

# 确保词汇表是正确的格式
if hasattr(src_vocab, 'token_to_idx'):  # 如果是自定义Vocab类
    token2idx_src = src_vocab.token_to_idx
    idx2token_src = src_vocab.idx_to_token
else:  # 如果是字典
    token2idx_src = src_vocab
    idx2token_src = {i:t for t,i in token2idx_src.items()}

if hasattr(tgt_vocab, 'token_to_idx'):  # 如果是自定义Vocab类
    token2idx_tgt = tgt_vocab.token_to_idx
    idx2token_tgt = tgt_vocab.idx_to_token
else:  # 如果是字典
    token2idx_tgt = tgt_vocab
    idx2token_tgt = {i:t for t,i in token2idx_tgt.items()}

# 调整以下特殊 token id 为你词表中的真实 id
PAD_IDX = 0
SOS_IDX = token2idx_tgt.get('<sos>', 1)
EOS_IDX = token2idx_tgt.get('<eos>', 2)
UNK_IDX = token2idx_src.get('<unk>', 3)

# 恢复模型结构并加载权重
# 使用与训练时相同的参数
d_model = 512  # 与训练时保持一致
num_heads = 8
num_layers = 3
d_ff = d_model * 4  # 通常为 d_model 的 4 倍
max_len = 512
dropout = 0.1

model = Transformer(
    src_vocab_size=len(idx2token_src),
    tgt_vocab_size=len(idx2token_tgt),
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len,
    dropout=dropout
).to(device)

try:
    model_path = os.path.join(project_root, 'translation_model.pth')
    ckpt = torch.load(model_path, map_location=device)
    # 如果 ckpt 是 state_dict 或 包含 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict) and any(k.startswith('encoder') or k.startswith('decoder') for k in ckpt.keys()):
        model.load_state_dict(ckpt)
    else:
        # 兼容直接保存整个模型的情况
        model.load_state_dict(ckpt)
except FileNotFoundError:
    print("❌ 模型文件未找到。请先运行训练脚本生成模型。")
    exit(1)

model.eval()

# 简单 tokenizer: 空格分词（请替换为 data/translation_data.py 中一致的 tokenizer）
def encode_src(text, max_len=50):
    tokens = text.strip().split()
    ids = [token2idx_src.get(t, UNK_IDX) for t in tokens][:max_len]
    return torch.tensor([ids], dtype=torch.long, device=device)  # shape (1, L)

# 贪心解码（逐步生成）
@torch.no_grad()
def translate(text, max_len=50):
    src_ids = encode_src(text, max_len=max_len)  # (1, L)
    # 初始 target 输入为 SOS
    tgt_ids = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
    for _ in range(max_len):
        out = model(src_ids, tgt_ids)  # 期望模型返回 logits (B, T, V)
        next_logits = out[:, -1, :]    # (B, V)
        next_tok = next_logits.argmax(-1).unsqueeze(1)  # (B,1)
        tgt_ids = torch.cat([tgt_ids, next_tok], dim=1)
        if next_tok.item() == EOS_IDX:
            break
    # 转回文本（不包含 SOS）
    ids = tgt_ids[0].cpu().tolist()[1:]
    tokens = []
    for i in ids:
        if i == EOS_IDX: break
        tokens.append(idx2token_tgt[i] if i < len(idx2token_tgt) else '<unk>')
    return ' '.join(tokens)

# 示例
if __name__ == "__main__":
    print("🚀 翻译模型推理开始...")
    print("请输入要翻译的英文句子（输入'quit'退出）:")
    
    while True:
        text = input("\n输入: ")
        if text.lower() == 'quit':
            break
        if text.strip():
            translation = translate(text)
            print(f"输出: {translation}")