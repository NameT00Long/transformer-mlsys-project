"""
翻译推理模块 - 已修复
"""
import os
import sys
import pickle
import torch
import re  # 导入正则用于简单分词
from model.full_transformer import Transformer

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir

if project_root not in sys.path:
    sys.path.append(project_root)

# 模型参数
d_model = 256        # 根据报错修正：256
num_layers = 3       # 根据报错修正：3 (因为缺失了 layer 3,4,5)
num_heads = 8        # 保持默认，通常 d_model(256) 能被 8 整除 (256/8=32)，应该没问题
                     # 如果再次报错，尝试改为 4

d_ff = d_model * 4   # 这会自动变成 1024，与报错信息吻合
max_len = 50        # 这个通常不影响权重加载，保持默认即可
dropout = 0.1

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vocab(vocab_path):
    """加载词表文件，支持多种格式"""
    print(f"🔍 尝试加载词表文件: {vocab_path}")
    
    if not os.path.exists(vocab_path):
        print(f"❌ 错误: 词表文件不存在: {vocab_path}")
        return None, None
    
    try:
        # 为了让pickle能找到Vocab类，有时候需要把当前目录加到path（虽然上面已经加了）
        # 如果pickle报错 "Can't get attribute 'Vocab'..."，需要确保包含定义Vocab的文件在路径中
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        print(f"✅ 成功加载词表文件，类型: {type(vocab)}")
        
        # 1. 检查是否是标准格式 {'token2idx': ...}
        if isinstance(vocab, dict):
            if 'token2idx' in vocab and 'idx2token' in vocab:
                return vocab['token2idx'], vocab['idx2token']
            return vocab, {v: k for k, v in vocab.items()}
        
        # 2. 检查 token_to_idx (你的Vocab类使用的格式)
        elif hasattr(vocab, 'token_to_idx') and hasattr(vocab, 'idx_to_token'):
            print(f"✅ 识别到 token_to_idx 属性格式")
            return vocab.token_to_idx, vocab.idx_to_token
            
        # 3. 检查 token2idx (另一种常见格式)
        elif hasattr(vocab, 'token2idx') and hasattr(vocab, 'idx2token'):
            print(f"✅ 识别到 token2idx 属性格式")
            return vocab.token2idx, vocab.idx2token
            
        # 4. 列表格式
        elif isinstance(vocab, list):
            token2idx = {token: idx for idx, token in enumerate(vocab)}
            idx2token = {idx: token for idx, token in enumerate(vocab)}
            return token2idx, idx2token
            
        else:
            # 尝试强转字典
            try:
                token2idx = dict(vocab)
                idx2token = {v: k for k, v in token2idx.items()}
                return token2idx, idx2token
            except:
                print(f"❌ 错误: 无法识别词表对象内部结构: {dir(vocab)}")
                return None, None
                
    except Exception as e:
        print(f"❌ 加载词表时出错: {e}")
        return None, None

def load_model():
    """加载训练好的模型"""
    token2idx_src, idx2token_src = load_vocab(os.path.join(project_root, 'src_vocab.pkl'))
    token2idx_tgt, idx2token_tgt = load_vocab(os.path.join(project_root, 'tgt_vocab.pkl'))
    
    if token2idx_src is None or token2idx_tgt is None:
        return None, None, None, None
    
    print(f"📊 源语言词汇表大小: {len(token2idx_src)}")
    print(f"📊 目标语言词汇表大小: {len(token2idx_tgt)}")
    
    model = Transformer(
        src_vocab_size=len(token2idx_src),
        tgt_vocab_size=len(token2idx_tgt),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    
    model_path = os.path.join(project_root, 'translation_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ 模型文件未找到: {model_path}")
        return None, None, None, None
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 模型权重已加载")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None, None, None
    
    model.eval()
    return model, token2idx_src, token2idx_tgt, idx2token_tgt

def translate(text, model, token2idx_src, token2idx_tgt, idx2token_tgt):
    """
    修正版翻译函数：添加了缺失的 Source SOS 标记
    """
    # 1. 获取特殊token索引
    # 假设源语言和目标语言的特殊token字符串是一样的
    # 如果你的 src_vocab 里没有 <sos>，通常可以用 <unk> 代替，但根据你的代码来看是有的
    SRC_SOS_IDX = token2idx_src.get('<sos>', token2idx_src.get('<unk>', 1))
    SRC_UNK_IDX = token2idx_src.get('<unk>', 3)
    
    TGT_SOS_IDX = token2idx_tgt.get('<sos>', 1)
    TGT_EOS_IDX = token2idx_tgt.get('<eos>', 2)
    
    # print(f"\n[DEBUG] Src SOS={SRC_SOS_IDX}, Tgt SOS={TGT_SOS_IDX}, EOS={TGT_EOS_IDX}")
    
    # 2. 分词与映射
    import re
    # 简单分词：把标点隔开
    text = re.sub(r"([?.!,])", r" \1 ", text)
    tokens = text.lower().split()
    
    # 3. 构建源语言序列 [SOS, ... tokens ..., EOS]
    src_indices = [SRC_SOS_IDX]  # <--- 【关键修复】添加 SOS 到开头
    
    for token in tokens:
        idx = token2idx_src.get(token, SRC_UNK_IDX)
        src_indices.append(idx)
    
    # 添加结束标记 (根据你的Dataset代码，源语言末尾也有EOS)
    src_indices.append(token2idx_src.get('<eos>', 2)) 
    
    print(f"[DEBUG] 源语言Tensor索引 (修正后): {src_indices}")
    
    # 4. 准备张量
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # 目标序列初始化 [SOS]
    tgt_indices = [TGT_SOS_IDX]
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            
            # 前向传播
            output = model(src_tensor, tgt_tensor)
            
            # 获取最后一个时间步的预测
            last_token_logits = output[0, -1, :]
            
            # 贪婪搜索: 取概率最大的词
            next_token = last_token_logits.argmax().item()
            
            # --- 调试打印 (可选) ---
            # probs = torch.softmax(last_token_logits, dim=0)
            # top3_prob, top3_idx = torch.topk(probs, 3)
            # top3_words = [idx2token_tgt[idx.item()] if 0 <= idx.item() < len(idx2token_tgt) else 'ERR' for idx in top3_idx]
            # print(f"[DEBUG] Step {i}: 预测={top3_words[0]}")
            # ---------------------
            
            if next_token == TGT_EOS_IDX:
                break
            
            tgt_indices.append(next_token)
    
    # 5. 结果转换 (跳过开头的 SOS)
    translation_tokens = []
    for idx in tgt_indices[1:]:
        if 0 <= idx < len(idx2token_tgt):
            translation_tokens.append(idx2token_tgt[idx])
        else:
            translation_tokens.append('<unk>')
        
    return ' '.join(translation_tokens)    
    """
    已修复列表访问错误的翻译函数
    """
    # 1. 获取特殊token索引
    # 注意：token2idx 是字典，所以这里用 .get() 是正确的
    PAD_IDX = token2idx_src.get('<pad>', 0)
    SOS_IDX = token2idx_tgt.get('<sos>', 1)
    EOS_IDX = token2idx_tgt.get('<eos>', 2)
    UNK_IDX = token2idx_src.get('<unk>', 3)
    
    print(f"\n[DEBUG] 特殊Token索引: SOS={SOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")
    
    # 2. 分词与映射
    import re
    text = re.sub(r"([?.!,])", r" \1 ", text)
    tokens = text.lower().split()
    
    print(f"[DEBUG] 分词结果: {tokens}")
    
    src_indices = []
    for token in tokens:
        idx = token2idx_src.get(token, UNK_IDX)
        src_indices.append(idx)
        if idx == UNK_IDX:
            print(f"[DEBUG] ⚠️ 警告: 单词 '{token}' 未在词表中找到，转换为 UNK")
    
    src_indices.append(EOS_IDX)
    print(f"[DEBUG] 源语言Tensor索引: {src_indices}")
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    tgt_indices = [SOS_IDX]
    
    print(f"[DEBUG] 开始生成 (最大长度 {max_len})...")
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            
            # 前向传播
            output = model(src_tensor, tgt_tensor)
            
            # 获取预测结果
            last_token_logits = output[0, -1, :]
            
            # --- 修复 1：调试打印 ---
            probs = torch.softmax(last_token_logits, dim=0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            current_top1 = top3_idx[0].item()
            
            # 安全地从列表中获取单词用于显示
            top3_words = []
            for idx_tensor in top3_idx:
                idx = idx_tensor.item()
                if 0 <= idx < len(idx2token_tgt):
                    top3_words.append(idx2token_tgt[idx])  # 使用列表索引访问
                else:
                    top3_words.append('<out_of_bounds>')

            print(f"[DEBUG] Step {i}: 当前序列={tgt_indices}, 预测Top3={top3_words} (IDs: {top3_idx.tolist()})")
            
            next_token = current_top1
            
            if next_token == EOS_IDX:
                print(f"[DEBUG] 🛑 模型生成了 EOS，停止生成。")
                break
            
            tgt_indices.append(next_token)
    
    # --- 修复 2：结果转换 ---
    translation_tokens = []
    for idx in tgt_indices[1:]: # 跳过开头的SOS
        # 使用列表索引访问
        if 0 <= idx < len(idx2token_tgt):
            token = idx2token_tgt[idx]
        else:
            token = '<unk>'
        translation_tokens.append(token)
        
    translation = ' '.join(translation_tokens)
    return translation    
    """
    带调试信息的翻译函数
    """
    # 1. 获取特殊token索引，并打印出来核对
    PAD_IDX = token2idx_src.get('<pad>', 0)
    SOS_IDX = token2idx_tgt.get('<sos>', 1)
    EOS_IDX = token2idx_tgt.get('<eos>', 2)
    UNK_IDX = token2idx_src.get('<unk>', 3)
    
    print(f"\n[DEBUG] 特殊Token索引: SOS={SOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")
    
    # 2. 分词与映射
    import re
    text = re.sub(r"([?.!,])", r" \1 ", text)
    tokens = text.lower().split()
    
    # 打印原始分词
    print(f"[DEBUG] 分词结果: {tokens}")
    
    src_indices = []
    for token in tokens:
        idx = token2idx_src.get(token, UNK_IDX)
        src_indices.append(idx)
        # 如果是UNK，打印警告
        if idx == UNK_IDX:
            print(f"[DEBUG] ⚠️ 警告: 单词 '{token}' 未在词表中找到，转换为 UNK (id={UNK_IDX})")
    
    src_indices.append(EOS_IDX)
    print(f"[DEBUG] 源语言Tensor索引: {src_indices}")
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    tgt_indices = [SOS_IDX]
    
    print(f"[DEBUG] 开始生成 (最大长度 {max_len})...")
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            
            # 前向传播
            output = model(src_tensor, tgt_tensor)
            
            # 获取预测结果
            # output: [1, seq_len, vocab_size]
            last_token_logits = output[0, -1, :]
            
            # 打印Top 3预测（看看模型最想输出什么）
            probs = torch.softmax(last_token_logits, dim=0)
            top3_prob, top3_idx = torch.topk(probs, 3)
            
            current_top1 = top3_idx[0].item()
            
            print(f"[DEBUG] Step {i}: 输入={tgt_indices}, 预测Top3={[idx2token_tgt.get(x.item(), str(x.item())) for x in top3_idx]} (IDs: {top3_idx.tolist()})")
            
            next_token = current_top1
            
            # 这里的判断非常关键
            if next_token == EOS_IDX:
                print(f"[DEBUG] 🛑 模型生成了 EOS (id={EOS_IDX})，停止生成。")
                break
            
            tgt_indices.append(next_token)
    
    # 转换结果
    translation_tokens = []
    for idx in tgt_indices[1:]: # 跳过开头的SOS
        token = idx2token_tgt.get(idx, '<unk>') # 安全获取
        translation_tokens.append(token)
        
    translation = ' '.join(translation_tokens)
    return translation    # 特殊token处理
    # 注意：这里使用了 .get()，如果词表中key不同，需要调整，但通常 key 都是字符串
    PAD_IDX = token2idx_src.get('<pad>', 0)
    SOS_IDX = token2idx_tgt.get('<sos>', 1)
    EOS_IDX = token2idx_tgt.get('<eos>', 2)
    UNK_IDX = token2idx_src.get('<unk>', 3)
    
    # 简单分词优化
    text = re.sub(r"([?.!,])", r" \1 ", text)
    tokens = text.lower().split()
    
    src_indices = [token2idx_src.get(token, UNK_IDX) for token in tokens]
    src_indices.append(EOS_IDX)
    
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    tgt_indices = [SOS_IDX]
    
    with torch.no_grad():
        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
            output = model(src_tensor, tgt_tensor)
            
            # --- 修复后的预测逻辑 ---
            # 取 batch=0, 最后一个时间步, argmax
            next_token = output[0, -1, :].argmax().item()
            
            tgt_indices.append(next_token)
            if next_token == EOS_IDX:
                break
    
    translation_tokens = [idx2token_tgt[idx] for idx in tgt_indices[1:]]
    # 去掉EOS如果存在
    if translation_tokens and translation_tokens[-1] == '<eos>':
        translation_tokens = translation_tokens[:-1]
        
    return ' '.join(translation_tokens)

if __name__ == "__main__":
    print("🚀 翻译模型推理开始...")
    model, token2idx_src, token2idx_tgt, idx2token_tgt = load_model()
    
    if model:
        print("\n✅ 系统就绪. 请输入英文句子 (输入 'quit' 退出):")
        while True:
            text = input("\n输入: ")
            if text.lower() == 'quit': break
            if text.strip():
                try:
                    res = translate(text, model, token2idx_src, token2idx_tgt, idx2token_tgt)
                    print(f"输出: {res}")
                except Exception as e:
                    print(f"❌ 推理出错: {e}")
                    import traceback
                    traceback.print_exc()