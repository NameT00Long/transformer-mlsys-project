import argparse
import os
import sys
from train_translation import train_translation_model


def get_translation_params():
    """返回翻译任务的默认参数"""
    return {
        'data_dir': 'data/opus_books', # 数据集缓存路径
        'src_lang': 'en',
        'tgt_lang': 'de',
        
        # --- 核心模型参数 (标准 Transformer) ---
        'd_model': 512,        # 从 256 改回 512，4060 毫无压力
        'num_heads': 8,        # 512 / 8 = 64 (标准头大小)
        'num_layers': 6,       # 深度增加到 6 层，提升翻译质量
        'dropout': 0.1,        # 防止过拟合
        
        # --- 训练参数 ---
        'batch_size': 64,      # 8GB 显存对于 max_len=100 可以轻松跑 64 甚至 128
                               # 如果显存不够，改回 32
        'max_seq_len': 100,    # 短句翻译不需要 512，设为 100 节省大量显存
        'lr': 0.0001,          # 稍微调低一点学习率，训练更稳定
        'epochs': 20,          # IWSLT 数据多了，建议跑 20 轮 (约1-2小时)
        
        # --- 路径 ---
        'model_save_path': 'translation_model.pth'
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Transformer翻译模型训练')
    
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
    parser.add_argument('--max_tokens', type=int, default=8000, help='最大词汇数量')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=3, help='层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--data_dir', type=str, default='data/multi30k', help='数据目录')
    parser.add_argument('--model_save_path', type=str, default='translation_model.pth', help='模型保存路径')
    parser.add_argument('--src_lang', type=str, default='en', help='源语言')
    parser.add_argument('--tgt_lang', type=str, default='de', help='目标语言')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取默认参数
    params = get_translation_params()
    
    # 使用命令行参数覆盖默认参数
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value
    
    print("🔧 训练参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n🚀 开始翻译训练...")
    best_loss = train_translation_model(params)
    print(f"\n🎉 翻译训练完成！最佳验证损失: {best_loss:.4f}")


if __name__ == "__main__":
    main()