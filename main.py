import argparse
import os
import sys
from train_translation import train_translation_model


def get_translation_params():
    """返回翻译任务的默认参数"""
    return {
        'batch_size': 64,  # 减小批次大小以适应更多数据
        'max_seq_len': 50,  # 适应Multi30k数据的平均长度
        'max_tokens': 8000,  # 增加词汇表大小
        'epochs': 10,
        'lr': 5e-4,  # 稍微增加学习率
        'd_model': 256,  # 减小模型大小以便快速训练测试
        'num_heads': 8,
        'num_layers': 3,  # 减少层数以便快速训练测试
        'dropout': 0.1,
        'data_dir': 'data/multi30k',
        'model_save_path': 'translation_model.pth',
        'src_lang': 'en',
        'tgt_lang': 'de'
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