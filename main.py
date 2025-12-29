import argparse
import os
import sys
from scripts.train import train_model


def get_default_params():
    """返回默认训练参数"""
    return {
        'batch_size': 32,
        'max_seq_len': 256,
        'max_tokens': 10000,
        'epochs': 5,
        'lr': 1e-4,
        'd_model': 256,
        'num_heads': 4,
        'num_layers': 4,
        'num_classes': 2,
        'dropout': 0.1,
        'data_dir': 'aclImdb',
        'model_save_path': 'best_model.pth'
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Transformer模型训练')
    
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_seq_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--max_tokens', type=int, default=10000, help='最大词汇数量')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4, help='层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--data_dir', type=str, default='aclImdb', help='数据目录')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help='模型保存路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取默认参数
    params = get_default_params()
    
    # 使用命令行参数覆盖默认参数
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value
    
    print("🔧 训练参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # 检查数据目录是否存在
    if not os.path.exists(params['data_dir']):
        print(f"❌ 错误: 未找到数据目录 {params['data_dir']}")
        print("请确保IMDb数据集已下载并放置在正确目录中")
        return
    
    # 开始训练
    print("\n🚀 开始训练...")
    best_accuracy = train_model(params)
    
    print(f"\n🎉 训练完成！最佳验证准确率: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()