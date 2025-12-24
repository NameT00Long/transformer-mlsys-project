try:
    import torch
    print(f"PyTorch 已安装")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 获取更多详细信息
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    
except ImportError:
    print("PyTorch 未安装")
    print("请运行以下命令安装:")
    print("pip install torch torchvision torchaudio")