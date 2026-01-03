"""
下载和准备训练数据的模块
"""
import os
from datasets import load_dataset, load_from_disk


def prepare_data_for_training():
    """下载并准备Multi30k数据用于训练"""
    print("📥 开始下载Multi30k数据集...")
    
    try:
        # 使用Hugging Face datasets加载Multi30k数据集
        # Multi30k数据集包含英语到德语、英语到法语等翻译任务
        dataset = load_dataset("bentrevett/multi30k")
        
        print("✅ Multi30k数据集下载完成!")
        print(f"📊 数据集信息:")
        print(f"   训练集大小: {len(dataset['train'])}")
        print(f"   验证集大小: {len(dataset['validation'])}")
        print(f"   测试集大小: {len(dataset['test'])}")
        
        # 检查数据集结构并处理不同的列名格式
        train_sample = dataset['train'][0]
        print(f"   示例数据: {train_sample}")
        
        # 根据Multi30k数据集的实际结构，确定源语言和目标语言列名
        src_lang, tgt_lang = get_language_columns(dataset)
        print(f"   检测到源语言列: {src_lang}")
        print(f"   检测到目标语言列: {tgt_lang}")
        
        # 创建数据目录（如果不存在）
        os.makedirs('data/multi30k', exist_ok=True)
        
        # 保存数据集到磁盘
        dataset.save_to_disk('data/multi30k/dataset')
        print("💾 数据集已保存到 data/multi30k/dataset")
        
        # 保存语言配置
        with open('data/multi30k/lang_config.txt', 'w') as f:
            f.write(f"{src_lang},{tgt_lang}")
        print(f"💾 语言配置已保存到 data/multi30k/lang_config.txt")
        
        print("💡 数据已准备就绪，可以直接使用Hugging Face datasets进行训练")
        
        return src_lang, tgt_lang
        
    except Exception as e:
        print(f"❌ 下载数据集时出错: {e}")
        print("💡 请确保已安装datasets库: pip install datasets")
        return None, None


def get_language_columns(dataset):
    """
    根据数据集结构确定源语言和目标语言的列名
    Multi30k数据集通常以字典形式存储翻译对
    """
    sample = dataset['train'][0]
    
    # 直接检查是否包含标准的en/de列名
    if 'en' in sample and 'de' in sample:
        return 'en', 'de'
    
    # 检查是否包含translation字段
    if 'translation' in sample:
        translation_dict = sample['translation']
        keys = list(translation_dict.keys())
        
        # 确保至少有两种语言
        if len(keys) >= 2:
            # 优先选择英语和德语
            if 'en' in keys and 'de' in keys:
                return 'en', 'de'
            # 如果没有英语和德语，使用前两种语言
            else:
                return keys[0], keys[1]
    
    # 如果以上都不匹配，打印更多数据样本以帮助调试
    print(f"⚠️  无法自动检测语言列名，数据结构: {sample}")
    print("💡 请检查数据集格式并相应地调整代码")
    
    # 默认返回en/de，但实际使用时需要根据具体数据格式调整
    return 'en', 'de'


def load_prepared_dataset():
    """加载已准备的数据集"""
    dataset_path = 'data/multi30k/dataset'
    lang_config_path = 'data/multi30k/lang_config.txt'
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"⚠️  数据集不存在于 {dataset_path}")
        print("💡 请先运行 prepare_data_for_training() 下载数据集")
        return None, None, None
    
    try:
        # 从磁盘加载数据集
        dataset = load_from_disk(dataset_path)
        
        # 加载语言配置
        with open(lang_config_path, 'r') as f:
            src_lang, tgt_lang = f.read().strip().split(',')
        
        print(f"✅ 数据集加载成功")
        print(f"   源语言: {src_lang}")
        print(f"   目标语言: {tgt_lang}")
        
        return dataset, src_lang, tgt_lang
    except Exception as e:
        print(f"❌ 加载数据集时出错: {e}")
        return None, None, None


def check_dataset_format():
    """检查数据集格式并提供信息"""
    print("🔍 检查Multi30k数据集格式...")
    try:
        # 加载小部分数据进行格式检查
        dataset = load_dataset("bentrevett/multi30k", split='train[:5]')  # 只加载前5个样本
        print(f"   数据集列名: {dataset.column_names}")
        print(f"   样本数据: {dataset[0]}")
    except Exception as e:
        print(f"   检查数据集格式时出错: {e}")


# 添加函数调用以执行数据准备
if __name__ == "__main__":
    # 检查数据集是否已存在
    if os.path.exists('data/multi30k/dataset'):
        print("📂 数据集已存在，跳过下载")
        dataset, src_lang, tgt_lang = load_prepared_dataset()
    else:
        src_lang, tgt_lang = prepare_data_for_training()
