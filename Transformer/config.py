from pathlib import Path

# 定义模型参数并管理训练过程中的模型保存和加载
def get_config():
    return {
        'src_vocab_size': 21128,   # BERT中文词典大小
        'tgt_vocab_size': 30522,   # BERT英文词典大小
        'batch_size': 32,  # 批次大小，设为8
        'num_epochs': 2,  # 总共训练的轮数，算力充足可设为 20 以上
        'lr': 10 ** -4,  # 学习率，设为 0.0001
        'src_seq_len': 60,  # 中文序列最大长度
        'tgt_seq_len': 60,  # 英文序列最大长度
        'd_model': 512,  # 模型的维度，即词嵌入的维度，设为512
        'heads': 8,  # 注意力头数
        'num_encoder_layers': 6,  # 编码器层数
        'num_decoder_layers': 6,  # 解码器层数
        'd_ff': 2048,  # 前馈网络第一层宽度
        'dropout': 0.1,  # Dropout参数
        'dataset': './dataset/zh_en_pairs.csv',  # 总的数据集路径与名称
        'train': './dataset/train.csv',   # 训练集
        'valid': './dataset/valid.csv',  # 验证集
        'test': './dataset/test.csv',  # 测试集
        'model_folder': 'weights',  # 模型存放的文件夹名称，设为 'weights'
        'model_basename': 'tmodel_',  # 模型文件名称前缀，设为 'tmodel_'
        'preload': 'latest',  # 预加载的权重文件，设为 'latest'（表示最新的权重文件）
        'experiment_name': 'runs/tmodel'  # 运行的中间结果保存的路径，设为 'runs/tmodel'
    }

# 根据配置和特定的 epoch（轮次）生成模型权重文件的路径
# 返回一个字符串，表示权重文件的完整路径。
def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# 查找并返回权重文件夹中最新的权重文件路径
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
