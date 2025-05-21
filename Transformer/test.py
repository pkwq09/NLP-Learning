import torch
from config import get_config,latest_weights_file_path
from train import get_model
from transformers import BertTokenizer
from dataset import get_dataset
from train import run_validation

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_src = BertTokenizer.from_pretrained('./dataset/vocab-chinese')
    tokenizer_tgt = BertTokenizer.from_pretrained('./dataset/vocab-english')
    tokenizer_src.clean_up_tokenization_spaces = False
    tokenizer_tgt.clean_up_tokenization_spaces = False
    config = get_config()
    model = get_model(config).to(device)

    train_loader, val_loader, test_loader = get_dataset(config)

    # 加载最后训练的模型
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    # 创建一个 SummaryWriter 实例，指定日志文件目录，用于Tensorboard可视化
    run_validation(model,
                   test_loader,
                   tokenizer_tgt,
                   config['tgt_seq_len'],
                   device,
                   lambda msg: print(msg),
                   0,
                   None,
                   num_examples=32)



