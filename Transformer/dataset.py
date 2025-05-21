import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer   # pip install transformers
from sklearn.model_selection import train_test_split   # pip install scikit-learn
from config import get_config

class TranslationDataset(Dataset):
    """
    初始化源语言和目标语言的分词器。
    获取目标语言的 [CLS]（开始符）、[SEP]（结束符）和 [PAD]（填充符） token 的 ID
    通过 torch.tensor() 创建对应的 Tensor，确保数据类型是 int64 类型
    """

    def __init__(self,
                 data,  # 数据集，包含翻译对 DataFrame
                 tokenizer_src,  # 源语言的分词器
                 tokenizer_tgt,  # 目标语言的分词器
                 seq_max_len):  # 序列的最大长度，输入和输出的句子将被截断或填充到此长度

        super().__init__()
        self.df = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_max_len = seq_max_len

    def __len__(self):  # 返回数据集的大小，即数据集中翻译对的数量
        return len(self.df)


    def __getitem__(self, idx):
        # 通过索引获取中文和英文句子对
        src_target_pair = self.df.iloc[idx]
        # 源语言文本 src_text 和目标语言文本 tgt_text
        src_text = src_target_pair['Chinese']
        tgt_text = src_target_pair['English']

        #  源语言文本分词、填充或截断
        enc_input_tokens = \
            self.tokenizer_src(
                src_text,
                padding='max_length',
                truncation=True,
                max_length=self.seq_max_len,
                add_special_tokens = True,
                return_tensors='pt')
        # 目标语言文本分词、填充或截断
        dec_input_tokens = \
            self.tokenizer_tgt(
                tgt_text,
                padding='max_length',
                truncation=True,
                max_length=self.seq_max_len,
                add_special_tokens=False,
                return_tensors='pt')

        # 构造源语言的输入张量 encoder_input，按照行的方向上下拼接
        encoder_input = enc_input_tokens['input_ids'].squeeze(0)  # 提取 'input_ids' 张量
        decoder_input = dec_input_tokens['input_ids'].squeeze(0)

        # 构造目标语言的输入张量 decoder_input
        decoder_input = torch.cat(
            [
                # 添加开头标识
                torch.tensor([self.tokenizer_tgt.cls_token_id], dtype=torch.int64),
                # 目标语言句子的 token，去掉尾部的token，保持最大长度
                decoder_input[:-1],
            ],
            dim=0,
        )

        # 构造目标语言的标签张量 label
        # 获取分词后的 token IDs
        tokens = dec_input_tokens['input_ids'][0].tolist()

        # 只在句子结尾添加 [SEP]，并确保不加到填充部分
        sep_token_id = self.tokenizer_tgt.sep_token_id
        pad_token_id = self.tokenizer_tgt.pad_token_id

        # 查找句子实际内容的结束位置（排除填充部分）
        # 可以通过过滤掉填充标记 [PAD] (ID 为 0) 来找到句子内容的末尾
        tokens = [token for token in tokens if token != self.tokenizer_tgt.pad_token_id]

        # 在句子的末尾添加 [SEP] 标记
        tokens.append(sep_token_id)

        # 如果总长度超过最大长度，进行截断，否则填充
        if len(tokens) >= self.seq_max_len:
            tokens[self.seq_max_len-1] = sep_token_id  # 确保标签最后一个元素为[SEP]
            tokens = tokens[:self.seq_max_len]
        else:
            tokens.extend([pad_token_id] * (self.seq_max_len - len(tokens)))
        label = torch.tensor(tokens)

        # 确保张量大小正确
        assert encoder_input.size(0) == self.seq_max_len
        assert decoder_input.size(0) == self.seq_max_len
        assert label.size(0) == self.seq_max_len

        return {
            'encoder_input': encoder_input,  # (seq_max_len)
            'decoder_input': decoder_input,  # (seq_max_len)
            'encoder_mask':  enc_input_tokens['attention_mask'].unsqueeze(0).int(), # (1, 1, seq_max_len)
            'decoder_mask': (decoder_input != self.tokenizer_tgt.pad_token_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, seq_max_len) & (1, seq_max_len, seq_max_len),
            'label': label,  # (seq_max_len)
            'src_text': src_text,  # 源语言文本
            'tgt_text': tgt_text   # 目标语言文本
        }

# 解码器屏蔽未来信息的掩码矩阵
def causal_mask(size):
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.int))
    return mask

# 划分数据集
def split_dataset(config):
    # 加载 CSV 数据
    df = pd.read_csv(config['dataset'], encoding='ansi')

    # 1. 划分训练集和测试集（95% 训练集，5% 测试集）
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=2025)

    # 2. 从训练集中划分出验证集（95% 训练集，10% 验证集）
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2025)

    # 将划分后的数据集保存为新的 CSV 文件
    train_df.to_csv('./dataset/train.csv', encoding='ansi', index=False)
    val_df.to_csv('./dataset/valid.csv', encoding='ansi', index=False)
    test_df.to_csv('./dataset/test.csv', encoding='ansi', index=False)

def get_dataset(config):   # 返回数据集

    # 加载训练集、验证集和测试集
    train_df = pd.read_csv(config['train'], encoding='ansi')
    val_df = pd.read_csv(config['valid'], encoding='ansi')
    test_df = pd.read_csv(config['test'], encoding='ansi')

    # 加载BERT分词器
    chinese_tokenizer = BertTokenizer.from_pretrained('./dataset/vocab-chinese')  # 中文的 BERT 分词器
    english_tokenizer = BertTokenizer.from_pretrained('./dataset/vocab-english')  # 英文的 BERT 分词器(小写词典）
    chinese_tokenizer.clean_up_tokenization_spaces = False
    english_tokenizer.clean_up_tokenization_spaces = False
    # 用 Dataset 训练、验证、测试数据集
    train_dataset = TranslationDataset(train_df,
                                       chinese_tokenizer,
                                       english_tokenizer,
                                       config['src_seq_len']
                                       )
    val_dataset = TranslationDataset(val_df,
                                     chinese_tokenizer,
                                     english_tokenizer,
                                     config['src_seq_len']
                                     )
    test_dataset = TranslationDataset(test_df,
                                      chinese_tokenizer,
                                      english_tokenizer,
                                      config['src_seq_len']
                                      )

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=True)  # 设为False便于观察模型改善
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    config = get_config()
    split_dataset(config)
    train_loader, val_loader, test_loader = get_dataset(config)
    # 打印数据集大小
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    # 观察单个批次的数据
    for batch in train_loader:
        print('源语言文本：\n', batch['src_text'][0])
        print('\n编码器输入：\n',batch['encoder_input'][0])
        print('\n目标语言文本：\n', batch['tgt_text'][0])
        print('\n解码器输入：\n',batch['decoder_input'][0])
        print('\n标签：\n',batch['label'][0])
        print('\n编码器掩码矩阵：\n',batch['encoder_mask'][0])
        print('\n解码器掩码矩阵：\n',batch['decoder_mask'][0])
        break