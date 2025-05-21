import torch
import torch.nn as nn
from model import Transformer
from dataset import get_dataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path
from tqdm import tqdm
from pathlib import Path
from torchmetrics.text import CharErrorRate, WordErrorRate, BLEUScore
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

"""
功能: 贪心解码算法。从编码器输出中获取信息，在解码器中一步步生成翻译
步骤:
(1)输入源语言的句子，获取编码器输出。
(2)逐步生成目标语言的翻译。每次生成一个词，并将其作为输入传递给下一个时间步。
(3)当模型生成了[SEP]（结束符）或达到最大长度时，停止生成。
"""
def greedy_decode(model,
                  encoder_input,
                  encoder_mask,
                  tokenizer_tgt,
                  max_len,
                  device):

    # 预先计算编码器输出，用于解码器的每一步推理
    encoder_output = model.encode(encoder_input, encoder_mask)
    # 初始化解码器输入的起始 token
    decoder_input = torch.empty(1, 1)\
        .fill_(tokenizer_tgt.cls_token_id)\
        .type_as(encoder_input)\
        .to(device)

    # 解码器自回归逻辑，递归生成目标输出
    while True:
        # 解码器输入序列的长度达到最大值，则结束推理
        if decoder_input.size(1) == max_len:
            break

        # 动态调整解码器掩码矩阵
        decoder_mask = \
            causal_mask(decoder_input.size(1))\
            .type_as(encoder_mask)\
            .to(device)

        # 解码器推理一次，out的维度为（b，seq_len, d_model）
        out = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)

        # 得到下一个 token，
        prob = model.output_layer(out[:, -1])  # 用序列的最后一个嵌入向量
        _, next_word = torch.max(prob, dim=1)  # 返回最大概率值及其在词典中的索引

        # 新预测的词追加到解码器的输入序列末尾
        decoder_input = torch.cat(
            [decoder_input,
             torch.empty(1, 1)
                 .type_as(encoder_input)
                 .fill_(next_word.item())
                 .to(device)],
            dim=1
        )
        # 如果下一个词是[SEP]，则结束推理
        if next_word == tokenizer_tgt.sep_token_id:
            break

    return decoder_input.squeeze(0)

"""
功能: 在每个训练周期后进行验证，评估模型的翻译效果。
它计算了字符错误率（CER）、词错误率（WER）和BLEU分数。
"""
def run_validation(model,
                   validation_ds,
                   tokenizer_tgt,
                   max_len,
                   device,
                   print_msg,  # 输出消息的函数
                   global_step,
                   writer,
                   num_examples=32):
    model.eval()
    count = 0  # 计数用于验证的样本数

    expected = []  # 期望的目标语言文本
    predicted = []  # 预测的目标语言文本

    # 设置输出窗口的宽度
    console_width = 80

    with torch.no_grad():  # 禁用梯度计算
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # 确保当前批次大小的值为 1
            assert encoder_input.size(
                0) == 1, "验证数据集的批次大小应为1"

            model_out = greedy_decode(model,
                                      encoder_input,
                                      encoder_mask,
                                      tokenizer_tgt,
                                      max_len,
                                      device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            expected.append(target_text)
            predicted.append(model_out_text)

            # 输出显示结果
            print_msg('-' * console_width)
            print_msg(f"{f'源语言文本: ':>12}{source_text}")
            print_msg(f"{f'目标语言文本: ':>12}{target_text}")
            print_msg(f"{f'当前预测的文本: ':>12}{model_out_text}")

            if count == num_examples:  # 达到观察数量退出验证
                print_msg('-' * console_width)
                break

    if writer:
        # 计算字符错误率（Character Error Rate，简称 CER）
        # 编辑距离：计算所需的最小编辑操作（插入、删除、替换）数目
        # CER= 编辑距离/期望文本的字符数
        metric = CharErrorRate()
        cer = metric(' '.join(predicted), ' '.join(expected))
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # 计算词错误率 WER，类似 CER
        metric = WordErrorRate()
        wer = metric(' '.join(predicted), ' '.join(expected))
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # 计算 BLEU 得分，越高说明越接近标签文本
        metric = BLEUScore(n_gram=2)
        bleu = metric([' '.join(predicted)], [[' '.join(expected)]])
        print(bleu) # 显示验证集上一个Batch的BLEU得分

        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_model(config):
    # 实例化 Transformer 模型
    model = Transformer(config['src_vocab_size'],
                        config['tgt_vocab_size'],
                        config['src_seq_len'],
                        config['tgt_seq_len'],
                        config['d_model'],
                        config['heads'],
                        config['num_encoder_layers'],
                        config['num_decoder_layers'],
                        config['d_ff'],
                        config['dropout'])
    return model


def train_model(config):
    # 从本地加载BERT分词器
    chinese_tokenizer = BertTokenizer.from_pretrained('./dataset/vocab-chinese')
    english_tokenizer = BertTokenizer.from_pretrained('./dataset/vocab-english')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 根据 config 中的 model_folder 生成保存权重参数的目录路径
    # 如果目录不存在，会被创建。如果目录已经存在，则忽略
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = get_dataset(config)
    model = get_model(config).to(device)

    # 创建一个 SummaryWriter 实例，指定日志文件目录，用于Tensorboard可视化
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        eps=1e-9)

    # 如果存在预训练模型，则加载预训练模型
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    if preload == 'latest':
        model_filename = latest_weights_file_path(config)
    elif preload:
        model_filename = get_weights_file_path(config, preload)
    else:
        model_filename = None

    if model_filename:
        print(f'在预训练模型： {model_filename} 的基础上开始训练')
        state = torch.load(model_filename, weights_only=True)
        model.load_state_dict(state['model_state_dict']) # 恢复权重
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])  # 恢复优化算法
        global_step = state['global_step']
    else:
        print('没有预训练模型, 从头开始训练...')

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=chinese_tokenizer.pad_token_id,   # 计算损失时忽略 [PAD]
        label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, initial_epoch+config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_loader, desc=f'Epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)  # (b, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)  # (b, 1, seq_len, seq_len)

            # 1. 模型推理，正向传播
            # （1）先编码器， 输出：(b, seq_len, d_model)
            encoder_output = model.encode(encoder_input, encoder_mask)
            # （2）再解码器，输出： (b, seq_len, d_model)
            decoder_output = model.decode(decoder_input,
                                          encoder_output,
                                          encoder_mask,
                                          decoder_mask)
            # （3） Transformer输出层，输出 (b, seq_len, vocab_size)
            logits = model.output_layer(decoder_output)

            # 提取本批次的标签，维度 (b, seq_len)
            label = batch['label'].to(device)

            # 2. 计算交叉熵损失
            loss = loss_fn(logits.view(-1, config['tgt_vocab_size']), label.view(-1))
            # 当前批次损失值实时显示在进度条右侧
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # 损失值写入Tensorboard，用于绘制损失函数曲线
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # 3. 反向传播，计算梯度
            loss.backward()

            # 4. 更新梯度
            optimizer.step()
            # 清空累积的梯度，为下一步迭代做准备
            optimizer.zero_grad(set_to_none=True)

            global_step += 1  # 记录总训练步数

        # 每个Epoch结束后，在验证集上测试，观察模型泛化能力
        run_validation(model,
                       val_loader,
                       english_tokenizer,
                       config['tgt_seq_len'],
                       device,
                       lambda msg: batch_iterator.write(msg),
                       global_step,
                       writer)

        # 在每个 Epoch 结束后保存模型相关参数
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # 权重
            'optimizer_state_dict': optimizer.state_dict(),  # 优化算法
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)