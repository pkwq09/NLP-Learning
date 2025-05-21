import torch
from config import get_config, get_weights_file_path
from train import get_model
from dataset import causal_mask
from transformers import BertTokenizer
# 评估模型
def evaluate_models(sentence,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_src = BertTokenizer.from_pretrained('./dataset/vocab-chinese')
    tokenizer_tgt = BertTokenizer.from_pretrained('./dataset/vocab-english')
    tokenizer_src.clean_up_tokenization_spaces = False
    tokenizer_tgt.clean_up_tokenization_spaces = False
    # 加载训练好的模型
    config = get_config()
    model = get_model(config).to(device)

    for epoch in range(epochs):
        model_filename = get_weights_file_path(config, f'0{str(epoch)}')

        state = torch.load(model_filename, weights_only=True)
        model.load_state_dict(state['model_state_dict'])  # 恢复权重
        model.eval()
        with torch.no_grad():

            # 编码器输入，分词，填充或截断，编码器端的掩码矩阵
            enc_input_tokens = tokenizer_src(
                sentence,
                padding='max_length',
                truncation=True,
                max_length=config['src_seq_len'],
                add_special_tokens=True,
                return_tensors='pt')

            encoder_input = enc_input_tokens['input_ids'].to(device)
            encoder_mask = enc_input_tokens['attention_mask'].unsqueeze(0).int().to(device)
            encoder_output = model.encode(encoder_input, encoder_mask)

            # 初始化解码器输入的起始 token
            decoder_input = torch.empty(1, 1) \
                .fill_(tokenizer_tgt.cls_token_id) \
                .type_as(encoder_input) \
                .to(device)

            # 解码器自回归逐个生成下一个字词
            while decoder_input.size(1) < config['tgt_seq_len']:
                # 动态调整解码器掩码矩阵
                decoder_mask = \
                    causal_mask(decoder_input.size(1)) \
                        .type_as(encoder_mask) \
                        .to(device)

                out = model.decode(decoder_input,
                                   encoder_output,
                                   encoder_mask,
                                   decoder_mask)

                # 预测下一个词
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

                # # 如果预测到的下一个词是[SEP]，则结束推理
                if next_word == tokenizer_tgt.sep_token_id:
                    break

                # 返回解码得到的下一个字词
                yield tokenizer_tgt.decode([next_word.item()], skip_special_tokens=False)

if __name__ == '__main__':
    sentence = '世界将走向何方？'
    print(f'输入的中文：{sentence}')
    print(f'翻译成英文：', end='')
    for next_word in evaluate_models(sentence, 8):
        print(f'{next_word}', end=' ')
