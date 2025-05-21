import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):  # 词向量层

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        return self.embedding(x.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):  # 位置编码层

    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 使用 torch.zeros 生成初值为 0 的位置矩阵，维度为：(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # 将 pe 的维度扩展为 (1, seq_len, d_model)，第一个维度，表示批次的大小
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # 创建位置张量 (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # 计算 position * (10000 ** (2i / d_model) ，2i用 torch.arange(0, d_model, 2) 生成
        angle = position / torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        # 对 pe 中偶数索引的列应用正弦函数
        pe[:, :, 0::2] = torch.sin(angle)
        # 对 pe 中奇数索引的列应用余弦函数
        pe[:, :, 1::2] = torch.cos(angle)
        # 使用 register_buffer 方法将 pe 注册为一个缓冲区
        self.register_buffer('pe', pe)

    def forward(self, x):
        position_encode = self.pe.requires_grad_(False)
         # 将位置编码与词向量相加， 得到的形状仍为 (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)  # dropout的目的是让输出更具泛化能力
        return x, position_encode


# 多头注意力模块，带掩码设置，也可计算交叉注意力
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # 模型维度
        self.heads = heads  # 头数
        # 需要确定 d_model 能被 heads 整除
        assert d_model % heads == 0, "d_model 不能被 heads 整除"

        self.d_k = d_model // heads  # 计算单个头的词向量长度
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Wq矩阵
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Wk矩阵
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Wv矩阵
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Wo矩阵
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(Q, K, V, mask, dropout: nn.Dropout):
        d_k = Q.shape[-1]  # 词向量长度
        # （1）根据论文公式计算注意力
        # (batch_size, heads, seq_len, d_k) --> (batch_size, heads, seq_len, seq_len)
        attention_matrix = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 掩码操作，根据 mask 矩阵，将 attention_matrix 掩码位置设为一个极小的数
            attention_matrix.masked_fill_(mask == 0, -1e9)
        # 在 (batch_size, heads, seq_len, seq_len) 最后一个维度上归一化
        attention_matrix = attention_matrix.softmax(dim=-1)
        if dropout is not None:
            attention_matrix = dropout(attention_matrix)
        # （2）加权求和
        # (batch_size, seq_len, seq_len) --> (batch_size, seq_len, d_v)
        X = torch.matmul(attention_matrix, V)
        # 返回注意力分数矩阵用于后续的可视化
        return X, attention_matrix

    # Query, Key, Value 表示来自上一层的输入
    def forward(self, Query, Key, Value, mask):
        # （1）上一层的输入映射为 Q、K、V 矩阵
        Q = self.W_q(Query)
        K = self.W_k(Key)
        V = self.W_v(Value)
        # （2）分头操作，调整矩阵维度
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, heads, d_k)
        # --> (batch_size, heads, seq_len, d_k)
        Q = Q.view(Q.shape[0], Q.shape[1], self.heads, self.d_k).transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.heads, self.d_k).transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.heads, self.d_k).transpose(1, 2)

        # （3）计算注意力，注意 Q、K、V 的矩阵维度此时是分头状态
        X, self.attention_matrix = MultiHeadAttention.attention(Q, K, V, mask, self.dropout)

        # （4）合并单头注意力，调整 Q、K、V 的矩阵维度
        # (batch_size, heads, seq_len, d_k) --> (batch_size, seq_len, heads, d_k)
        # --> (batch_size, seq_len, d_model)
        X = X.transpose(1, 2).contiguous().view(X.shape[0], -1, self.heads * self.d_k)

        # （5）多头注意力的最后一层是线性层，用 Wo 矩阵表示
        X = self.W_o(X)
        X = self.dropout(X)
        return X

class FeedForward(nn.Module):  # 前馈网络

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # 第一层宽度为 d_ff = 4 * d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_ff)
        x = self.linear_1(self.norm(x))
        x = torch.relu(x)
        x = self.dropout(x)
        # (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):  # 编码器的单层定义

    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)  # 多头自注意力
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈网络

    # 编码器输入 x 和 掩码矩阵 src_mask
    def forward(self, x, src_mask):
        # 多头注意力计算，注意力残差块
        x = x + self.self_attention(self.norm(x),  # Q
                                    self.norm(x),  # K
                                    self.norm(x),  # V
                                    src_mask)  # 填充掩码矩阵
        # 前馈网络计算，残差块
        x = x + self.feed_forward(self.norm(x))
        return x


class Encoder(nn.Module):  # 编码器

    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 num_layers: int = 6):

        super().__init__()
        # 创建编码器各层
        encoder_blocks = []
        for _ in range(num_layers):
            encoder_layer = EncoderLayer(d_model, heads, d_ff, dropout)
            encoder_blocks.append(encoder_layer)
        self.layers = nn.ModuleList(encoder_blocks)  # 编码器各层列表

    # 编码器输入 x 和 填充掩码矩阵 src_mask
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)  # 连接各个编码器层
        return x


class DecoderLayer(nn.Module):  # 解码器的单层定义

    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)  # 掩码多头自注意力
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)  # 交叉多头注意力
        self.feed_forward = FeedForward(d_model, d_ff, dropout)  # 前馈网络

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 多头自注意力计算， 注意力残差块，输入的是 Q 、K、V，tgt_mask 解码器掩码
        x = x + self.self_attention(self.norm(x), self.norm(x), self.norm(x), tgt_mask)
        # 交叉多头注意力计算， 注意力残差块
        x = x + self.cross_attention(self.norm(x),  # 解码器上一层输出的 Q矩阵
                                     self.norm(encoder_output),  # 编码器输出的 K矩阵
                                     self.norm(encoder_output),  # 编码器输出的 V矩阵
                                     src_mask)  # src_mask 表示填充掩码
        # 前馈网络计算， 残差块
        x = x + self.feed_forward(self.norm(x))
        return x


class Decoder(nn.Module):  # 解码器

    def __init__(self,
                 d_model: int,
                 heads: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 num_layers: int = 6):

        super().__init__()
        # 创建解码器各层
        decoder_blocks = []
        for _ in range(num_layers):
            decoder_layer = DecoderLayer(d_model, heads, d_ff, dropout)
            decoder_blocks.append(decoder_layer)
        self.layers = nn.ModuleList(decoder_blocks)  # 解码器各层列表

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)  # 连接各层
        return x

 # 字典映射层，将解码器输出的词向量映射为词典长度的向量
class OutputLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # 映射回词典空间

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):  # Transformer 模型定义

    def __init__(self,
                 src_vocab_size: int,  # 源词典大小
                 tgt_vocab_size: int,  # 目标词典大小
                 src_seq_len: int,  # 源序列最大长度
                 tgt_seq_len: int,  # 目标序列最大长度
                 d_model: int = 512,  # 模型宽度
                 heads: int = 8,  # 注意力头数
                 num_encoder_layers: int = 6,  # 编码器层数
                 num_decoder_layers: int = 6,  # 解码器层数
                 d_ff: int = 2048,  # 前馈网络第一层的宽度
                 dropout: float = 0.1):
        super().__init__()
        # 创建词向量输入层
        self.src_embed = InputEmbeddings(src_vocab_size, d_model)
        self.tgt_embed = InputEmbeddings(tgt_vocab_size, d_model)

        # 创建词向量位置编码层
        self.src_pos = PositionalEncoding(src_seq_len, d_model, dropout)
        self.tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)
        # 创建编码器
        self.encoder = Encoder(d_model, heads, d_ff, dropout, num_encoder_layers)
        # 创建解码器
        self.decoder = Decoder(d_model, heads, d_ff, dropout, num_decoder_layers)
        # 词向量投射到字典空间
        self.project_layer = OutputLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):  # 编码器编码过程
        # (batch, seq_len, d_model)
        src = self.src_embed(src)  # 编码器词嵌入
        src, _ = self.src_pos(src)  # 编码器的位置编码
        encoder_output = self.encoder(src, src_mask)  # 编码器推理
        return encoder_output  # 输出的形状维度：(batch_size, seq_len, d_model)

    def decode(self,
               tgt: torch.Tensor,
               encoder_output: torch.Tensor,
               src_mask: torch.Tensor,
               tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)  # 解码器词嵌入
        tgt, _ = self.tgt_pos(tgt)  # 解码器的位置编码
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)  # 解码器推理
        return decoder_output  # 输出的形状维度：(batch_size, seq_len, d_model)

    def output_layer(self, x):
        # (batch, seq_len, vocab_size)
        return self.project_layer(x)