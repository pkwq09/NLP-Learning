import torch
import torch.nn as nn
from torch.nn import functional as F


# 位置编码
class PositionEncoding(nn.Module):

    def __init__(self, max_len: int = 1024, d_model: int = 512):
        super().__init__()

        # 初始化一个零矩阵 pe，形状为 (seq_len, d_model)，用于存储位置编码
        pe = torch.zeros(max_len, d_model)

        # 生成从 0 到 max_len-1 的整数序列，表示位置索引。将其变为列向量，形状为 (max_lenn, 1)     
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)

        ## 位置编码计算公式：
        ## PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        ## PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        # 计算频率缩放因子
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)

        # 应用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  ## every other column, starting with the 1st, has sin() values
        pe[:, 1::2] = torch.cos(position * div_term)  ## every other column, starting with the 2nd, has cos() values
        pe = pe.unsqueeze(0)  # 在第0维添加一个维度，变成 (1, max_len, d_model)

        ## 注册位置编码矩阵
        self.register_buffer('pe', pe)

        # 词向量叠加位置编码向量
    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:, :word_embeddings.size(1), :]

# 单头注意力计算
class Head(nn.Module):
    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # 线性变换得到 K 值
        K = self.key(x)   # (batch, seq_len, head_size)
        # 线性变换得到 Q 值
        Q = self.query(x) # (batch, seq_len, head_size)
        # 计算注意力得分
        # (batch, seq_len, head_size)->(batch, seq_len, seq_len)
        wei = Q @ K.transpose(-2,-1) * d_model**-0.5 
        # 掩码操作，以确保每个时间步只能看到当前或之前的词汇
        # (batch, seq_len, seq_len)
        mask = torch.tril(torch.ones((1, seq_len, seq_len))).to(x.device)
        wei = wei.masked_fill( mask == 0, float('-inf'))
        # 通过 softmax 函数将注意力得分归一化，使得每个查询的注意力得分变成一个概率分布
        wei = F.softmax(wei, dim=-1) # (batch, seq_len, seq_len)
        wei = self.dropout(wei)
        # 线性变换得到 V 值
        V = self.value(x) # (batch, seq_len, head_size)
        # wei @ v 是对值（Value）的加权求和，得到最终的输出 out，形状为 (batch, seq_len, head_size)
        # # (batch, seq_len, seq_len) @ (batch, seq_len, head_size) -> (batch, seq_len, head_size)
        out = wei @ V  
        return out

# 多个注意力头并行计算，然后将其结果合并
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, block_size, dropout):
        super().__init__()

        # 通过 ModuleList 创建多个 Head 实例，每个头的大小为 head_size，共有 num_heads 个头
        # ModuleList 允许将这些子模块（即多个头）一起管理
        head_size = d_model // num_heads
        self.heads = nn.ModuleList([Head(d_model, head_size, block_size, dropout) for _ in range(num_heads)])
       
        # 将多个注意力头的输出拼接起来后映射回原始的嵌入空间
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 对每个注意力头 h(x) 计算得到的输出进行拼接，拼接的维度是最后一维（dim=-1）。
        # 每个头的输出 (batch, seq_len, head_size) 
        # 会被拼接成一个更大的张量 (batch, seq_len, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
        
# 前馈网络
class FeedFoward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),    # GPT3由ReLU替换为GeLU
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# 单层解码器
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout):
        # d_model: 词向量长度, num_heads: 注意力头数, block_size:最大文本块长度
        super().__init__()
        self.ma = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        self.ffwd = FeedFoward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)  # 层标准化
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))  # 此处先标准化，再计算多头注意力
        x = x + self.ffwd(self.ln2(x)) # 先层标准化，再前馈网络
        return x


# GPT类封装
class GPT(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 d_model: int = 512, 
                 n_layer: int = 6, 
                 n_head: int = 8, 
                 block_size:int = 32,
                 dropout:int = 0.1):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, d_model) # 定义输入层
        self.position = PositionEncoding(d_model = d_model)  # 定义位置编码层       
        self.blocks = nn.Sequential(*[DecoderBlock(d_model, 
                                                   num_heads=n_head, 
                                                   block_size=block_size, dropout = dropout) \
                                      for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model) # 输出层标准化
        self.output = nn.Linear(d_model, vocab_size)  # 定义输出层
        # 输出层与词嵌入层共享参数，这和 LLama、通义千问等开源模型类似
        # self.embedding.weight = self.output.weight

    def forward(self, idx):       
        idx = idx.long()  # 将索引转换为 LongTensor
        # (batch, seq_len) --> (batch, seq_len, d_model)
        x = self.embedding(idx)
        x = self.position(x)
        x = self.blocks(x)   #  (batch, seq_len, d_model)
        x = self.ln(x)   #  (batch, seq_len, d_model)
        logits = self.output(x)   # (batch, seq_len,,vocab_size)

        return logits
              
    # 模型自回归持续推理，生成长度为 max_new_tokens 的文本块
    def generate(self, idx, max_new_tokens):
        # idx 的维度为： (batch, seq_len) ，表示当前上下文token在词典中的索引（indices）
        # max_new_tokens表示新生成的文本的最大长度
        for _ in range(max_new_tokens):
            # 切片取出每个样本最后 block_size 个 tokens 元素
            idx_cond = idx[:, -self.block_size:]
            # 模型正向传播
            logits = self(idx_cond)  
            # 选择每个批次中序列的最后一个时间步
             # (batch, seq_len, d_model)->(batch, d_model)
            logits = logits[:, -1, :]
            # 沿着最后一维进行 softmax 归一化，将原始得分转换为概率分布
            probs = F.softmax(logits, dim=-1) # (batch, d_model)
            # 根据概率分布probs采样下一个字词的索引
            idx_next = torch.multinomial(probs, num_samples=1) # (batch,1)
            if (idx_next == 102):   # 遇到[SEP]结束推理
                break
            # 追加到已有的字词序列中，将两个张量沿着指定的维度连接起来
            idx = torch.cat((idx, idx_next), dim=1) # (batch, seq_len+1)
            yield idx_next

if __name__ =="__main__":
    # 参数设置
    random_seed = 2025    # 随机数种子
    vocab_size = 21128   # 词典大小
    batch_size = 16
    block_size = 32     # 模型最长文本推理能力
    train_steps = 1000  # 训练步数
    eval_interval = 200
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200  # 评估迭代步数
    d_model = 512
    n_head = 8
    n_layer = 6
    dropout = 0.1
    # ------------
    
    model = GPT(vocab_size, 
                d_model, 
                n_layer=n_layer, 
                n_head=n_head, 
                block_size=block_size,
                dropout = dropout)
    # 初始化模型参数，对于维度大于 1 的参数，使用 Xavier 均匀初始化方法进行初始化
    for p in model.parameters():  # 遍历模型中的可学习参数 
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    model = model.to(device)
    
    # 优化算法
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    from torchsummary import summary
    
    print(summary(model, (64,)))  # 输入形状应该是 (block_size, ),即文本的最大长度 block_size
    # 打印模型参数
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
    
    
    torch.manual_seed(random_seed)

    # 读文件
    with open('./data/people.cn/news.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    from transformers import BertTokenizer
    
    # 加载 BERT 中文词典
    bert_tokenizer = BertTokenizer.from_pretrained("./BERT中文词典")
    
    # 分词
    bert_tokens = bert_tokenizer.tokenize(corpus)
    del corpus
    # Token 转 ID
    data = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
    
    
    # 数据集划分
    # 将数据集分为训练集和验证集两部分。训练集占比90%，验证集占比10%
    data = torch.tensor(data, dtype=torch.long)   # 数据集转换为 Pytorch 张量
    n_train = int(0.9 * len(data))   # 计算训练集样本数
    train_data = data[:n_train]   # n_train之前的给训练集
    val_data = data[n_train:]
    
    
    # 从数据集中随机抽取一批样本, 返回文本块和标签
    def get_batch(dataset, batch_size=32, block_size=1024, device='cpu'):
        # 根据随机样本的抽取范围和批次大小随机抽取，返回每个文本块的起始索引
        ix = torch.randint(len(dataset) - block_size, (batch_size,))
        # 把本批次的文本块堆叠在一起，形成文本矩阵
        x = torch.stack([dataset[i:i + block_size] for i in ix], dim=0)  # 按行堆叠
        # 标签堆叠在一起，形成标签矩阵
        y = torch.stack([dataset[i + 1:i + block_size + 1] for i in ix], dim=0)
        x, y = x.to(device), y.to(device)
        return x, y

    ## ======================开始训练模型===========================
    # 训练
    his_train_loss = []
    his_val_loss = []
    train_losses =[]
    for step in range(train_steps):
        # 随机抽取一批样本
        batchX, batchY = get_batch(train_data, 
                                   batch_size = batch_size, 
                                   block_size = block_size,
                                   device = device)
    
        # 正向传播，计算损失
        logits = model(batchX)
        batch, seq_len, d_model = logits.shape
        logits = logits.view(batch*seq_len, d_model)
        batchY = batchY.view(batch*seq_len)
    
        loss = F.cross_entropy(logits, batchY)
        train_losses.append(loss.item())
        
        optimizer.zero_grad(set_to_none=True) # 累积梯度清零
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 优化算法更新参数
    
         # 间隔 eval_interval 步评估模型表现
        if step % eval_interval == 0 or step == train_steps - 1:
            # 模型此时在验证集上的损失表现
            model.eval()
            val_losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(val_data, 
                                 batch_size = batch_size, 
                                 block_size = block_size,
                                 device = device)
                val_logits = model(X)
                b, s, d = val_logits.shape
                val_logits = val_logits.view(b * s, d)
                Y = Y.view(b * s)
                val_loss = F.cross_entropy(val_logits, Y)
                val_losses[k] = val_loss.item()     
            model.train()
    
            # 记录间隔 eval_interval 步的损失
            his_train_loss.append(torch.tensor(train_losses).mean()) 
            his_val_loss.append(val_losses.mean())
            print(f"step {step}: train loss {torch.tensor(train_losses).mean():.4f}, \
                  val loss {val_losses.mean():.4f}")
            train_losses = []
    # 用训练好的模型生成一篇新稿件
    idx = torch.zeros((1, 1), dtype = torch.long, device = device)
    for idx_next in model.generate(idx, max_new_tokens=500):
    # 将生成的token解码为文本
        print(f'{bert_tokenizer.decode(idx_next.item(), skip_special_tokens=False)}', end='')