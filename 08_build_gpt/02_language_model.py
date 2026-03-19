"""
====================================================================
第8章 · 第2节 · 语言模型原理
====================================================================

【一句话总结】
语言模型学习"给定前面的词，预测下一个词"——GPT 就是一个巨大的语言模型，
它的全部训练目标就是 Next Token Prediction。

【为什么深度学习需要这个？】
- GPT = Generative Pre-trained Transformer，核心就是语言模型
- "预测下一个token" 看似简单，却能涌现出推理、翻译、编程等能力
- 理解语言模型的训练目标是理解 LLM 的基础

【核心概念】

1. 语言模型的定义
   - P(w_1, w_2, ..., w_n) = Π P(w_t | w_1, ..., w_{t-1})
   - 自回归分解：每个词的概率依赖于前面所有词
   - 训练目标：最大化训练数据的似然（等价于最小化交叉熵）

2. Next Token Prediction
   - 输入："今天天气" → 模型预测 → "真好"
   - 训练时：已知整个序列，用前缀预测下一个token
   - 推理时：逐个生成，每个新token加入前缀继续预测

3. 困惑度（Perplexity）
   - PPL = exp(cross_entropy_loss)
   - 直觉：模型在每个位置"平均犹豫多少个选项"
   - PPL=1 完美预测，PPL=V 完全随机（V是词汇表大小）

4. Teacher Forcing
   - 训练时输入真实序列（不用模型自己的预测）
   - 可以并行训练所有位置（因果掩码保证不偷看）
   - 推理时才需要自回归逐步生成

5. 数据准备
   - 长文本切分为固定长度的窗口（context length）
   - 输入：tokens[:-1]，标签：tokens[1:]（右移一位）
   - 一个序列可以同时训练多个预测任务

【前置知识】
第8章第1节 - 分词，第7章 - Transformer

====================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第1部分：自回归分解
# ============================================================
print("=" * 60)
print("第1部分：自回归分解")
print("=" * 60)

# 语言模型的核心思想：
# P("我 爱 深度 学习") = P("我") × P("爱"|"我") × P("深度"|"我 爱") × P("学习"|"我 爱 深度")
#
# 每一步只需要预测一个词的概率分布
# 这就是"自回归"(autoregressive)的含义：每一步依赖前面的输出

# 用一个简单例子演示
vocab = ["<BOS>", "我", "爱", "深度", "学习", "机器", "自然", "语言", "<EOS>"]
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}

sentence = ["<BOS>", "我", "爱", "深度", "学习", "<EOS>"]
print(f"\n句子: {' '.join(sentence)}")
print(f"词汇表大小: {vocab_size}")

# 自回归分解可视化
print("\n自回归分解:")
for t in range(1, len(sentence)):
    context = " ".join(sentence[:t])
    target = sentence[t]
    print(f"  P(\"{target}\" | \"{context}\")")

print("\n总概率 = 以上所有条件概率的乘积")


# ============================================================
# 第2部分：N-gram 语言模型（最简单的基线）
# ============================================================
print("\n" + "=" * 60)
print("第2部分：Bigram 语言模型")
print("=" * 60)

# Bigram: P(w_t | w_{t-1})，只看前一个词
# 这是最简单的语言模型，用来建立直觉

class BigramLM:
    """
    Bigram 语言模型：只看前一个词预测下一个词
    P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        # 转移计数矩阵：counts[i][j] = 词i后面出现词j的次数
        self.counts = np.zeros((vocab_size, vocab_size))

    def train(self, sequences):
        """从数据中统计bigram计数"""
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.counts[seq[i]][seq[i + 1]] += 1
        # 加1平滑（Laplace smoothing），避免零概率
        self.counts += 1

    def predict(self, prev_token):
        """给定前一个词，返回下一个词的概率分布"""
        row = self.counts[prev_token]
        return row / row.sum()

    def generate(self, start_token, max_len=10):
        """自回归生成"""
        tokens = [start_token]
        for _ in range(max_len):
            probs = self.predict(tokens[-1])
            next_token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(next_token)
            if next_token == word2idx["<EOS>"]:
                break
        return tokens

# 训练数据
train_data = [
    [word2idx[w] for w in ["<BOS>", "我", "爱", "深度", "学习", "<EOS>"]],
    [word2idx[w] for w in ["<BOS>", "我", "爱", "机器", "学习", "<EOS>"]],
    [word2idx[w] for w in ["<BOS>", "我", "爱", "自然", "语言", "<EOS>"]],
]

bigram = BigramLM(vocab_size)
bigram.train(train_data)

print("\nBigram 转移概率（部分）:")
for w in ["<BOS>", "我", "爱"]:
    idx = word2idx[w]
    probs = bigram.predict(idx)
    top3 = np.argsort(probs)[-3:][::-1]
    print(f"  P(? | \"{w}\") → ", end="")
    print(", ".join(f"\"{vocab[t]}\": {probs[t]:.2f}" for t in top3))

# 生成几个句子
print("\nBigram 生成的句子:")
for i in range(3):
    tokens = bigram.generate(word2idx["<BOS>"])
    print(f"  {' '.join(vocab[t] for t in tokens)}")

# Bigram 的局限性
print("\n【Bigram 的局限】")
print("  - 只看前1个词，无法捕捉长距离依赖")
print("  - '我 爱 深度 ___' → Bigram 只看 '深度'，不知道前面有 '爱'")
print("  - 解决方案：用神经网络看更长的上下文 → 神经语言模型")


# ============================================================
# 第3部分：数据准备（滑动窗口）
# ============================================================
print("\n" + "=" * 60)
print("第3部分：数据准备 — 滑动窗口")
print("=" * 60)

# 语言模型的数据准备非常简单：
# 给定一段文本的token序列，用滑动窗口切分为 (输入, 标签) 对
# 输入 = tokens[i:i+L]，标签 = tokens[i+1:i+L+1]（右移一位）

class TextDataset(torch.utils.data.Dataset):
    """
    将文本转换为语言模型训练数据

    关键理解：
    - 输入序列: [t_0, t_1, t_2, ..., t_{L-1}]
    - 标签序列: [t_1, t_2, t_3, ..., t_L]
    - 每个位置都在预测"下一个token"
    - 一个长度为L的序列，可以同时训练L个预测任务！
    """
    def __init__(self, text, seq_len=32):
        # 简单字符级分词
        self.chars = sorted(set(text))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.chars)

        # 将文本转为token id序列
        self.data = torch.tensor([self.char2idx[c] for c in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]        # 输入
        y = self.data[idx + 1:idx + self.seq_len + 1]  # 标签（右移一位）
        return x, y

    def decode(self, ids):
        return "".join(self.idx2char[i.item() if hasattr(i, 'item') else i] for i in ids)

# 演示
sample_text = "深度学习是人工智能的重要分支。深度学习通过多层神经网络来学习数据的表示。"
dataset = TextDataset(sample_text, seq_len=8)

print(f"\n文本: {sample_text}")
print(f"词汇表大小: {dataset.vocab_size}")
print(f"序列长度: {dataset.seq_len}")
print(f"训练样本数: {len(dataset)}")

# 展示几个训练样本
print("\n训练样本示例（输入 → 标签）:")
for i in range(min(3, len(dataset))):
    x, y = dataset[i]
    print(f"  输入: \"{dataset.decode(x)}\"")
    print(f"  标签: \"{dataset.decode(y)}\"")
    print(f"  （每个位置预测右边一个字符）")
    print()


# ============================================================
# 第4部分：困惑度（Perplexity）
# ============================================================
print("=" * 60)
print("第4部分：困惑度")
print("=" * 60)

def compute_perplexity(loss):
    """
    困惑度 = exp(交叉熵损失)

    直觉理解：
    - PPL = 1：模型完美预测每个token（不可能达到）
    - PPL = 10：模型平均在10个选项中犹豫
    - PPL = V：等价于随机猜测（V是词汇表大小）

    实际参考值：
    - GPT-2 在 WikiText-103 上 PPL ≈ 20
    - GPT-3 在 Penn Treebank 上 PPL ≈ 15
    - 人类在英文文本上 PPL ≈ 12（香农实验）
    """
    return np.exp(loss)

# 演示不同损失对应的困惑度
print("\n损失值 → 困惑度:")
for loss in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]:
    ppl = compute_perplexity(loss)
    print(f"  CE Loss = {loss:.1f} → PPL = {ppl:.1f}")

# 随机猜测的困惑度
random_loss = np.log(dataset.vocab_size)
random_ppl = compute_perplexity(random_loss)
print(f"\n随机猜测（词汇表={dataset.vocab_size}）:")
print(f"  CE Loss = ln({dataset.vocab_size}) = {random_loss:.2f}")
print(f"  PPL = {random_ppl:.1f}（和词汇表大小相等，最差情况）")


# ============================================================
# 第5部分：Teacher Forcing vs 自回归
# ============================================================
print("\n" + "=" * 60)
print("第5部分：Teacher Forcing vs 自回归生成")
print("=" * 60)

print("""
【Teacher Forcing（训练时）】
  输入:  [<BOS>] [我]   [爱]   [深度]
  标签:  [我]   [爱]   [深度] [学习]
  ← 每个位置的输入都是"真实的上一个词"
  ← 所有位置可以并行计算！（配合因果掩码）

【自回归生成（推理时）】
  步骤1: 输入 [<BOS>]       → 预测 [我]
  步骤2: 输入 [<BOS>, 我]    → 预测 [爱]
  步骤3: 输入 [<BOS>, 我, 爱] → 预测 [深度]
  ...
  ← 每个位置的输入是"模型自己上一步的预测"
  ← 必须串行，一步一步来

【关键区别】
  训练：并行（一次前向传播，所有位置同时预测）
  推理：串行（逐token生成）
  这也是为什么训练比推理快得多
""")


# ============================================================
# 第6部分：简单神经语言模型
# ============================================================
print("=" * 60)
print("第6部分：神经语言模型训练")
print("=" * 60)

class SimpleLM(nn.Module):
    """
    一个极简的神经语言模型

    结构：Embedding → LSTM → Linear → Logits
    这不是 Transformer，但展示了语言模型的通用训练流程
    （第7-8章的 GPT 用 Transformer 替换 LSTM，其他完全一样）
    """
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) → logits: (batch, seq_len, vocab_size)
        emb = self.embedding(x)           # (batch, seq_len, embed_dim)
        hidden, _ = self.lstm(emb)        # (batch, seq_len, hidden_dim)
        logits = self.head(hidden)        # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, start_ids, max_new_tokens=50, temperature=1.0):
        """自回归生成"""
        self.eval()
        ids = start_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids)
            # 只看最后一个位置的预测
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

# 准备训练数据
train_text = """深度学习是机器学习的一个分支。
深度学习使用多层神经网络来处理数据。
神经网络可以自动学习数据中的特征和模式。
通过反向传播算法，神经网络不断优化自身的参数。
深度学习在图像识别、自然语言处理等领域取得了突破性进展。
Transformer 是一种基于注意力机制的神经网络架构。
GPT 是一个基于 Transformer 的语言模型。
语言模型的目标是预测下一个词。"""

dataset = TextDataset(train_text, seq_len=16)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# 创建模型
model = SimpleLM(dataset.vocab_size, embed_dim=32, hidden_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

param_count = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {param_count:,}")
print(f"词汇表大小: {dataset.vocab_size}")
print(f"训练样本数: {len(dataset)}")

# 训练
losses = []
num_epochs = 100
print(f"\n开始训练（{num_epochs} epochs）...")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    n_batches = 0

    for x, y in dataloader:
        logits = model(x)  # (batch, seq_len, vocab_size)
        # CrossEntropyLoss 需要 (batch*seq_len, vocab_size) 和 (batch*seq_len,)
        loss = criterion(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)

    if (epoch + 1) % 20 == 0:
        ppl = compute_perplexity(avg_loss)
        print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | PPL: {ppl:.1f}")

        # 生成一段文本
        start = torch.tensor([[dataset.char2idx.get("深", 0)]])
        generated = model.generate(start, max_new_tokens=20, temperature=0.8)
        text = dataset.decode(generated[0])
        print(f"         生成: \"{text}\"")

# 损失曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("训练损失曲线")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
ppls = [compute_perplexity(l) for l in losses]
plt.plot(ppls)
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("困惑度曲线")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("08_02_training_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n损失曲线已保存: 08_02_training_curves.png")


# ============================================================
# 第7部分：为什么 Next Token Prediction 如此强大？
# ============================================================
print("\n" + "=" * 60)
print("第7部分：Next Token Prediction 的涌现能力")
print("=" * 60)

print("""
【看似简单的训练目标，为什么能产生智能？】

1. 预测下一个词需要理解世界
   "法国的首都是___" → 需要知道地理知识
   "2 + 3 = ___"     → 需要知道算术
   "def sort(arr):\\n___" → 需要知道编程

2. 上下文窗口越长，需要的推理越复杂
   "小明今天很开心，因为他昨天___" → 需要因果推理
   "翻译：Hello→你好，World→___" → 需要跨语言对应

3. 规模效应（Scaling Laws）
   - 模型越大 → 能捕捉更复杂的模式
   - 数据越多 → 能看到更多知识
   - 计算越多 → 学得越充分
   - 到一定规模后，"涌现"出训练中没有显式教过的能力

4. 从 Next Token → 通用智能
   ChatGPT 本质上仍然只在做"预测下一个token"
   但通过 RLHF 对齐后，它学会了：
   - 理解指令
   - 多轮对话
   - 拒绝有害请求
   - 承认不确定性

【Ilya Sutskever（OpenAI首席科学家）的名言】
"如果你能足够好地预测下一个token，你就在某种程度上理解了
产生这些token的底层现实。"
""")


# ============================================================
# 思考题
# ============================================================
print("=" * 60)
print("思考题")
print("=" * 60)

print("""
1. 为什么语言模型使用交叉熵损失而不是MSE损失？
   提示：语言模型的输出是离散概率分布（词汇表上的分布），
   MSE适用于连续值回归...

2. 如果一个语言模型的困惑度是10，这意味着什么？
   如果困惑度等于词汇表大小，又意味着什么？
   提示：PPL可以理解为"每步平均在多少个选项中犹豫"

3. Teacher Forcing 会导致什么问题？（Exposure Bias）
   提示：训练时总是看到"正确的"历史，但推理时可能产生错误，
   错误会累积...

4. 为什么 GPT 使用固定的上下文窗口长度（如4096/8192/128K）？
   为什么不能处理无限长的文本？
   提示：注意力机制的计算复杂度 O(n²)、KV Cache 的内存...

5. Next Token Prediction 和人类学习语言有什么相似和不同？
   提示：婴儿也在"预测下一个词"吗？人类还有哪些学习信号？
""")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了：
1. 语言模型的自回归分解: P(sequence) = Π P(token_t | context)
2. Bigram 模型作为最简单的基线
3. 滑动窗口数据准备: input=tokens[:-1], label=tokens[1:]
4. 困惑度 PPL = exp(CE_loss) 衡量模型质量
5. Teacher Forcing（训练）vs 自回归生成（推理）
6. 神经语言模型的完整训练流程
7. 为什么 Next Token Prediction 能产生智能

下一节：用 PyTorch 实现完整的 GPT 模型！
""")
