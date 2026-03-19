"""
====================================================================
第4章 · 第3节 · Seq2Seq + 注意力机制（通向 Transformer 的桥梁）
====================================================================

【一句话总结】
Seq2Seq 引入了编码器-解码器框架，注意力机制让解码器能"回头看"
编码器的每个位置——这正是 Transformer 的直接前驱。

【为什么这一节特别重要？】
- Seq2Seq 的编码器-解码器架构被 Transformer 直接继承
- Bahdanau 注意力 → Transformer 的 Cross-Attention
- 理解了这里的注意力，Transformer 的 Self-Attention 就是一步之遥
- 这是从 RNN 时代到 Transformer 时代的关键过渡

【核心概念】

1. Seq2Seq（序列到序列）
   - 问题：输入和输出长度不同（如翻译：3个英文词→5个中文词）
   - 编码器（Encoder）：将输入序列压缩为一个上下文向量
   - 解码器（Decoder）：从上下文向量生成输出序列
   - 瓶颈问题：所有信息压缩到一个固定长度的向量，信息丢失

2. 注意力机制的动机
   - 瓶颈问题：长句子无法被一个向量完整表示
   - 解决方案：解码器每一步都能"看到"编码器的所有隐状态
   - 类比：翻译时不是先读完整篇文章再翻，而是翻到每个词时回头看对应的原文

3. Bahdanau 注意力（Additive Attention）
   - score(s_t, h_i) = v^T · tanh(W_s·s_t + W_h·h_i)
   - α_i = softmax(score_i)  （注意力权重）
   - context = Σ α_i · h_i   （加权求和）
   - 注意力权重 α 可以可视化为"对齐矩阵"

4. 从 Bahdanau 到 Transformer
   - Bahdanau：score = v^T·tanh(W_s·s + W_h·h)（加法注意力）
   - Transformer：score = Q·K^T/√d（点积注意力，更快）
   - Bahdanau 的 h_i → Transformer 的 Key 和 Value
   - Bahdanau 的 s_t → Transformer 的 Query
   - 本质上是同一个思想的不同实现！

5. Teacher Forcing
   - 训练时：解码器输入用真实标签（不是自己的预测）
   - 推理时：解码器输入用自己上一步的预测
   - 好处：训练更稳定
   - 坏处：训练和推理分布不一致（exposure bias）

【前置知识】
第4章第1-2节 - RNN/LSTM 基础
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════

def softmax(x):
    """Softmax 函数，对最后一维做归一化，防溢出"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def tanh(x):
    """tanh 激活函数"""
    return np.tanh(x)


def one_hot(indices, depth):
    """将整数索引转为 one-hot 编码，indices 形状任意，返回多一维"""
    oh = np.zeros((*np.array(indices).shape, depth))
    idx = np.array(indices)
    if idx.ndim == 0:
        oh[int(idx)] = 1.0
    elif idx.ndim == 1:
        oh[np.arange(len(idx)), idx] = 1.0
    return oh


# ════════════════════════════════════════════════════════════════════
# 第1部分：Seq2Seq 无注意力 —— 编码器-解码器与信息瓶颈
# ════════════════════════════════════════════════════════════════════
#
# 经典 Seq2Seq 架构（Sutskever et al., 2014）：
#   1. 编码器 RNN 逐个读入源序列，最终隐状态作为"上下文向量"
#   2. 解码器 RNN 从上下文向量出发，逐个生成目标序列
#
# 问题：所有源序列信息被压缩到一个固定维度的向量中
#   - 短序列还行，长序列信息严重丢失
#   - 这就是"信息瓶颈"
#

print("=" * 60)
print("第1部分：Seq2Seq 无注意力 —— 信息瓶颈问题")
print("=" * 60)


class SimpleRNNCell:
    """
    简易 RNN 单元，用于编码器和解码器。

    公式：h_t = tanh(W_ih · x_t + W_hh · h_{t-1} + b)

    参数:
        input_dim  : 输入维度
        hidden_dim : 隐状态维度
    """
    def __init__(self, input_dim, hidden_dim):
        scale = 0.1
        self.W_ih = np.random.randn(input_dim, hidden_dim) * scale
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * scale
        self.b = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev):
        """
        前向传播一步。

        参数:
            x      : 输入向量，形状 (input_dim,)
            h_prev : 上一步隐状态，形状 (hidden_dim,)

        返回:
            h_next : 新的隐状态，形状 (hidden_dim,)
        """
        h_next = tanh(x @ self.W_ih + h_prev @ self.W_hh + self.b)
        return h_next


class Seq2SeqNoAttention:
    """
    无注意力的 Seq2Seq 模型。

    编码器读完整个源序列后，将最终隐状态作为唯一的上下文向量传给解码器。
    解码器从这个固定向量出发，自回归地生成目标序列。

    这就是信息瓶颈：不管源序列多长，都被压缩到一个 hidden_dim 维向量中。

    参数:
        vocab_size  : 词表大小（源和目标共享同一个词表）
        hidden_dim  : RNN 隐状态维度
    """
    def __init__(self, vocab_size, hidden_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # 编码器和解码器各自拥有独立的 RNN 参数
        self.encoder_cell = SimpleRNNCell(vocab_size, hidden_dim)
        self.decoder_cell = SimpleRNNCell(vocab_size, hidden_dim)
        # 解码器输出层：将隐状态映射为词表上的 logits
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.1
        self.b_out = np.zeros(vocab_size)

    def encode(self, src_seq):
        """
        编码器：逐步读入源序列，返回最终隐状态。

        参数:
            src_seq : 源序列，整数列表，如 [3, 1, 4]

        返回:
            context   : 最终隐状态（上下文向量），形状 (hidden_dim,)
            all_hiddens: 所有时间步的隐状态列表（用于后续对比）
        """
        h = np.zeros(self.hidden_dim)
        all_hiddens = []
        for token in src_seq:
            x = one_hot(token, self.vocab_size)
            h = self.encoder_cell.forward(x, h)
            all_hiddens.append(h.copy())
        return h, all_hiddens

    def decode_step(self, token, h):
        """
        解码器前向一步：给定上一步输出的 token 和隐状态，生成下一步。

        返回:
            logits : 词表上的分数，形状 (vocab_size,)
            h_next : 更新后的隐状态
        """
        x = one_hot(token, self.vocab_size)
        h_next = self.decoder_cell.forward(x, h)
        logits = h_next @ self.W_out + self.b_out
        return logits, h_next

    def greedy_decode(self, src_seq, max_len, sos_token=0):
        """
        贪心解码：每一步选概率最大的 token。

        参数:
            src_seq   : 源序列
            max_len   : 最大生成长度
            sos_token : 起始符号的索引

        返回:
            output_seq : 生成的 token 序列
        """
        context, _ = self.encode(src_seq)
        h = context
        token = sos_token
        output_seq = []
        for _ in range(max_len):
            logits, h = self.decode_step(token, h)
            token = np.argmax(logits)
            output_seq.append(token)
        return output_seq


# --- 演示信息瓶颈 ---
vocab_size = 10
hidden_dim = 8

model_no_attn = Seq2SeqNoAttention(vocab_size, hidden_dim)

# 测试不同长度的序列，观察上下文向量的信息保留情况
print("\n信息瓶颈演示：所有信息被压缩到一个固定大小的向量中")
print(f"  隐状态维度 = {hidden_dim}（一个 {hidden_dim} 维向量要记住整个序列！）\n")

for length in [3, 5, 10, 20]:
    src = list(range(min(length, vocab_size)))[:length]
    context, all_h = model_no_attn.encode(src)
    # 看上下文向量的范数——长序列的信息被"挤压"
    norm = np.linalg.norm(context)
    print(f"  源序列长度 = {length:2d} | 上下文向量 L2 范数 = {norm:.4f} | "
          f"向量维度 = {hidden_dim}")

print("\n  观察：不管序列多长，上下文向量的维度都是固定的。")
print("  长序列的信息被严重压缩——这就是信息瓶颈！")


# ════════════════════════════════════════════════════════════════════
# 第2部分：Bahdanau 注意力机制实现
# ════════════════════════════════════════════════════════════════════
#
# Bahdanau 注意力（加法注意力，2015）让解码器在每一步都能"回头看"
# 编码器的所有隐状态，而不是只看最后一个。
#
# 计算过程：
#   1. score_i = v^T · tanh(W_s · s_t + W_h · h_i)
#      其中 s_t 是解码器当前状态，h_i 是编码器第 i 步的隐状态
#   2. α = softmax(scores)  — 注意力权重
#   3. context = Σ α_i · h_i — 加权求和得到上下文向量
#
# 直觉：解码器在翻译每个词时，会"关注"源句的不同部分。
#

print("\n" + "=" * 60)
print("第2部分：Bahdanau 注意力机制（加法注意力）")
print("=" * 60)


class BahdanauAttention:
    """
    Bahdanau 加法注意力机制。

    公式：
        score(s, h_i) = v^T · tanh(W_s · s + W_h · h_i)
        α_i = softmax(score_i)
        context = Σ α_i · h_i

    参数:
        hidden_dim   : 编码器/解码器的隐状态维度
        attention_dim: 注意力内部空间的维度
    """
    def __init__(self, hidden_dim, attention_dim):
        scale = 0.1
        # W_s 将解码器状态投影到注意力空间
        self.W_s = np.random.randn(hidden_dim, attention_dim) * scale
        # W_h 将编码器状态投影到注意力空间
        self.W_h = np.random.randn(hidden_dim, attention_dim) * scale
        # v 将注意力空间的向量压缩为一个标量分数
        self.v = np.random.randn(attention_dim) * scale

    def __call__(self, decoder_state, encoder_hiddens):
        """
        计算注意力权重和上下文向量。

        参数:
            decoder_state    : 解码器当前隐状态，形状 (hidden_dim,)
            encoder_hiddens  : 编码器所有隐状态，形状 (src_len, hidden_dim)

        返回:
            context : 上下文向量，形状 (hidden_dim,)
            weights : 注意力权重，形状 (src_len,)
        """
        src_len = encoder_hiddens.shape[0]

        # 步骤 1：将解码器状态投影到注意力空间 (attention_dim,)
        s_proj = decoder_state @ self.W_s  # (attention_dim,)

        # 步骤 2：将编码器每个隐状态投影到注意力空间 (src_len, attention_dim)
        h_proj = encoder_hiddens @ self.W_h  # (src_len, attention_dim)

        # 步骤 3：计算注意力分数
        # s_proj 广播加到每个 h_proj 上，过 tanh，再用 v 压缩为标量
        scores = tanh(s_proj[np.newaxis, :] + h_proj) @ self.v  # (src_len,)

        # 步骤 4：softmax 归一化为概率分布
        weights = softmax(scores)  # (src_len,)

        # 步骤 5：加权求和得到上下文向量
        context = weights @ encoder_hiddens  # (hidden_dim,)

        return context, weights


# --- 演示注意力计算过程 ---
hidden_dim = 8
attention_dim = 6
attn = BahdanauAttention(hidden_dim, attention_dim)

# 模拟编码器输出 5 个隐状态
src_len = 5
encoder_hiddens = np.random.randn(src_len, hidden_dim) * 0.5
decoder_state = np.random.randn(hidden_dim) * 0.5

context, weights = attn(decoder_state, encoder_hiddens)

print(f"\n  编码器隐状态数量: {src_len}")
print(f"  注意力权重: {np.array2string(weights, precision=3)}")
print(f"  权重之和 = {weights.sum():.6f}（应为 1.0）")
print(f"  上下文向量维度: {context.shape}")
print(f"\n  解读：解码器通过注意力权重，对编码器的不同位置赋予不同的关注度。")
print(f"  权重最大的位置 = {np.argmax(weights)}（解码器最关注编码器的第 {np.argmax(weights)} 个位置）")


# ════════════════════════════════════════════════════════════════════
# 第3部分：注意力权重可视化 —— 对齐矩阵
# ════════════════════════════════════════════════════════════════════
#
# 注意力权重可以被可视化为一个"对齐矩阵"（alignment matrix）：
#   - 行 = 解码器每一步（目标序列的每个位置）
#   - 列 = 编码器每一步（源序列的每个位置）
#   - 颜色深浅 = 注意力权重大小
#
# 如果模型学好了，对齐矩阵应该显示出清晰的对齐模式。
# 例如在序列翻转任务中，应该出现反对角线的模式。
#

print("\n" + "=" * 60)
print("第3部分：注意力权重可视化 —— 对齐矩阵")
print("=" * 60)


def visualize_attention(attention_weights, src_tokens, tgt_tokens, title="注意力对齐矩阵"):
    """
    可视化注意力权重矩阵。

    参数:
        attention_weights : 注意力矩阵，形状 (tgt_len, src_len)
        src_tokens        : 源序列标签列表
        tgt_tokens        : 目标序列标签列表
        title             : 图标题
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attention_weights, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="注意力权重")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=11)
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=11)
    ax.set_xlabel("源序列（编码器）", fontsize=12)
    ax.set_ylabel("目标序列（解码器）", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # 在每个格子里标注权重数值
    for i in range(len(tgt_tokens)):
        for j in range(len(src_tokens)):
            val = attention_weights[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.tight_layout()
    return fig


# --- 演示1：模拟"理想"的注意力对齐 ---
# 假设任务是翻转序列：[A, B, C, D] → [D, C, B, A]
# 理想的注意力应该是反对角线模式

src_tokens_demo = ["A", "B", "C", "D"]
tgt_tokens_demo = ["D", "C", "B", "A"]

# 构造理想的反对角线注意力矩阵
ideal_attn = np.zeros((4, 4))
for i in range(4):
    ideal_attn[i, 3 - i] = 1.0

# 加一点噪声让它更真实
noisy_attn = ideal_attn * 0.75 + np.random.rand(4, 4) * 0.05
noisy_attn = noisy_attn / noisy_attn.sum(axis=1, keepdims=True)

fig = visualize_attention(noisy_attn, src_tokens_demo, tgt_tokens_demo,
                          "理想的注意力对齐（序列翻转任务）")
plt.savefig("04_03_attention_alignment_ideal.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_03_attention_alignment_ideal.png")

print("\n  解读：在序列翻转任务中，理想的注意力是反对角线模式。")
print("  解码第 1 个输出 D 时，应该关注源序列的最后一个位置 D。")


# ════════════════════════════════════════════════════════════════════
# 第4部分：Seq2Seq + Attention 完整模型
# ════════════════════════════════════════════════════════════════════
#
# 将注意力机制集成到 Seq2Seq 中：
#   1. 编码器不变：逐步读入源序列，收集所有隐状态
#   2. 解码器每一步：
#      a. 用当前隐状态和编码器所有隐状态计算注意力
#      b. 得到上下文向量
#      c. 将上下文向量拼接到解码器输入中
#      d. 更新隐状态，生成输出
#
# 对比无注意力版本：
#   - 无注意力：只看编码器最终状态（信息瓶颈）
#   - 有注意力：每步都能看编码器所有状态（无瓶颈）
#

print("\n" + "=" * 60)
print("第4部分：Seq2Seq + Attention 完整模型")
print("=" * 60)


class Seq2SeqWithAttention:
    """
    带 Bahdanau 注意力的 Seq2Seq 模型。

    与无注意力版本的核心区别：
    解码器每一步都通过注意力机制动态计算上下文向量，
    而不是固定使用编码器的最终隐状态。

    参数:
        vocab_size     : 词表大小
        hidden_dim     : 隐状态维度
        attention_dim  : 注意力内部维度
    """
    def __init__(self, vocab_size, hidden_dim, attention_dim):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        # 编码器 RNN
        self.encoder_cell = SimpleRNNCell(vocab_size, hidden_dim)
        # 解码器 RNN：输入 = token embedding + 上下文向量
        self.decoder_cell = SimpleRNNCell(vocab_size + hidden_dim, hidden_dim)
        # 注意力模块
        self.attention = BahdanauAttention(hidden_dim, attention_dim)
        # 输出层
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.1
        self.b_out = np.zeros(vocab_size)

    def encode(self, src_seq):
        """编码器：收集所有时间步的隐状态"""
        h = np.zeros(self.hidden_dim)
        all_hiddens = []
        for token in src_seq:
            x = one_hot(token, self.vocab_size)
            h = self.encoder_cell.forward(x, h)
            all_hiddens.append(h.copy())
        # all_hiddens: (src_len, hidden_dim)
        return np.array(all_hiddens), h

    def decode_step(self, token, h, encoder_hiddens):
        """
        带注意力的解码器前向一步。

        参数:
            token            : 上一步输出的 token 索引
            h                : 解码器当前隐状态
            encoder_hiddens  : 编码器所有隐状态 (src_len, hidden_dim)

        返回:
            logits  : 词表上的分数
            h_next  : 更新后的隐状态
            attn_w  : 注意力权重
        """
        # 步骤 1：计算注意力上下文
        context, attn_w = self.attention(h, encoder_hiddens)

        # 步骤 2：拼接 token embedding 和上下文向量作为解码器输入
        x_embed = one_hot(token, self.vocab_size)
        decoder_input = np.concatenate([x_embed, context])  # (vocab_size + hidden_dim,)

        # 步骤 3：RNN 前向一步
        h_next = self.decoder_cell.forward(decoder_input, h)

        # 步骤 4：生成输出 logits
        logits = h_next @ self.W_out + self.b_out

        return logits, h_next, attn_w

    def forward(self, src_seq, tgt_seq, teacher_forcing=True):
        """
        完整前向传播（训练模式）。

        参数:
            src_seq         : 源序列（整数列表）
            tgt_seq         : 目标序列（整数列表）
            teacher_forcing : 是否使用 Teacher Forcing

        返回:
            all_logits      : 每步的 logits 列表
            all_attn_weights: 每步的注意力权重列表
        """
        # 编码
        encoder_hiddens, enc_final = self.encode(src_seq)

        # 解码
        h = enc_final  # 用编码器最终状态初始化解码器
        all_logits = []
        all_attn_weights = []
        token = 0  # 起始符号 <SOS>

        for t in range(len(tgt_seq)):
            logits, h, attn_w = self.decode_step(token, h, encoder_hiddens)
            all_logits.append(logits)
            all_attn_weights.append(attn_w)

            if teacher_forcing:
                # Teacher Forcing：用真实标签作为下一步输入
                token = tgt_seq[t]
            else:
                # 自回归：用自己的预测作为下一步输入
                token = np.argmax(logits)

        return all_logits, all_attn_weights

    def greedy_decode(self, src_seq, max_len, sos_token=0):
        """贪心解码（推理模式）"""
        encoder_hiddens, enc_final = self.encode(src_seq)
        h = enc_final
        token = sos_token
        output_seq = []
        all_attn_weights = []

        for _ in range(max_len):
            logits, h, attn_w = self.decode_step(token, h, encoder_hiddens)
            token = np.argmax(logits)
            output_seq.append(token)
            all_attn_weights.append(attn_w)

        return output_seq, np.array(all_attn_weights)


# --- 对比有无注意力模型的结构差异 ---
vocab_size = 10
hidden_dim = 16
attention_dim = 12

model_with_attn = Seq2SeqWithAttention(vocab_size, hidden_dim, attention_dim)

print(f"\n  模型参数配置：")
print(f"    词表大小      = {vocab_size}")
print(f"    隐状态维度    = {hidden_dim}")
print(f"    注意力内部维度 = {attention_dim}")

print(f"\n  无注意力：解码器输入 = token embedding ({vocab_size}维)")
print(f"  有注意力：解码器输入 = token embedding + 上下文向量 ({vocab_size}+{hidden_dim}={vocab_size+hidden_dim}维)")
print(f"  注意力让解码器每一步都能访问编码器所有位置的信息！")


# ════════════════════════════════════════════════════════════════════
# 第5部分：从注意力到 QKV —— Transformer 的前兆
# ════════════════════════════════════════════════════════════════════
#
# Bahdanau 注意力和 Transformer 的注意力，本质上是同一个思想！
# 只是计算分数的方式不同：
#
#   Bahdanau（加法注意力）：
#     score = v^T · tanh(W_s · s + W_h · h)
#     - s 就是 Query（"我想要什么？"）
#     - h 就是 Key 和 Value（"我有什么？"）
#
#   Transformer（缩放点积注意力）：
#     score = Q · K^T / √d_k
#     context = softmax(score) · V
#     - Q、K、V 是通过不同的线性变换得到的
#
# 关键对应关系：
#   Bahdanau s_t     ←→  Transformer Query
#   Bahdanau h_i     ←→  Transformer Key（用于计算相似度）
#   Bahdanau h_i     ←→  Transformer Value（用于加权求和）
#   Bahdanau W_s, W_h ←→  Transformer W_Q, W_K
#
# Transformer 把 Key 和 Value 分开了——这是更灵活的设计！
#

print("\n" + "=" * 60)
print("第5部分：从注意力到 QKV —— 通向 Transformer")
print("=" * 60)


def attention_bahdanau_style(query, keys, values, W_q, W_k, v):
    """
    Bahdanau 风格的加法注意力（显式展示 QKV 对应关系）。

    参数:
        query  : 解码器状态 s_t，形状 (d,)    ← 这就是 Q
        keys   : 编码器隐状态，形状 (n, d)    ← 这就是 K
        values : 编码器隐状态，形状 (n, d)    ← 这就是 V（与 K 相同）
        W_q    : Query 的投影矩阵 (d, attn_d) ← 对应 Transformer 的 W_Q
        W_k    : Key 的投影矩阵 (d, attn_d)   ← 对应 Transformer 的 W_K
        v      : 分数向量 (attn_d,)
    """
    # 投影到注意力空间
    q_proj = query @ W_q               # (attn_d,)
    k_proj = keys @ W_k                # (n, attn_d)
    # 计算分数：加法 + tanh
    scores = tanh(q_proj + k_proj) @ v  # (n,)
    # 注意力权重
    weights = softmax(scores)           # (n,)
    # 加权求和 Value
    context = weights @ values          # (d,)
    return context, weights


def attention_transformer_style(query, keys, values, W_q, W_k, W_v):
    """
    Transformer 风格的缩放点积注意力（显式展示与 Bahdanau 的对应）。

    参数:
        query  : 形状 (d,)     ← 来源同 Bahdanau 的解码器状态
        keys   : 形状 (n, d)   ← 来源同 Bahdanau 的编码器隐状态
        values : 形状 (n, d)   ← Transformer 中 V 可以不等于 K！
        W_q    : Query 投影 (d, d_k)
        W_k    : Key 投影 (d, d_k)
        W_v    : Value 投影 (d, d_v)   ← Bahdanau 没有这个！
    """
    # 线性投影
    Q = query @ W_q                     # (d_k,)
    K = keys @ W_k                      # (n, d_k)
    V = values @ W_v                    # (n, d_v)

    d_k = Q.shape[-1]
    # 缩放点积分数
    scores = K @ Q / np.sqrt(d_k)       # (n,)
    # 注意力权重
    weights = softmax(scores)           # (n,)
    # 加权求和
    context = weights @ V               # (d_v,)
    return context, weights


# --- 对比演示 ---
d = 8      # 隐状态维度
n = 5      # 源序列长度
d_k = 6    # Key/Query 投影维度
d_v = 8    # Value 投影维度

np.random.seed(42)
query = np.random.randn(d) * 0.3
keys = np.random.randn(n, d) * 0.3
values = keys.copy()  # Bahdanau 中 K == V

# Bahdanau 参数
W_q_bah = np.random.randn(d, d_k) * 0.1
W_k_bah = np.random.randn(d, d_k) * 0.1
v_bah = np.random.randn(d_k) * 0.1

# Transformer 参数
W_q_tfm = np.random.randn(d, d_k) * 0.1
W_k_tfm = np.random.randn(d, d_k) * 0.1
W_v_tfm = np.random.randn(d, d_v) * 0.1

ctx_bah, w_bah = attention_bahdanau_style(query, keys, values, W_q_bah, W_k_bah, v_bah)
ctx_tfm, w_tfm = attention_transformer_style(query, keys, values, W_q_tfm, W_k_tfm, W_v_tfm)

print(f"\n  Bahdanau 注意力权重:    {np.array2string(w_bah, precision=3)}")
print(f"  Transformer 注意力权重: {np.array2string(w_tfm, precision=3)}")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │         Bahdanau 注意力  ←对应→  Transformer 注意力     │
  ├─────────────────────────────────────────────────────────┤
  │  解码器状态 s_t          ←→    Query (Q)               │
  │  编码器隐状态 h_i        ←→    Key (K)                 │
  │  编码器隐状态 h_i        ←→    Value (V)               │
  │  W_s                     ←→    W_Q                     │
  │  W_h                     ←→    W_K                     │
  │  （没有）                ←→    W_V（独立变换 V）       │
  │  v^T·tanh(Ws·s + Wh·h)  ←→    Q·K^T / sqrt(d_k)      │
  │  加法注意力              ←→    点积注意力（更快）      │
  ├─────────────────────────────────────────────────────────┤
  │  本质上是同一个思想：                                   │
  │  "Query 问编码器每个位置：你和我有多相关？"             │
  │  "根据相关度加权汇总 Value，得到上下文。"               │
  └─────────────────────────────────────────────────────────┘
""")


# ════════════════════════════════════════════════════════════════════
# 第6部分：Teacher Forcing 机制
# ════════════════════════════════════════════════════════════════════
#
# Teacher Forcing 是训练 Seq2Seq（以及所有自回归模型）的核心技巧。
#
# 核心区别：
#   训练时（Teacher Forcing）：解码器每一步的输入 = 真实标签
#   推理时（自回归）        ：解码器每一步的输入 = 上一步的预测
#
# 为什么需要 Teacher Forcing？
#   - 如果训练时也用自己的预测，一步错步步错（错误传播）
#   - Teacher Forcing 让训练信号更稳定，收敛更快
#
# Teacher Forcing 的问题：Exposure Bias
#   - 训练时模型从未见过"自己犯错后该怎么办"
#   - 推理时一旦犯错，后续就完全偏离
#   - 缓解方案：Scheduled Sampling（随训练逐步减少 Teacher Forcing 比例）
#

print("=" * 60)
print("第6部分：Teacher Forcing 机制")
print("=" * 60)

print("""
  Teacher Forcing 图解：

  【训练时 — Teacher Forcing】
  解码器:  <SOS> → [RNN] → 输出1    真实标签1 → [RNN] → 输出2    真实标签2 → [RNN] → 输出3
                                          ↑                             ↑
                                   用真实标签                     用真实标签
                                   (不用预测)                     (不用预测)

  【推理时 — 自回归】
  解码器:  <SOS> → [RNN] → 预测1 → [RNN] → 预测2 → [RNN] → 预测3
                               ↑                 ↑
                          用自己的预测        用自己的预测
                          (可能是错的)        (错误累积)
""")


def demonstrate_teacher_forcing():
    """演示 Teacher Forcing 和自回归解码的区别"""
    vocab_size_demo = 6
    hidden_dim_demo = 12
    attention_dim_demo = 8

    model = Seq2SeqWithAttention(vocab_size_demo, hidden_dim_demo, attention_dim_demo)

    src = [1, 2, 3, 4]
    tgt = [4, 3, 2, 1]  # 翻转任务

    # 使用 Teacher Forcing
    logits_tf, attn_tf = model.forward(src, tgt, teacher_forcing=True)
    preds_tf = [np.argmax(l) for l in logits_tf]

    # 不使用 Teacher Forcing
    logits_no_tf, attn_no_tf = model.forward(src, tgt, teacher_forcing=False)
    preds_no_tf = [np.argmax(l) for l in logits_no_tf]

    print(f"  源序列      : {src}")
    print(f"  目标序列    : {tgt}")
    print(f"  Teacher Forcing 预测: {preds_tf}")
    print(f"  自回归 预测 : {preds_no_tf}")
    print(f"\n  注意：随机初始化的模型预测都不对，但训练时 Teacher Forcing 会帮助模型更快学习。")
    print(f"  因为即使上一步预测错了，下一步仍然使用正确的输入。")

    return model


model_demo = demonstrate_teacher_forcing()

# --- Scheduled Sampling 比率可视化 ---
epochs = np.arange(100)
# 线性衰减：从 1.0（完全 Teacher Forcing）降到 0.0（完全自回归）
tf_ratio_linear = np.maximum(0, 1.0 - epochs / 80)
# 指数衰减
tf_ratio_exp = 0.99 ** epochs
# 逆sigmoid衰减
k = 20
tf_ratio_inv_sig = k / (k + np.exp(epochs / k))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, tf_ratio_linear, "-", linewidth=2, label="线性衰减")
ax.plot(epochs, tf_ratio_exp, "--", linewidth=2, label="指数衰减")
ax.plot(epochs, tf_ratio_inv_sig, "-.", linewidth=2, label="逆 Sigmoid 衰减")
ax.set_xlabel("训练轮数 (Epoch)", fontsize=12)
ax.set_ylabel("Teacher Forcing 比例", fontsize=12)
ax.set_title("Scheduled Sampling：逐步减少 Teacher Forcing", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("04_03_scheduled_sampling.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_03_scheduled_sampling.png")


# ════════════════════════════════════════════════════════════════════
# 第7部分：数字序列翻译任务 —— 训练并观察注意力学习对齐
# ════════════════════════════════════════════════════════════════════
#
# 任务：序列翻转
#   输入 [1, 2, 3, 4] → 输出 [4, 3, 2, 1]
#
# 这是一个简单但很好的测试任务：
#   - 如果模型学对了，注意力矩阵应显示反对角线模式
#   - 可以清楚看到有/无注意力的性能差距
#
# 我们用简化的训练流程（梯度用数值差分近似），重点展示概念。
#

print("\n" + "=" * 60)
print("第7部分：数字序列翻译任务 —— 训练注意力模型")
print("=" * 60)


def generate_reverse_data(n_samples, seq_len, vocab_size):
    """
    生成序列翻转数据集。

    参数:
        n_samples  : 样本数量
        seq_len    : 序列长度
        vocab_size : 词表大小（token 取值范围 1 ~ vocab_size-1，0 留给 <SOS>）

    返回:
        src_data : 源序列列表
        tgt_data : 目标序列列表（翻转后的源序列）
    """
    src_data = []
    tgt_data = []
    for _ in range(n_samples):
        seq = np.random.randint(1, vocab_size, size=seq_len).tolist()
        src_data.append(seq)
        tgt_data.append(seq[::-1])  # 翻转
    return src_data, tgt_data


def cross_entropy_loss(logits_list, targets):
    """
    计算交叉熵损失。

    参数:
        logits_list : logits 列表，每个形状 (vocab_size,)
        targets     : 目标 token 列表

    返回:
        loss : 平均交叉熵损失
    """
    total_loss = 0.0
    for logits, target in zip(logits_list, targets):
        probs = softmax(logits)
        # 防止 log(0)
        total_loss += -np.log(probs[target] + 1e-10)
    return total_loss / len(targets)


def compute_accuracy(model, src_data, tgt_data):
    """计算序列级别的准确率（整个序列都正确才算对）"""
    correct = 0
    for src, tgt in zip(src_data, tgt_data):
        pred, _ = model.greedy_decode(src, len(tgt), sos_token=0)
        if pred == tgt:
            correct += 1
    return correct / len(src_data)


def numerical_gradient_update(model, src, tgt, lr=0.001, eps=1e-4):
    """
    用数值梯度更新模型参数（简化版训练，用于演示）。

    对模型每个参数矩阵的每个元素：
      1. 加 eps 计算 loss+
      2. 减 eps 计算 loss-
      3. 梯度 ≈ (loss+ - loss-) / (2*eps)
      4. 参数 -= lr * 梯度

    注意：这种方式非常慢，实际中需要用反向传播。
    这里为了教学简洁性使用数值方法。
    """
    # 收集所有可训练的参数矩阵
    param_refs = [
        model.encoder_cell.W_ih, model.encoder_cell.W_hh, model.encoder_cell.b,
        model.decoder_cell.W_ih, model.decoder_cell.W_hh, model.decoder_cell.b,
        model.attention.W_s, model.attention.W_h, model.attention.v,
        model.W_out, model.b_out,
    ]

    # 当前损失
    logits, _ = model.forward(src, tgt, teacher_forcing=True)
    base_loss = cross_entropy_loss(logits, tgt)

    # 对每个参数做数值梯度更新（只随机抽样部分元素以加速）
    for param in param_refs:
        flat = param.ravel()
        # 随机选择一小部分参数更新（加速训练）
        n_update = max(1, len(flat) // 4)
        indices = np.random.choice(len(flat), size=n_update, replace=False)

        for idx in indices:
            old_val = flat[idx]

            flat[idx] = old_val + eps
            logits_p, _ = model.forward(src, tgt, teacher_forcing=True)
            loss_p = cross_entropy_loss(logits_p, tgt)

            flat[idx] = old_val - eps
            logits_m, _ = model.forward(src, tgt, teacher_forcing=True)
            loss_m = cross_entropy_loss(logits_m, tgt)

            grad = (loss_p - loss_m) / (2 * eps)
            flat[idx] = old_val - lr * grad

    return base_loss


# --- 训练模型 ---
vocab_size = 6       # 词表：0=<SOS>, 1-5=数字
hidden_dim = 16
attention_dim = 12
seq_len = 4

np.random.seed(42)
model_attn = Seq2SeqWithAttention(vocab_size, hidden_dim, attention_dim)

# 生成训练数据
n_train = 30
src_train, tgt_train = generate_reverse_data(n_train, seq_len, vocab_size)

print(f"\n  任务：序列翻转")
print(f"  词表大小 = {vocab_size}, 序列长度 = {seq_len}")
print(f"  训练样本示例：{src_train[0]} → {tgt_train[0]}")
print(f"\n  开始训练（数值梯度，请耐心等待）...\n")

n_epochs = 60
losses = []
for epoch in range(n_epochs):
    epoch_loss = 0.0
    # 每个 epoch 随机选一批样本训练
    indices = np.random.permutation(n_train)[:8]
    for i in indices:
        loss = numerical_gradient_update(
            model_attn, src_train[i], tgt_train[i], lr=0.01, eps=1e-4
        )
        epoch_loss += loss
    avg_loss = epoch_loss / len(indices)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        acc = compute_accuracy(model_attn, src_train[:10], tgt_train[:10])
        print(f"  Epoch {epoch+1:3d} | 损失 = {avg_loss:.4f} | 训练准确率 = {acc:.1%}")

# --- 可视化训练过程 ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, "b-", linewidth=1.5)
ax.set_xlabel("训练轮数 (Epoch)", fontsize=12)
ax.set_ylabel("交叉熵损失", fontsize=12)
ax.set_title("Seq2Seq + Attention 训练损失曲线（序列翻转任务）", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_03_training_loss.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_03_training_loss.png")

# --- 可视化注意力权重 ---
# 用一个训练样本测试模型，观察注意力对齐
test_src = src_train[0]
test_tgt = tgt_train[0]
pred, attn_weights = model_attn.greedy_decode(test_src, len(test_tgt), sos_token=0)

print(f"\n  测试样本：")
print(f"    源序列  : {test_src}")
print(f"    目标序列: {test_tgt}")
print(f"    模型预测: {pred}")

# 可视化注意力矩阵
src_labels = [str(t) for t in test_src]
tgt_labels = [str(t) for t in pred]

fig = visualize_attention(
    attn_weights,
    src_labels,
    tgt_labels,
    title=f"训练后的注意力对齐\n源: {test_src} → 目标: {test_tgt}"
)
plt.savefig("04_03_learned_attention.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_03_learned_attention.png")

print(f"\n  解读注意力矩阵：")
print(f"  - 如果模型学对了，翻转任务的注意力应该呈反对角线分布")
print(f"  - 解码第 1 个输出时，应该关注源序列的最后一个位置")
print(f"  - 解码第 2 个输出时，应该关注源序列的倒数第二个位置")
print(f"  - 这就是注意力机制学到的"对齐"（alignment）！")


# ════════════════════════════════════════════════════════════════════
# 第8部分：完整对比总结 + 思考题
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("完整总结")
print("=" * 60)
print("""
本节展示了从 Seq2Seq 到 Transformer 的关键演进路线：

  1. Seq2Seq 无注意力
     - 编码器将整个源序列压缩为一个固定向量
     - 信息瓶颈：长序列信息严重丢失
     - 解码器"盲人摸象"，只能凭最后一刻的记忆翻译

  2. Bahdanau 注意力
     - 解码器每一步都能"回头看"编码器的所有位置
     - 注意力权重 = 动态的、内容相关的权重分配
     - 消除了信息瓶颈

  3. 从注意力到 QKV
     - Bahdanau: s_t → Query, h_i → Key=Value
     - Transformer: 独立的 Q, K, V 投影
     - 点积注意力比加法注意力更快（矩阵乘法优化）

  4. Teacher Forcing
     - 训练时用真实标签引导，推理时用自己的预测
     - Scheduled Sampling 缓解 Exposure Bias

  5. 通向 Transformer 的桥梁
     - Seq2Seq 的编码器-解码器 → Transformer 的编码器-解码器
     - Cross-Attention（本节所学）→ Transformer 的核心组件之一
     - 只需再加上 Self-Attention 和位置编码，就是完整的 Transformer！

  ┌────────────────────────────────────────────────────┐
  │  演进路线：                                        │
  │  RNN → Seq2Seq → Attention → Transformer → GPT   │
  │                      ↑                             │
  │                  你在这里！                        │
  └────────────────────────────────────────────────────┘
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【信息瓶颈的量化】
   如果编码器隐状态维度是 256，源序列有 100 个 token，
   无注意力的 Seq2Seq 需要用 256 个浮点数存储 100 个 token 的信息。
   而有注意力的模型实际上可以访问 100×256 = 25600 个浮点数的信息。
   这个 100 倍的信息差距如何影响翻译质量？
   提示：想想翻译一段 100 个词的话，和只看一眼就翻译的区别。

2. 【注意力权重的含义】
   在翻译"I love machine learning"→"我喜欢机器学习"时，
   解码器生成"机器"这个词时，注意力权重应该集中在源句的哪个位置？
   如果注意力分散在所有位置（均匀分布），模型可能有什么问题？
   提示：均匀注意力等价于取平均，信息被"稀释"了。

3. 【加法注意力 vs 点积注意力】
   Bahdanau 的加法注意力需要一个额外的 v 向量和 tanh 运算，
   而 Transformer 的点积注意力只需要矩阵乘法。
   在 GPU 上，为什么点积注意力更快？
   Transformer 的 √d_k 缩放因子起什么作用？如果不缩放会怎样？
   提示：d_k 很大时，点积的值也会很大，导致 softmax 输出趋近
   one-hot（梯度消失）。

4. 【Teacher Forcing 的两难困境】
   如果完全不用 Teacher Forcing（训练时也用模型自己的预测），
   训练会很不稳定。但如果一直用 Teacher Forcing，推理时性能可能很差。
   Scheduled Sampling 是如何解决这个矛盾的？
   你能想到其他解决方案吗？
   提示：想想 GAN 的思想——用一个判别器区分"训练分布"和"推理分布"。

5. 【从这里到 Transformer 还差什么？】
   我们已经有了：编码器-解码器、Cross-Attention。
   Transformer 还额外引入了哪些关键组件？
   a) Self-Attention 和 Cross-Attention 有什么区别？
   b) 为什么 Transformer 不用 RNN？它用什么替代了 RNN 的序列建模能力？
   c) 多头注意力（Multi-Head Attention）解决了什么问题？
   提示：Self-Attention 让序列内部的 token 也互相"看"到彼此；
   位置编码替代了 RNN 天然的位置感知能力。
""")

print("下一节预告: 第5章 · PyTorch 基础 / 第6章 · 注意力机制深入")
