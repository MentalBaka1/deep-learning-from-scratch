"""
====================================================================
第6章 · 第4节 · 注意力变体：自注意力、交叉注意力、因果注意力、MQA/GQA
====================================================================

【一句话总结】
注意力有多种变体：自注意力让序列"审视自己"，交叉注意力连接两个序列，
因果注意力防止"偷看未来"，MQA/GQA 在效率和效果间取得平衡。

【为什么深度学习需要这个？】
- 自注意力 = Transformer Encoder 的核心（BERT）
- 因果注意力 = Transformer Decoder 的核心（GPT）
- 交叉注意力 = Encoder-Decoder 连接方式（翻译、多模态）
- MQA/GQA = 现代大模型的效率优化（Qwen、LLaMA 2、DeepSeek 都用 GQA）

【核心概念】

1. 自注意力（Self-Attention）
   - Q, K, V 都来自同一个序列
   - 序列中的每个位置都能"看到"其他所有位置
   - "The cat sat on the mat" → "cat" 知道和 "sat"、"mat" 的关系
   - 用在：Transformer Encoder（BERT）

2. 交叉注意力（Cross-Attention）
   - Q 来自一个序列（如解码器），K 和 V 来自另一个序列（如编码器）
   - 相当于第4章 Seq2Seq 的 Bahdanau 注意力
   - 用在：Transformer Decoder 的第二个注意力层
   - 多模态中：文本 Query + 图像 Key/Value

3. 因果注意力 / 掩码自注意力（Causal / Masked Self-Attention）
   - 位置 i 只能关注位置 ≤ i（不能看到未来）
   - 实现：在注意力分数矩阵上加一个上三角 -∞ 掩码
   - softmax(-∞) = 0，即未来位置的权重为 0
   - 用在：GPT、LLaMA 等自回归语言模型
   - 这是 GPT 能生成文本的关键！

4. MHA vs MQA vs GQA
   - MHA（Multi-Head Attention）：每个头有独立的 Q, K, V
   - MQA（Multi-Query Attention）：所有头共享同一组 K, V
     * 好处：KV Cache 减少 h 倍，推理更快
     * 坏处：质量有些下降
   - GQA（Grouped-Query Attention）：K, V 分成 g 组，每组共享
     * MHA (g=h) 和 MQA (g=1) 的折中
     * LLaMA 2/3, Qwen, DeepSeek 都用 GQA
     * 例：8头注意力，4组 KV → 每2个Q头共享1组KV

5. KV Cache（预告）
   - 自回归生成时，每步只新增1个token的Q
   - 之前所有token的K, V可以缓存复用
   - MQA/GQA的KV更小 → 缓存更省内存 → 能处理更长上下文

【前置知识】
第6章第1-3节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

torch.manual_seed(42)


# ════════════════════════════════════════════════════════════════════
# 第一部分：自注意力（Self-Attention）
# ════════════════════════════════════════════════════════════════════
# 自注意力的核心：Q, K, V 都来自同一个输入序列。
# 每个位置同时扮演三个角色：
#   - 作为 Query：「我想找什么信息？」
#   - 作为 Key：  「我有什么信息可以被找到？」
#   - 作为 Value：「如果被选中，我提供什么内容？」
#
# 例如句子 "The cat sat on the mat"：
#   "cat" 作为 Query 会去找 "sat"（谁做了什么）和 "the"（我的修饰词）
#   "cat" 作为 Key 会被 "sat" 的 Query 找到（坐的主语是猫）
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第一部分：自注意力（Self-Attention）")
print("=" * 60)


class SelfAttention(nn.Module):
    """
    自注意力模块。

    Q, K, V 全部由同一个输入 x 线性变换得到。
    公式：Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    参数:
        d_model : 输入/输出维度
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 三个线性投影：把输入分别映射成 Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        参数:
            x : (batch, seq_len, d_model) — 输入序列

        返回:
            output  : (batch, seq_len, d_model) — 注意力输出
            weights : (batch, seq_len, seq_len) — 注意力权重矩阵
        """
        # Q, K, V 都来自同一个 x —— 这就是"自"注意力的含义
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)

        # 注意力分数 = Q K^T / sqrt(d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores 形状: (batch, seq_len, seq_len)
        # scores[i][j] = 位置 i 对位置 j 的关注程度

        # softmax 归一化 → 注意力权重
        weights = F.softmax(scores, dim=-1)

        # 加权求和 Value
        output = torch.matmul(weights, V)

        return output, weights


# --- 演示：一个句子对自己做注意力 ---
# 模拟 6 个 token，每个 token 用 8 维向量表示
seq_len, d_model = 6, 8
tokens = ["The", "cat", "sat", "on", "the", "mat"]

x = torch.randn(1, seq_len, d_model)  # (1, 6, 8)
self_attn = SelfAttention(d_model)

with torch.no_grad():
    output, weights = self_attn(x)

print(f"\n输入形状:  {x.shape}     ← (batch=1, seq_len=6, d_model=8)")
print(f"输出形状:  {output.shape}  ← 和输入完全相同")
print(f"权重形状:  {weights.shape} ← (batch=1, 6, 6)")

# 展示注意力权重矩阵
print(f"\n注意力权重矩阵（每一行之和 = 1.0）：")
print(f"{'':8s}", end="")
for t in tokens:
    print(f"{t:>7s}", end="")
print()
w = weights[0].numpy()
for i, token in enumerate(tokens):
    print(f"{token:8s}", end="")
    for j in range(seq_len):
        print(f"{w[i][j]:7.3f}", end="")
    print(f"  | 和={w[i].sum():.3f}")

print("\n关键观察：")
print("  - 每一行代表一个 token 对所有 token（包括自己）的关注分布")
print("  - 每一行之和 = 1（softmax 的性质）")
print("  - 自注意力允许每个位置看到所有其他位置（包括自己）")


# ════════════════════════════════════════════════════════════════════
# 第二部分：交叉注意力（Cross-Attention）
# ════════════════════════════════════════════════════════════════════
# 交叉注意力的核心：Q 来自序列 A，K 和 V 来自序列 B。
# 这让序列 A 能"查询"序列 B 的信息。
#
# 典型应用：
#   - 机器翻译：解码器（生成目标语言）查询编码器（理解源语言）
#   - 多模态：文本 Query 查询图像 Key/Value
#   - Transformer Decoder 的第二个注意力层
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第二部分：交叉注意力（Cross-Attention）")
print("=" * 60)


class CrossAttention(nn.Module):
    """
    交叉注意力模块。

    Q 来自 decoder 序列，K 和 V 来自 encoder 序列。
    让 decoder 的每个位置都能"查询" encoder 的所有位置。

    参数:
        d_model : 输入/输出维度
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Q 投影（用于 decoder）
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # K 投影（用于 encoder）
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # V 投影（用于 encoder）

    def forward(self, x_decoder, x_encoder):
        """
        参数:
            x_decoder : (batch, tgt_len, d_model) — 解码器输入（提供 Q）
            x_encoder : (batch, src_len, d_model) — 编码器输出（提供 K, V）

        返回:
            output  : (batch, tgt_len, d_model) — 注意力输出
            weights : (batch, tgt_len, src_len) — 注意力权重
        """
        # Q 来自 decoder，K 和 V 来自 encoder —— 这是"交叉"的含义
        Q = self.W_q(x_decoder)  # (batch, tgt_len, d_model)
        K = self.W_k(x_encoder)  # (batch, src_len, d_model)
        V = self.W_v(x_encoder)  # (batch, src_len, d_model)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores 形状: (batch, tgt_len, src_len)

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        return output, weights


# --- 演示：中翻英场景 ---
src_tokens = ["我", "喜欢", "猫"]        # 编码器输入（源语言）
tgt_tokens = ["I", "like", "cats", "<eos>"]  # 解码器输入（目标语言）

src_len, tgt_len, d_model = 3, 4, 8
x_encoder = torch.randn(1, src_len, d_model)  # 编码器输出
x_decoder = torch.randn(1, tgt_len, d_model)  # 解码器输入

cross_attn = CrossAttention(d_model)

with torch.no_grad():
    output, weights = cross_attn(x_decoder, x_encoder)

print(f"\n编码器输出形状: {x_encoder.shape}  ← (batch=1, src_len=3, d=8)")
print(f"解码器输入形状: {x_decoder.shape}  ← (batch=1, tgt_len=4, d=8)")
print(f"注意力输出形状: {output.shape}     ← (batch=1, tgt_len=4, d=8)")
print(f"权重矩阵形状:   {weights.shape}    ← (batch=1, tgt_len=4, src_len=3)")

# 展示交叉注意力权重
print(f"\n交叉注意力权重（decoder 查询 encoder）：")
print(f"{'':10s}", end="")
for t in src_tokens:
    print(f"{t:>8s}", end="")
print()
w = weights[0].numpy()
for i, token in enumerate(tgt_tokens):
    print(f"{token:10s}", end="")
    for j in range(src_len):
        print(f"{w[i][j]:8.3f}", end="")
    print()

print("\n关键观察：")
print("  - 权重矩阵不是方阵！形状是 (tgt_len, src_len)")
print("  - decoder 的每个 token 都能看到 encoder 的所有 token")
print("  - 理想情况下 'I' 应该关注 '我'，'like' 关注 '喜欢'，'cats' 关注 '猫'")
print("  - 与自注意力的区别：Q 和 K/V 来自不同的序列")


# ════════════════════════════════════════════════════════════════════
# 第三部分：因果掩码 / 掩码自注意力（Causal Attention）
# ════════════════════════════════════════════════════════════════════
# 因果注意力 = 自注意力 + 上三角掩码
#
# 为什么需要掩码？
#   自回归语言模型（GPT）在生成第 i 个 token 时，
#   只能看到前面的 token（位置 0 到 i-1），不能"偷看"后面的。
#
# 实现方法：
#   1. 创建一个上三角矩阵，对角线以上填 -∞（或一个很大的负数）
#   2. 把这个掩码加到注意力分数上
#   3. softmax(-∞) = 0，所以未来位置的权重被置零
#
# 掩码矩阵长这样（5个位置的例子）：
#   位置0:  0    -∞   -∞   -∞   -∞    ← 只能看自己
#   位置1:  0     0   -∞   -∞   -∞    ← 能看 0 和 1
#   位置2:  0     0    0   -∞   -∞    ← 能看 0、1、2
#   位置3:  0     0    0    0   -∞    ← 能看 0、1、2、3
#   位置4:  0     0    0    0    0    ← 能看所有（最后一个位置）
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第三部分：因果注意力 / 掩码自注意力（Causal Attention）")
print("=" * 60)


def create_causal_mask(seq_len):
    """
    创建因果掩码矩阵。

    上三角部分填 -inf，其余填 0。
    加到注意力分数上后，softmax 会将未来位置的权重置零。

    参数:
        seq_len : 序列长度

    返回:
        mask : (seq_len, seq_len) 的掩码张量
    """
    # torch.triu 返回上三角矩阵，diagonal=1 表示从主对角线上方一行开始
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # 上三角部分填 -inf，其他部分填 0
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


# --- 展示因果掩码 ---
seq_len = 5
causal_mask = create_causal_mask(seq_len)

print(f"\n因果掩码矩阵（{seq_len}x{seq_len}）：")
print("  0 = 可以看到，-inf = 不能看到\n")
for i in range(seq_len):
    print(f"  位置{i}: ", end="")
    for j in range(seq_len):
        val = causal_mask[i][j].item()
        if val == float('-inf'):
            print(f"{'   -inf':>7s}", end="")
        else:
            print(f"{'  0.000':>7s}", end="")
    visible = i + 1
    print(f"    ← 能看到 {visible} 个位置")


class CausalSelfAttention(nn.Module):
    """
    因果自注意力模块（GPT 的核心）。

    在标准自注意力的基础上加了因果掩码，
    确保位置 i 只能关注位置 0 到 i。

    参数:
        d_model : 输入/输出维度
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        参数:
            x : (batch, seq_len, d_model)

        返回:
            output  : (batch, seq_len, d_model)
            weights : (batch, seq_len, seq_len) — 因果注意力权重
        """
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # === 关键步骤：加上因果掩码 ===
        seq_len = x.size(1)
        mask = create_causal_mask(seq_len).to(x.device)
        scores = scores + mask  # -inf 位置在 softmax 后变成 0

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        return output, weights


# ════════════════════════════════════════════════════════════════════
# 第四部分：因果注意力效果验证
# ════════════════════════════════════════════════════════════════════
# 验证：位置 i 是否真的只能看到 ≤ i 的位置？
# 方法：检查注意力权重矩阵的上三角部分是否全为 0
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第四部分：因果注意力效果验证")
print("=" * 60)

seq_len, d_model = 6, 8
tokens_gpt = ["I", "love", "deep", "learn", "-ing", "<eos>"]

x = torch.randn(1, seq_len, d_model)
causal_attn = CausalSelfAttention(d_model)

with torch.no_grad():
    output, weights = causal_attn(x)

print(f"\n因果注意力权重矩阵：")
print(f"{'':10s}", end="")
for t in tokens_gpt:
    print(f"{t:>8s}", end="")
print()

w = weights[0].numpy()
for i, token in enumerate(tokens_gpt):
    print(f"{token:10s}", end="")
    for j in range(seq_len):
        if w[i][j] < 1e-7:
            print(f"{'---':>8s}", end="")  # 被掩码屏蔽的位置
        else:
            print(f"{w[i][j]:8.3f}", end="")
    print(f"  | 和={w[i].sum():.3f}")

# 验证上三角是否全为零
upper_triangle = w[torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()]
is_causal = (upper_triangle < 1e-7).all()
print(f"\n验证：上三角权重全部为 0？ → {'是！因果掩码生效' if is_causal else '否！有 bug'}")

print("\n关键观察：")
print("  - '---' 表示权重为 0（被掩码屏蔽）")
print("  - 位置 0（'I'）只能看到自己 → 权重 1.000")
print("  - 位置 1（'love'）能看到 'I' 和 'love' 两个位置")
print("  - 位置 5（'<eos>'）能看到所有 6 个位置")
print("  - 对比自注意力：那里每个位置都能看到所有位置")

# --- 自注意力 vs 因果注意力 的权重矩阵对比 ---
print("\n--- 自注意力 vs 因果注意力 权重矩阵对比 ---")

self_attn_2 = SelfAttention(d_model)
with torch.no_grad():
    _, w_self = self_attn_2(x)
    _, w_causal = causal_attn(x)

print("\n  自注意力（所有位置都可见）:")
w_s = w_self[0].numpy()
for i in range(seq_len):
    print(f"    位置{i}: ", end="")
    for j in range(seq_len):
        print(f"{w_s[i][j]:.3f} ", end="")
    print()

print("\n  因果注意力（只能看到当前及之前）:")
w_c = w_causal[0].numpy()
for i in range(seq_len):
    print(f"    位置{i}: ", end="")
    for j in range(seq_len):
        if w_c[i][j] < 1e-7:
            print(f" ---  ", end="")
        else:
            print(f"{w_c[i][j]:.3f} ", end="")
    print()


# ════════════════════════════════════════════════════════════════════
# 第五部分：MQA — 多查询注意力（Multi-Query Attention）
# ════════════════════════════════════════════════════════════════════
# 标准 MHA：每个注意力头有独立的 Q, K, V
# MQA：所有头共享同一组 K 和 V，只有 Q 是独立的
#
# 好处：
#   - KV Cache 从 h 份减少到 1 份，推理显存减少 h 倍
#   - 推理速度更快（KV 矩阵更小，计算量少）
# 坏处：
#   - 所有头被迫使用相同的 K/V，表达能力下降
#   - 训练质量可能略有降低
#
# 原始论文：Fast Transformer Decoding (Shazeer, 2019)
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第五部分：MQA — 多查询注意力（Multi-Query Attention）")
print("=" * 60)


class MultiQueryAttention(nn.Module):
    """
    多查询注意力（Multi-Query Attention）。

    每个注意力头有独立的 Q 投影，但所有头共享同一组 K 和 V。
    KV Cache 只需要存 1 份，而不是 h 份。

    参数:
        d_model  : 模型维度
        n_heads  : 注意力头数
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Q 投影：每个头有独立的 Q（所以输出维度 = d_model = n_heads * d_head）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        # K, V 投影：所有头共享（输出维度 = d_head，只有一份）
        self.W_k = nn.Linear(d_model, self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, self.d_head, bias=False)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        参数:
            x    : (batch, seq_len, d_model)
            mask : 可选的注意力掩码

        返回:
            output : (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Q: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_head)
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, d_head)

        # K, V: (batch, seq_len, d_head) → 只有 1 份，所有头共享
        K = self.W_k(x)  # (batch, seq_len, d_head)
        V = self.W_v(x)  # (batch, seq_len, d_head)

        # 为了与 Q 做矩阵乘法，给 K, V 增加一个 head 维度
        # (batch, seq_len, d_head) → (batch, 1, seq_len, d_head)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
        # 广播机制：(batch, n_heads, seq_len, d_head) @ (batch, 1, d_head, seq_len)
        # → 自动扩展为 (batch, n_heads, seq_len, seq_len)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)  # (batch, n_heads, seq_len, d_head)

        # 拼接所有头的输出
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.W_o(context)

        return output


# --- 演示 MQA ---
batch, seq_len, d_model, n_heads = 1, 6, 32, 8
x = torch.randn(batch, seq_len, d_model)

mqa = MultiQueryAttention(d_model, n_heads)
with torch.no_grad():
    output = mqa(x)

print(f"\nMQA 配置: d_model={d_model}, n_heads={n_heads}, d_head={d_model // n_heads}")
print(f"输入形状:  {x.shape}")
print(f"输出形状:  {output.shape}")

# 参数量分析
q_params = d_model * d_model        # W_q: 每个头独立的 Q
k_params = d_model * (d_model // n_heads)  # W_k: 只有 1 份
v_params = d_model * (d_model // n_heads)  # W_v: 只有 1 份
o_params = d_model * d_model        # W_o: 输出投影
total_mqa = q_params + k_params + v_params + o_params

print(f"\nMQA 参数量分析:")
print(f"  W_q: {d_model} x {d_model} = {q_params}")
print(f"  W_k: {d_model} x {d_model // n_heads} = {k_params}  ← 只有 1 份（所有头共享）")
print(f"  W_v: {d_model} x {d_model // n_heads} = {v_params}  ← 只有 1 份（所有头共享）")
print(f"  W_o: {d_model} x {d_model} = {o_params}")
print(f"  总计: {total_mqa}")


# ════════════════════════════════════════════════════════════════════
# 第六部分：GQA — 分组查询注意力（Grouped-Query Attention）
# ════════════════════════════════════════════════════════════════════
# GQA 是 MHA 和 MQA 的折中方案：
#   - MHA: g = h（每个头独立 KV） → 质量最好，内存最大
#   - MQA: g = 1（所有头共享 KV） → 内存最小，质量可能下降
#   - GQA: 1 < g < h（每组内共享 KV） → 在两者之间取得平衡
#
# 例：8 个注意力头，4 个 KV 组
#   头 0, 1 → 共享 KV 组 0
#   头 2, 3 → 共享 KV 组 1
#   头 4, 5 → 共享 KV 组 2
#   头 6, 7 → 共享 KV 组 3
#
# LLaMA 2 (70B): 64 个 Q 头, 8 个 KV 组 → 每 8 个 Q 头共享 1 组 KV
# LLaMA 3 (8B):  32 个 Q 头, 8 个 KV 组 → 每 4 个 Q 头共享 1 组 KV
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第六部分：GQA — 分组查询注意力（Grouped-Query Attention）")
print("=" * 60)


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped-Query Attention）。

    n_heads 个 Q 头，n_kv_groups 个 KV 组。
    每 (n_heads // n_kv_groups) 个 Q 头共享同一组 K 和 V。

    特殊情况：
      - n_kv_groups == n_heads → 标准 MHA
      - n_kv_groups == 1      → MQA

    参数:
        d_model    : 模型维度
        n_heads    : Q 头的数量
        n_kv_groups: KV 组的数量
    """

    def __init__(self, d_model, n_heads, n_kv_groups):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert n_heads % n_kv_groups == 0, "n_heads 必须能被 n_kv_groups 整除"

        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.heads_per_group = n_heads // n_kv_groups  # 每组里有几个 Q 头
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Q 投影：n_heads 个独立的头
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # K, V 投影：只有 n_kv_groups 组
        self.W_k = nn.Linear(d_model, n_kv_groups * self.d_head, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_groups * self.d_head, bias=False)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        参数:
            x    : (batch, seq_len, d_model)
            mask : 可选的注意力掩码

        返回:
            output : (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Q: (batch, seq_len, n_heads * d_head) → (batch, n_heads, seq_len, d_head)
        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)

        # K, V: (batch, seq_len, n_kv_groups * d_head) → (batch, n_kv_groups, seq_len, d_head)
        K = self.W_k(x).view(batch, seq_len, self.n_kv_groups, self.d_head)
        K = K.transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_kv_groups, self.d_head)
        V = V.transpose(1, 2)

        # 关键步骤：将 KV 扩展到与 Q 相同的头数
        # (batch, n_kv_groups, seq_len, d_head)
        # → (batch, n_kv_groups, 1, seq_len, d_head)
        # → (batch, n_kv_groups, heads_per_group, seq_len, d_head)
        # → (batch, n_heads, seq_len, d_head)
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        K = K.reshape(batch, self.n_heads, seq_len, self.d_head)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.reshape(batch, self.n_heads, seq_len, self.d_head)

        # 标准注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        # 拼接输出
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.W_o(context)

        return output


# --- 演示 GQA ---
batch, seq_len, d_model = 1, 6, 32
n_heads, n_kv_groups = 8, 4  # 每 2 个 Q 头共享 1 组 KV

x = torch.randn(batch, seq_len, d_model)
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_groups)

with torch.no_grad():
    output = gqa(x)

print(f"\nGQA 配置: d_model={d_model}, n_heads={n_heads}, n_kv_groups={n_kv_groups}")
print(f"  每组内的 Q 头数: {n_heads // n_kv_groups}")
print(f"  → 头 0,1 共享 KV 组 0")
print(f"  → 头 2,3 共享 KV 组 1")
print(f"  → 头 4,5 共享 KV 组 2")
print(f"  → 头 6,7 共享 KV 组 3")
print(f"输入形状:  {x.shape}")
print(f"输出形状:  {output.shape}")

# 验证特殊情况
print("\n验证 GQA 的两个特殊情况：")
gqa_as_mha = GroupedQueryAttention(d_model, n_heads=8, n_kv_groups=8)
gqa_as_mqa = GroupedQueryAttention(d_model, n_heads=8, n_kv_groups=1)
print(f"  n_kv_groups = n_heads = 8 → 等价于 MHA")
print(f"  n_kv_groups = 1           → 等价于 MQA")


# ════════════════════════════════════════════════════════════════════
# 第七部分：MHA vs MQA vs GQA 全面对比
# ════════════════════════════════════════════════════════════════════
# 从三个维度对比：
#   1. 参数量（模型大小）
#   2. KV Cache 内存（推理显存）
#   3. 前向传播速度
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第七部分：MHA vs MQA vs GQA 全面对比")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力（MHA），作为对比基线。

    每个头有独立的 Q, K, V 投影。

    参数:
        d_model : 模型维度
        n_heads : 注意力头数
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # 每个头独立的 K
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # 每个头独立的 V
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(context)


# ---- 1. 参数量对比 ----
print("\n--- 1. 参数量对比 ---")
d_model, n_heads = 512, 8
d_head = d_model // n_heads

mha = MultiHeadAttention(d_model, n_heads)
mqa = MultiQueryAttention(d_model, n_heads)
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_groups=4)

# 计算参数量
def count_params(model):
    """统计模型的可训练参数总数"""
    return sum(p.numel() for p in model.parameters())

mha_params = count_params(mha)
mqa_params = count_params(mqa)
gqa_params = count_params(gqa)

print(f"\n  模型配置: d_model={d_model}, n_heads={n_heads}, d_head={d_head}")
print(f"  GQA 配置: n_kv_groups=4（每 2 个 Q 头共享 1 组 KV）\n")
print(f"  {'方法':15s} {'参数量':>10s} {'相对 MHA':>12s}")
print(f"  {'-'*40}")
print(f"  {'MHA':15s} {mha_params:>10,} {'100%':>12s}")
print(f"  {'GQA (g=4)':15s} {gqa_params:>10,} {gqa_params/mha_params*100:>11.1f}%")
print(f"  {'MQA (g=1)':15s} {mqa_params:>10,} {mqa_params/mha_params*100:>11.1f}%")

# 手动拆解参数量计算
print(f"\n  参数量拆解:")
print(f"  ┌─────────┬──────────────────────┬──────────────────────┬──────────────────────┐")
print(f"  │  投影   │  MHA                 │  GQA (g=4)           │  MQA (g=1)           │")
print(f"  ├─────────┼──────────────────────┼──────────────────────┼──────────────────────┤")
print(f"  │  W_q    │  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{d_model} = {d_model*d_model:>6} │")
print(f"  │  W_k    │  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{4*d_head:>3} = {d_model*4*d_head:>6} │"
      f"  {d_model}x{d_head:>2}  = {d_model*d_head:>6} │")
print(f"  │  W_v    │  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{4*d_head:>3} = {d_model*4*d_head:>6} │"
      f"  {d_model}x{d_head:>2}  = {d_model*d_head:>6} │")
print(f"  │  W_o    │  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{d_model} = {d_model*d_model:>6} │"
      f"  {d_model}x{d_model} = {d_model*d_model:>6} │")
print(f"  ├─────────┼──────────────────────┼──────────────────────┼──────────────────────┤")
print(f"  │  总计   │  {mha_params:>20,} │  {gqa_params:>20,} │  {mqa_params:>20,} │")
print(f"  └─────────┴──────────────────────┴──────────────────────┴──────────────────────┘")


# ---- 2. KV Cache 内存对比 ----
print("\n--- 2. KV Cache 内存对比（推理阶段的关键） ---")
print("""
  自回归生成时，每一步只需要新计算 1 个 token 的 Q，
  但需要用到之前所有 token 的 K 和 V。
  所以我们把 K、V 缓存起来，称为 KV Cache。

  KV Cache 大小 = 2 × n_kv_heads × seq_len × d_head × bytes_per_element
  （2 是因为有 K 和 V 两部分）
""")

seq_lens = [512, 2048, 8192, 32768]

print(f"  {'序列长度':>10s} │ {'MHA Cache':>12s} │ {'GQA Cache':>12s} │ {'MQA Cache':>12s} │ {'MQA节省':>8s}")
print(f"  {'-'*70}")

for sl in seq_lens:
    # KV Cache = 2(K和V) × n_kv_heads × seq_len × d_head × 2(float16)
    mha_cache = 2 * n_heads * sl * d_head * 2         # n_kv_heads = n_heads
    gqa_cache = 2 * 4 * sl * d_head * 2               # n_kv_heads = 4
    mqa_cache = 2 * 1 * sl * d_head * 2               # n_kv_heads = 1

    def fmt_bytes(b):
        """将字节数格式化为可读字符串"""
        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b/1024:.1f} KB"
        else:
            return f"{b/(1024*1024):.1f} MB"

    saving = (1 - mqa_cache / mha_cache) * 100
    print(f"  {sl:>10,} │ {fmt_bytes(mha_cache):>12s} │ {fmt_bytes(gqa_cache):>12s} │"
          f" {fmt_bytes(mqa_cache):>12s} │ {saving:>6.1f}%")


# ---- 3. 前向传播速度对比 ----
print("\n--- 3. 前向传播速度对比 ---")

batch, seq_len, d_model, n_heads = 4, 128, 256, 8

mha = MultiHeadAttention(d_model, n_heads)
mqa = MultiQueryAttention(d_model, n_heads)
gqa = GroupedQueryAttention(d_model, n_heads, n_kv_groups=4)

x = torch.randn(batch, seq_len, d_model)
n_runs = 100

# 预热
for _ in range(10):
    with torch.no_grad():
        _ = mha(x)
        _ = mqa(x)
        _ = gqa(x)

# 计时
results = {}
for name, model in [("MHA", mha), ("GQA (g=4)", gqa), ("MQA (g=1)", mqa)]:
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # 毫秒
    results[name] = elapsed

# 按速度排序展示
print(f"\n  配置: batch={batch}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
print(f"  运行 {n_runs} 次取平均（CPU）\n")
print(f"  {'方法':15s} {'平均耗时':>10s} {'相对 MHA':>12s}")
print(f"  {'-'*40}")
mha_time = results["MHA"]
for name, t in results.items():
    speedup = mha_time / t
    print(f"  {name:15s} {t:>8.2f}ms {speedup:>11.2f}x")


# ---- 4. 综合对比表格 ----
print(f"\n--- 4. 综合对比 ---")
print("""
  ┌──────────────┬────────────────┬──────────────┬─────────────┬──────────────┐
  │  方法        │  KV 头数       │  KV 参数量   │  KV Cache   │  质量        │
  ├──────────────┼────────────────┼──────────────┼─────────────┼──────────────┤
  │  MHA         │  h（全部独立） │  最大        │  最大       │  最好        │
  │  GQA         │  g（分组共享） │  中等        │  中等       │  接近 MHA    │
  │  MQA         │  1（全部共享） │  最小        │  最小       │  略有下降    │
  └──────────────┴────────────────┴──────────────┴─────────────┴──────────────┘

  实际使用情况：
  ┌──────────────────────────┬──────────────────────────────────┐
  │  模型                   │  注意力方案                      │
  ├──────────────────────────┼──────────────────────────────────┤
  │  GPT-3, GPT-4           │  MHA（标准多头注意力）           │
  │  PaLM                   │  MQA（多查询注意力）             │
  │  LLaMA 2 (70B)          │  GQA (64Q头, 8KV组)             │
  │  LLaMA 3 (8B)           │  GQA (32Q头, 8KV组)             │
  │  Qwen 2                 │  GQA                             │
  │  DeepSeek-V2            │  MLA（多头潜在注意力，更激进）   │
  └──────────────────────────┴──────────────────────────────────┘

  趋势：新模型几乎都使用 GQA，因为它在质量和效率之间取得了最佳平衡。
""")


# ════════════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("总结")
print("=" * 60)
print("""
  本节学到了注意力的四大变体：

  1. 自注意力（Self-Attention）
     - Q, K, V 来自同一个序列
     - 每个位置能看到所有位置（包括自己）
     - 用于: Transformer Encoder (BERT)

  2. 交叉注意力（Cross-Attention）
     - Q 来自序列 A，K/V 来自序列 B
     - 让一个序列"查询"另一个序列的信息
     - 用于: Transformer Decoder 的第二层、多模态融合

  3. 因果注意力（Causal Attention）
     - 自注意力 + 上三角 -inf 掩码
     - 位置 i 只能看到 ≤ i 的位置
     - 用于: GPT、LLaMA 等自回归模型

  4. MHA / MQA / GQA
     - MHA: 每个头独立 KV → 质量最好，内存最大
     - MQA: 所有头共享 KV → 内存最小，质量略降
     - GQA: 分组共享 KV   → 最佳平衡，现代主流

  关键公式：
     Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

  下一节预告：第6章 · 第5节 · 位置编码
""")


# ════════════════════════════════════════════════════════════════════
# 思考题
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【自注意力 vs 交叉注意力】
   自注意力的权重矩阵是方阵（seq_len x seq_len），
   交叉注意力的权重矩阵是矩形（tgt_len x src_len）。
   如果编码器输出 100 个 token，解码器当前有 20 个 token，
   交叉注意力的权重矩阵形状是多少？计算量和哪个长度的关系更大？
   提示：形状是 (20, 100)，计算量 = O(tgt_len × src_len × d_model)。

2. 【因果掩码的数学】
   假设位置 2 的注意力分数（未掩码前）是 [0.5, 1.2, 0.8, 2.0, 0.3]，
   加上因果掩码后变成什么？softmax 后的权重是什么？
   手动计算并验证位置 3, 4 的权重确实为 0。
   提示：加掩码后 = [0.5, 1.2, 0.8, -inf, -inf]，
   然后对 [0.5, 1.2, 0.8] 做 softmax。

3. 【GQA 的组数选择】
   LLaMA 2 (70B) 用 64 个 Q 头和 8 个 KV 组。
   如果改成 4 个 KV 组或 16 个 KV 组，分别会怎样？
   极端情况：1 个 KV 组（退化为 MQA）和 64 个 KV 组（退化为 MHA）。
   提示：组数越少 → KV Cache 越小 → 推理越快，但质量可能下降。

4. 【KV Cache 的数学】
   一个 7B 参数量的模型，32 层，32 个 Q 头，8 个 KV 组，
   d_model=4096，用 float16 推理。
   计算处理 4096 长度序列时的 KV Cache 总大小。
   如果用 MHA 而不是 GQA，KV Cache 会大多少？
   提示：KV Cache = 2 × n_layers × n_kv_heads × seq_len × d_head × 2 字节。

5. 【为什么 GPT 需要因果掩码而 BERT 不需要？】
   GPT 是自回归模型（一个一个生成 token），
   BERT 是双向模型（一次看完整个句子再做预测）。
   它们的训练目标有什么根本区别？
   如果 GPT 不用因果掩码会发生什么？
   提示：GPT 预测下一个 token，如果能看到未来，
   就相当于考试时偷看答案——模型学不到真正的语言规律。
""")

print("下一节预告: 第6章 · 第5节 · 位置编码")
