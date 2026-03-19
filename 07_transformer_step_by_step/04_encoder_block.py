"""
====================================================================
第7章 · 第4节 · Encoder 模块组装
====================================================================

【一句话总结】
将前面的所有组件——多头注意力、FFN、LayerNorm、残差连接——
组装成一个完整的 Transformer Encoder Block。

【为什么深度学习需要这个？】
- Encoder 是 BERT、ViT 等模型的核心
- 理解 Encoder 的组装方式是理解 Decoder 的前提
- 一个 Encoder Block 只有几十行代码，但每行都有讲究

【核心概念】

1. Encoder Block 的结构（Pre-Norm 版本）
   - 子层1：多头自注意力
     x = x + MultiHeadAttention(LayerNorm(x))
   - 子层2：前馈网络
     x = x + FFN(LayerNorm(x))
   - 就是这么简单！两个子层，每个都有 LN + 残差

2. Encoder 堆叠
   - 一个 Encoder = N 个相同的 Encoder Block 堆叠
   - 原始 Transformer: N=6
   - BERT-base: N=12, BERT-large: N=24
   - 每一层的权重不共享（独立训练）

3. Padding Mask
   - batch 中的序列长度不同，短的需要填充(pad)
   - Padding 位置不应该参与注意力计算
   - 实现：在注意力分数中将 pad 位置设为 -∞

4. 完整的 Encoder 流程
   input_ids → Embedding + PositionalEncoding → [EncoderBlock × N] → output
   输出的每个位置都包含了整个序列的上下文信息

【前置知识】
第6章 - 多头注意力，第7章第1-3节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)
torch.manual_seed(42)
np.random.seed(42)


# ====================================================================
# 第1部分：回顾所有组件 —— 从零组装 Encoder 需要哪些积木？
# ====================================================================
#
# Encoder Block 由以下组件构成：
#   1. 多头自注意力 (Multi-Head Self-Attention)    ← 第6章
#   2. 前馈网络 (Position-wise FFN)                ← 第7章第2节
#   3. 层归一化 (LayerNorm)                        ← 第7章第3节
#   4. 残差连接 (Residual Connection)              ← 第7章第1节
#   5. Dropout                                      ← 正则化
#
# 这里我们将每个组件紧凑地重新实现一遍，方便后续组装。
# 如果你忘记了某个组件的原理，请回到对应章节复习。

print("=" * 60)
print("第1部分：回顾所有组件")
print("=" * 60)


# ---- 组件1：Scaled Dot-Product Attention ----
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力——Transformer 注意力的核心计算。

    公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V

    参数:
        Q    : (batch, heads, seq_len, d_k)  查询矩阵
        K    : (batch, heads, seq_len, d_k)  键矩阵
        V    : (batch, heads, seq_len, d_k)  值矩阵
        mask : (batch, 1, 1, seq_len) 或 None  掩码（True 表示被遮挡）

    返回:
        output  : (batch, heads, seq_len, d_k)  注意力输出
        weights : (batch, heads, seq_len, seq_len)  注意力权重
    """
    d_k = Q.size(-1)
    # 第1步：计算注意力分数 QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 第2步：应用掩码（将 padding 位置设为 -inf，softmax 后权重趋近 0）
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # 第3步：softmax 归一化
    weights = F.softmax(scores, dim=-1)

    # 第4步：加权求和 Value
    output = torch.matmul(weights, V)
    return output, weights


# ---- 组件2：多头注意力 ----
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制——让模型同时从多个"视角"关注不同的信息。

    核心思想：
    - 将 d_model 维的 Q/K/V 分成 n_heads 个头
    - 每个头独立计算注意力
    - 最后拼接所有头的输出，通过线性层投影

    参数:
        d_model : 模型维度（如 512）
        n_heads : 注意力头数（如 8）
        dropout : dropout 比率
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q/K/V 的线性投影层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影层：将多头拼接结果映射回 d_model 维
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 保存注意力权重，供可视化使用
        self.attn_weights = None

    def forward(self, x, mask=None):
        """
        前向传播：自注意力（Q=K=V=x）。

        参数:
            x    : (batch, seq_len, d_model)  输入序列
            mask : (batch, 1, 1, seq_len) 或 None  掩码

        返回:
            output : (batch, seq_len, d_model)  注意力输出
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影 → 分头
        # (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # 保存注意力权重（可视化用）
        self.attn_weights = attn_weights.detach()

        # 合并多头：(batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 输出投影 + dropout
        output = self.dropout(self.W_O(attn_output))
        return output


# ---- 组件3：前馈网络 (Position-wise FFN) ----
class PositionWiseFFN(nn.Module):
    """
    位置级前馈网络——对每个位置独立做两层全连接变换。

    结构: x → Linear(d_model, d_ff) → GELU → Dropout → Linear(d_ff, d_model) → Dropout

    参数:
        d_model : 输入/输出维度（如 512）
        d_ff    : 中间层维度（通常 = 4 * d_model，如 2048）
        dropout : dropout 比率
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)      # 扩展维度
        self.fc2 = nn.Linear(d_ff, d_model)       # 压缩回来
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()                     # GELU 激活（GPT/BERT 使用）

    def forward(self, x):
        """
        前向传播。

        参数:
            x : (batch, seq_len, d_model)

        返回:
            output : (batch, seq_len, d_model)
        """
        x = self.fc1(x)          # (batch, seq_len, d_ff)
        x = self.gelu(x)         # 非线性激活
        x = self.dropout(x)      # 正则化
        x = self.fc2(x)          # (batch, seq_len, d_model)
        x = self.dropout(x)
        return x


# ---- 组件4：位置编码 ----
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码——给序列的每个位置注入位置信息。

    公式:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    参数:
        d_model : 模型维度
        max_len : 支持的最大序列长度
        dropout : dropout 比率
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码表（不需要梯度）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维：cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        将位置编码加到输入上。

        参数:
            x : (batch, seq_len, d_model)

        返回:
            output : (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


print("  所有组件已定义完毕：")
print("    - scaled_dot_product_attention(): 缩放点积注意力")
print("    - MultiHeadAttention:  多头自注意力")
print("    - PositionWiseFFN:     前馈网络")
print("    - PositionalEncoding:  正弦位置编码")
print("    - nn.LayerNorm:        层归一化（PyTorch内置）")
print("  接下来，把它们组装成 Encoder Block！")


# ====================================================================
# 第2部分：EncoderBlock 类 —— 一个 Encoder 层的完整实现
# ====================================================================
#
# Encoder Block 的结构（Pre-Norm 版本）：
#
#   输入 x
#    │
#    ├──────────────────────────────┐
#    ↓                              │ (残差连接)
#   LayerNorm                       │
#    ↓                              │
#   Multi-Head Self-Attention       │
#    ↓                              │
#    + ←────────────────────────────┘
#    │
#    ├──────────────────────────────┐
#    ↓                              │ (残差连接)
#   LayerNorm                       │
#    ↓                              │
#   Feed-Forward Network            │
#    ↓                              │
#    + ←────────────────────────────┘
#    │
#   输出
#
# Pre-Norm vs Post-Norm:
#   - Post-Norm (原始 Transformer): x = LayerNorm(x + Sublayer(x))
#   - Pre-Norm  (现代常用):         x = x + Sublayer(LayerNorm(x))
#   - Pre-Norm 训练更稳定，不需要 warm-up，现代模型普遍采用

print("\n\n" + "=" * 60)
print("第2部分：EncoderBlock 类")
print("=" * 60)


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block（Pre-Norm 版本）。

    一个 Block 包含两个子层：
      子层1: LayerNorm → Multi-Head Self-Attention → Dropout → 残差连接
      子层2: LayerNorm → Position-wise FFN → Dropout → 残差连接

    参数:
        d_model : 模型维度
        n_heads : 注意力头数
        d_ff    : FFN 中间层维度
        dropout : dropout 比率
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # 子层1：多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 子层2：前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout（用于残差连接后，虽然子层内部已有dropout，
        # 这里的 dropout 作用于整个子层输出，是一层额外的正则化）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向传播。

        参数:
            x    : (batch, seq_len, d_model)  输入
            mask : padding mask 或 None

        返回:
            output : (batch, seq_len, d_model)  输出（与输入同形状）
        """
        # ---- 子层1: Pre-Norm + Self-Attention + 残差 ----
        # Pre-Norm: 先归一化，再送入子层
        residual = x
        x = self.norm1(x)                    # LayerNorm
        x = self.self_attn(x, mask=mask)     # Multi-Head Self-Attention
        x = residual + x                     # 残差连接

        # ---- 子层2: Pre-Norm + FFN + 残差 ----
        residual = x
        x = self.norm2(x)                    # LayerNorm
        x = self.ffn(x)                      # Feed-Forward Network
        x = residual + x                     # 残差连接

        return x


# 实例化并测试
d_model, n_heads, d_ff = 64, 8, 256
encoder_block = EncoderBlock(d_model, n_heads, d_ff, dropout=0.1)
print(f"\n  EncoderBlock 结构:")
print(f"  {encoder_block}")

# 统计参数量
total_params = sum(p.numel() for p in encoder_block.parameters())
print(f"\n  单个 EncoderBlock 参数量: {total_params:,}")
print(f"  分解：")
print(f"    多头注意力 (W_Q + W_K + W_V + W_O): "
      f"{sum(p.numel() for p in encoder_block.self_attn.parameters()):,}")
print(f"    FFN (fc1 + fc2): "
      f"{sum(p.numel() for p in encoder_block.ffn.parameters()):,}")
print(f"    LayerNorm x2: "
      f"{sum(p.numel() for p in encoder_block.norm1.parameters()) + sum(p.numel() for p in encoder_block.norm2.parameters()):,}")

# 测试前向传播
test_input = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
encoder_block.eval()
with torch.no_grad():
    test_output = encoder_block(test_input)
print(f"\n  输入形状:  {list(test_input.shape)}")
print(f"  输出形状:  {list(test_output.shape)}")
print(f"  形状一致:  {test_input.shape == test_output.shape}  (残差连接保证输入输出同形状)")


# ====================================================================
# 第3部分：TransformerEncoder 类 —— 堆叠 N 个 Encoder Block
# ====================================================================
#
# 完整的 Transformer Encoder 流程：
#
#   input_ids (batch, seq_len)
#       ↓
#   Token Embedding (词嵌入)
#       ↓
#   + Positional Encoding (位置编码)
#       ↓
#   Dropout
#       ↓
#   ┌─────────────────┐
#   │  EncoderBlock 1  │
#   ├─────────────────┤
#   │  EncoderBlock 2  │     N 层堆叠
#   ├─────────────────┤     （每层权重独立）
#   │       ...        │
#   ├─────────────────┤
#   │  EncoderBlock N  │
#   └─────────────────┘
#       ↓
#   Final LayerNorm (Pre-Norm 需要在最后加一层 LN)
#       ↓
#   output (batch, seq_len, d_model)

print("\n\n" + "=" * 60)
print("第3部分：TransformerEncoder 类")
print("=" * 60)


class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder。

    包含：词嵌入 + 位置编码 + N 个 Encoder Block + 最终 LayerNorm。

    参数:
        vocab_size : 词表大小
        d_model    : 模型维度
        n_heads    : 注意力头数
        d_ff       : FFN 中间层维度
        n_layers   : Encoder Block 的数量（堆叠层数）
        max_len    : 支持的最大序列长度
        dropout    : dropout 比率
        pad_idx    : padding token 的索引（用于生成 mask）
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # 词嵌入层：将整数 token ID 映射为 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码：注入位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N 个 Encoder Block（注意：每层权重独立，不共享！）
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Pre-Norm 方案需要在最后加一层 LayerNorm
        # 因为最后一个 Block 的输出没有经过 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

        # 嵌入缩放因子（原始 Transformer 论文的做法）
        # 嵌入向量的值通常较小，乘以 sqrt(d_model) 使其与位置编码在同一量级
        self.embed_scale = math.sqrt(d_model)

    def create_padding_mask(self, input_ids):
        """
        创建 Padding Mask。

        将 padding 位置标记为 True，表示这些位置在注意力计算中应该被忽略。

        参数:
            input_ids : (batch, seq_len)  输入的 token ID 序列

        返回:
            mask : (batch, 1, 1, seq_len)  padding mask
                   True 表示该位置是 padding，需要被遮挡
        """
        # pad_idx 位置为 True（需要被遮挡）
        mask = (input_ids == self.pad_idx)  # (batch, seq_len)
        # 扩展维度以适配注意力分数矩阵 (batch, n_heads, seq_q, seq_k)
        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        return mask

    def forward(self, input_ids, verbose=False):
        """
        前向传播。

        参数:
            input_ids : (batch, seq_len)  输入的 token ID 序列
            verbose   : 是否打印中间形状（维度追踪）

        返回:
            output : (batch, seq_len, d_model)  编码器输出
        """
        # ---- 创建 Padding Mask ----
        mask = self.create_padding_mask(input_ids)

        if verbose:
            print(f"\n  [维度追踪] 输入 input_ids: {list(input_ids.shape)}")
            print(f"  [维度追踪] Padding mask:  {list(mask.shape)}")

        # ---- Token Embedding ----
        x = self.token_embedding(input_ids)  # (batch, seq_len) → (batch, seq_len, d_model)
        x = x * self.embed_scale             # 缩放嵌入向量

        if verbose:
            print(f"  [维度追踪] 词嵌入后:      {list(x.shape)}")

        # ---- Positional Encoding ----
        x = self.pos_encoding(x)  # 加上位置编码 + dropout

        if verbose:
            print(f"  [维度追踪] 位置编码后:    {list(x.shape)}")

        # ---- 逐层通过 Encoder Blocks ----
        for i, layer in enumerate(self.layers):
            x = layer(x, mask=mask)
            if verbose:
                print(f"  [维度追踪] Encoder层{i+1}后:  {list(x.shape)}")

        # ---- 最终 LayerNorm ----
        x = self.final_norm(x)

        if verbose:
            print(f"  [维度追踪] 最终LN后:      {list(x.shape)}")

        return x


# 实例化一个小型 Encoder
vocab_size = 1000
d_model = 64
n_heads = 8
d_ff = 256
n_layers = 4

encoder = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    n_layers=n_layers,
    max_len=128,
    dropout=0.1,
    pad_idx=0,
)

print(f"\n  TransformerEncoder 配置:")
print(f"    词表大小:     {vocab_size}")
print(f"    模型维度:     {d_model}")
print(f"    注意力头数:   {n_heads}")
print(f"    FFN 维度:     {d_ff}")
print(f"    层数:         {n_layers}")
print(f"    每个头维度:   {d_model // n_heads}")

total_params = sum(p.numel() for p in encoder.parameters())
print(f"\n  总参数量: {total_params:,}")
print(f"  分解:")
print(f"    Token Embedding: {encoder.token_embedding.weight.numel():,}")
encoder_block_params = sum(p.numel() for p in encoder.layers[0].parameters())
print(f"    单个 EncoderBlock: {encoder_block_params:,}")
print(f"    {n_layers} 个 EncoderBlock: {encoder_block_params * n_layers:,}")
print(f"    其他 (PosEnc + FinalLN): "
      f"{total_params - encoder.token_embedding.weight.numel() - encoder_block_params * n_layers:,}")


# ====================================================================
# 第4部分：Padding Mask 详解
# ====================================================================
#
# 为什么需要 Padding Mask？
#   - Batch 训练时，不同序列长度不一致
#   - 短序列需要用 pad_token 填充到相同长度
#   - 但 pad 位置不包含有意义的信息，不应该参与注意力计算
#   - 如果不加 mask，模型会"关注"到无意义的 padding，降低性能
#
# Padding Mask 的工作原理：
#   1. 找到 input_ids 中所有 pad_idx 的位置
#   2. 在注意力分数矩阵中，将这些位置设为 -inf
#   3. softmax(-inf) = 0，这些位置的注意力权重为 0
#   4. 效果：模型完全忽略 padding 位置

print("\n\n" + "=" * 60)
print("第4部分：Padding Mask 详解")
print("=" * 60)

# 构造一个包含 padding 的 batch
# 假设 pad_idx=0，三个序列长度分别为 6、4、2
pad_idx = 0
batch_ids = torch.tensor([
    [15, 23, 7, 42, 88, 6],   # 序列1：长度6，无 padding
    [31, 9, 55, 12,  0, 0],   # 序列2：长度4，后2个是 padding
    [44, 8,  0,  0,  0, 0],   # 序列3：长度2，后4个是 padding
])

print(f"\n  输入 batch (pad_idx=0):")
for i, seq in enumerate(batch_ids):
    tokens = seq.tolist()
    real_len = (seq != pad_idx).sum().item()
    print(f"    序列{i+1}: {tokens}  (有效长度={real_len})")

# 创建 padding mask
mask = encoder.create_padding_mask(batch_ids)
print(f"\n  Padding Mask 形状: {list(mask.shape)}")
print(f"  (batch=3, 1, 1, seq_len=6)")

# 展示每个序列的 mask
for i in range(batch_ids.size(0)):
    mask_values = mask[i, 0, 0].tolist()
    print(f"    序列{i+1} mask: {mask_values}")
    print(f"      True 位置 = padding，注意力分数设为 -inf → softmax 后权重为 0")

# 演示 mask 对注意力分数的影响
print(f"\n  --- Mask 对注意力分数的影响演示 ---")
raw_scores = torch.tensor([[1.2, 0.8, 0.5, 1.0, 0.3, 0.7]])  # 假设的注意力分数
mask_demo = torch.tensor([[False, False, False, False, True, True]])  # 后2个是 padding

# 不加 mask
weights_no_mask = F.softmax(raw_scores, dim=-1)
# 加 mask
scores_masked = raw_scores.masked_fill(mask_demo, float("-inf"))
weights_with_mask = F.softmax(scores_masked, dim=-1)

print(f"  原始分数:        {raw_scores[0].tolist()}")
print(f"  Mask 后分数:     {['  -inf' if v == float('-inf') else f'{v:.4f}' for v in scores_masked[0].tolist()]}")
print(f"  无 Mask 权重:    {[f'{w:.3f}' for w in weights_no_mask[0].tolist()]}")
print(f"  有 Mask 权重:    {[f'{w:.3f}' for w in weights_with_mask[0].tolist()]}")
print(f"  无 Mask 权重和:  {weights_no_mask.sum().item():.3f}")
print(f"  有 Mask 权重和:  {weights_with_mask.sum().item():.3f}")
print(f"\n  效果: Padding 位置权重为 0，注意力完全集中在有效 token 上")


# ====================================================================
# 第5部分：维度追踪 —— 数据在 Encoder 中的完整流动
# ====================================================================
#
# 维度追踪是理解 Transformer 架构的关键技巧。
# 我们将打印数据在 Encoder 每一步的形状变化，
# 直观展示"从整数 ID 到上下文化表示"的完整过程。

print("\n\n" + "=" * 60)
print("第5部分：维度追踪")
print("=" * 60)

# 使用第4部分的 batch
print(f"\n  输入 batch_ids 形状: {list(batch_ids.shape)}")
print(f"  配置: vocab_size={vocab_size}, d_model={d_model}, "
      f"n_heads={n_heads}, n_layers={n_layers}")
print(f"\n  {'='*50}")
print(f"  完整前向传播的维度变化:")
print(f"  {'='*50}")

encoder.eval()
with torch.no_grad():
    output = encoder(batch_ids, verbose=True)

print(f"\n  最终输出形状: {list(output.shape)}")
print(f"  含义: batch={output.shape[0]}个序列, 每个序列{output.shape[1]}个位置, "
      f"每个位置{output.shape[2]}维表示")

# 详细拆解单个 EncoderBlock 内部的维度变化
print(f"\n  {'='*50}")
print(f"  单个 EncoderBlock 内部维度拆解:")
print(f"  {'='*50}")

x_demo = torch.randn(3, 6, d_model)  # 模拟 embedding + pos_enc 之后的输入
block = encoder.layers[0]

with torch.no_grad():
    # 子层1: Self-Attention
    residual = x_demo
    normed = block.norm1(x_demo)
    print(f"  输入 x:             {list(x_demo.shape)}")
    print(f"  LayerNorm(x):       {list(normed.shape)}")

    # 多头注意力的内部拆解
    B, S, D = normed.shape
    H = n_heads
    dk = D // H
    Q = block.self_attn.W_Q(normed)
    print(f"  W_Q(x) → Q:        {list(Q.shape)}  (batch, seq, d_model)")
    Q = Q.view(B, S, H, dk).transpose(1, 2)
    print(f"  Q 分头后:           {list(Q.shape)}  (batch, heads, seq, d_k)")
    K = block.self_attn.W_K(normed).view(B, S, H, dk).transpose(1, 2)
    V = block.self_attn.W_V(normed).view(B, S, H, dk).transpose(1, 2)
    print(f"  K 分头后:           {list(K.shape)}")
    print(f"  V 分头后:           {list(V.shape)}")
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
    print(f"  QK^T/sqrt(d_k):    {list(scores.shape)}  (batch, heads, seq, seq)")
    weights = F.softmax(scores, dim=-1)
    print(f"  softmax(scores):    {list(weights.shape)}  (注意力权重)")
    attn_out = torch.matmul(weights, V)
    print(f"  weights @ V:        {list(attn_out.shape)}  (batch, heads, seq, d_k)")
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    print(f"  合并多头后:         {list(attn_out.shape)}  (batch, seq, d_model)")
    attn_out = block.self_attn.W_O(attn_out)
    print(f"  W_O(concat):        {list(attn_out.shape)}  (输出投影)")
    x_after_attn = residual + attn_out
    print(f"  残差连接后:         {list(x_after_attn.shape)}")

    # 子层2: FFN
    residual2 = x_after_attn
    normed2 = block.norm2(x_after_attn)
    print(f"  \n  LayerNorm(x):       {list(normed2.shape)}")
    ffn_mid = block.ffn.fc1(normed2)
    print(f"  FFN fc1:            {list(ffn_mid.shape)}  (扩展到 d_ff={d_ff})")
    ffn_mid = block.ffn.gelu(ffn_mid)
    ffn_out = block.ffn.fc2(ffn_mid)
    print(f"  FFN fc2:            {list(ffn_out.shape)}  (压缩回 d_model={d_model})")
    x_after_ffn = residual2 + ffn_out
    print(f"  残差连接后:         {list(x_after_ffn.shape)}")


# ====================================================================
# 第6部分：完整前向传播 —— 模拟真实场景
# ====================================================================
#
# 模拟一个真实的 NLP 任务场景：
#   - 构造一个小型词表
#   - 创建一个 batch 的输入序列（带 padding）
#   - 将整个 batch 送入 Encoder
#   - 观察输出的特性

print("\n\n" + "=" * 60)
print("第6部分：完整前向传播 —— 模拟真实场景")
print("=" * 60)

# ---- 模拟词表 ----
# 在真实场景中，词表由 tokenizer 构建（如 BPE、WordPiece）
# 这里我们用一个简单的映射来演示
word_to_id = {
    "<pad>": 0,    # padding token，必须是 pad_idx
    "<unk>": 1,    # 未知词
    "我": 2,
    "喜欢": 3,
    "深度": 4,
    "学习": 5,
    "Transformer": 6,
    "很": 7,
    "强大": 8,
    "因为": 9,
    "它": 10,
    "能": 11,
    "理解": 12,
    "语言": 13,
}
id_to_word = {v: k for k, v in word_to_id.items()}

# 构造 3 个不等长的句子
sentences = [
    ["我", "喜欢", "深度", "学习"],              # 长度 4
    ["Transformer", "很", "强大"],               # 长度 3
    ["因为", "它", "能", "理解", "语言"],         # 长度 5
]

# 转为 token ID 并 padding 到最大长度
max_len_batch = max(len(s) for s in sentences)
batch_input = []
for sent in sentences:
    ids = [word_to_id[w] for w in sent]
    ids += [0] * (max_len_batch - len(ids))  # 用 0 (pad) 填充
    batch_input.append(ids)

batch_tensor = torch.tensor(batch_input)

print(f"\n  模拟词表大小: {len(word_to_id)}")
print(f"  输入句子:")
for i, sent in enumerate(sentences):
    ids = batch_input[i]
    print(f"    句{i+1}: {sent}")
    print(f"          → ID: {ids}")
print(f"\n  Batch tensor 形状: {list(batch_tensor.shape)}  (batch={len(sentences)}, max_len={max_len_batch})")

# ---- 构建适配词表大小的 Encoder ----
small_encoder = TransformerEncoder(
    vocab_size=len(word_to_id),
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=3,
    max_len=64,
    dropout=0.0,  # 关闭 dropout 以获得确定性输出
    pad_idx=0,
)
small_encoder.eval()

print(f"\n  小型 Encoder 配置: d_model=32, n_heads=4, d_ff=128, n_layers=3")
print(f"  参数量: {sum(p.numel() for p in small_encoder.parameters()):,}")

# ---- 前向传播 ----
with torch.no_grad():
    output = small_encoder(batch_tensor, verbose=True)

print(f"\n  Encoder 输出形状: {list(output.shape)}")
print(f"  每个位置的表示维度: {output.shape[-1]}")

# ---- 分析输出特性 ----
print(f"\n  --- 输出分析 ---")

# Padding 位置的输出 vs 有效位置的输出
for i, sent in enumerate(sentences):
    real_len = len(sent)
    real_output = output[i, :real_len]     # 有效位置的输出
    pad_output = output[i, real_len:]      # padding 位置的输出
    real_norm = real_output.norm(dim=-1).mean().item()

    print(f"  句{i+1} ({' '.join(sent)}):")
    print(f"    有效位置输出均值范数: {real_norm:.4f}")
    if pad_output.numel() > 0:
        pad_norm = pad_output.norm(dim=-1).mean().item()
        print(f"    Padding位置输出均值范数: {pad_norm:.4f}")
    else:
        print(f"    (无 padding)")

# ---- 上下文化表示的验证 ----
# 同一个词在不同上下文中的表示应该不同
print(f"\n  --- 上下文化表示验证 ---")
# 构造两个包含相同词但不同上下文的句子
context_batch = torch.tensor([
    [word_to_id["我"], word_to_id["喜欢"], word_to_id["学习"], 0, 0],   # "我 喜欢 学习"
    [word_to_id["它"], word_to_id["能"], word_to_id["学习"], 0, 0],     # "它 能 学习"
])

with torch.no_grad():
    ctx_output = small_encoder(context_batch)

# "学习"在两个句子中的表示
repr_1 = ctx_output[0, 2]  # 句1中的"学习"
repr_2 = ctx_output[1, 2]  # 句2中的"学习"
cos_sim = F.cosine_similarity(repr_1.unsqueeze(0), repr_2.unsqueeze(0)).item()

print(f"  '学习'在句1('我 喜欢 学习')的表示 vs 句2('它 能 学习')的表示:")
print(f"    余弦相似度: {cos_sim:.4f}")
print(f"    表示向量差的范数: {(repr_1 - repr_2).norm().item():.4f}")
print(f"    结论: 同一个词在不同上下文中获得了不同的表示 — 这就是上下文化表示！")


# ====================================================================
# 第7部分：注意力可视化 —— 不同层的注意力模式
# ====================================================================
#
# 注意力权重可视化是理解 Transformer 行为的重要工具。
# 不同层的注意力模式通常有不同的特征：
#   - 浅层：倾向于关注相邻的词（局部模式）
#   - 深层：倾向于捕捉更远距离的依赖（全局模式）
#   - 某些头可能专注于特定的语言现象

print("\n\n" + "=" * 60)
print("第7部分：注意力可视化")
print("=" * 60)

# 使用刚才的小型 Encoder，收集每一层、每一个头的注意力权重
vis_encoder = TransformerEncoder(
    vocab_size=len(word_to_id),
    d_model=32,
    n_heads=4,
    d_ff=128,
    n_layers=3,
    max_len=64,
    dropout=0.0,
    pad_idx=0,
)
vis_encoder.eval()

# 输入一个句子
vis_sentence = ["我", "喜欢", "深度", "学习", "Transformer"]
vis_ids = torch.tensor([[word_to_id[w] for w in vis_sentence]])
vis_words = vis_sentence

print(f"\n  可视化句子: {vis_sentence}")
print(f"  Token IDs:  {vis_ids[0].tolist()}")

# 前向传播并收集每层的注意力权重
all_attn_weights = []
with torch.no_grad():
    # 手动逐步通过 Encoder 以收集注意力
    mask = vis_encoder.create_padding_mask(vis_ids)
    x = vis_encoder.token_embedding(vis_ids) * vis_encoder.embed_scale
    x = vis_encoder.pos_encoding(x)

    for layer_idx, layer in enumerate(vis_encoder.layers):
        x = layer(x, mask=mask)
        # 每层的注意力权重保存在 self_attn.attn_weights 中
        attn_w = layer.self_attn.attn_weights  # (batch, n_heads, seq, seq)
        all_attn_weights.append(attn_w[0])     # 取 batch 中的第一个样本

n_layers_vis = len(all_attn_weights)
n_heads_vis = all_attn_weights[0].shape[0]

print(f"  收集到 {n_layers_vis} 层 x {n_heads_vis} 头 = {n_layers_vis * n_heads_vis} 个注意力矩阵")

# ---- 可视化所有层、所有头的注意力 ----
fig, axes = plt.subplots(
    n_layers_vis, n_heads_vis,
    figsize=(3.5 * n_heads_vis, 3.5 * n_layers_vis)
)

for layer_i in range(n_layers_vis):
    for head_j in range(n_heads_vis):
        ax = axes[layer_i][head_j] if n_layers_vis > 1 else axes[head_j]
        attn = all_attn_weights[layer_i][head_j].numpy()

        im = ax.imshow(attn, cmap="Blues", vmin=0, vmax=1.0, aspect="auto")

        # 添加数值标注
        for ii in range(len(vis_words)):
            for jj in range(len(vis_words)):
                val = attn[ii, jj]
                color = "white" if val > 0.5 else "black"
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_xticks(range(len(vis_words)))
        ax.set_yticks(range(len(vis_words)))

        if layer_i == n_layers_vis - 1:
            ax.set_xticklabels(vis_words, fontsize=8, rotation=45)
        else:
            ax.set_xticklabels([])

        if head_j == 0:
            ax.set_yticklabels(vis_words, fontsize=8)
            ax.set_ylabel(f"Layer {layer_i+1}", fontsize=10, fontweight="bold")
        else:
            ax.set_yticklabels([])

        if layer_i == 0:
            ax.set_title(f"Head {head_j+1}", fontsize=10, fontweight="bold")

plt.suptitle("Transformer Encoder 各层各头的注意力权重\n(行=Query, 列=Key, 颜色深=权重大)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_04_encoder_attention_all.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 07_04_encoder_attention_all.png")

# ---- 分析注意力模式 ----
print(f"\n  --- 注意力模式分析 ---")
for layer_i in range(n_layers_vis):
    attn_layer = all_attn_weights[layer_i]  # (n_heads, seq, seq)
    # 计算每层注意力的"集中度"（熵越低 = 越集中）
    avg_attn = attn_layer.mean(dim=0)  # 多头平均
    # 计算每行（每个 query）的熵
    entropy = -(avg_attn * (avg_attn + 1e-9).log()).sum(dim=-1)
    avg_entropy = entropy.mean().item()
    # 计算对角线权重（自注意力程度）
    diag_weight = avg_attn.diagonal().mean().item()
    print(f"  Layer {layer_i+1}: 平均注意力熵={avg_entropy:.3f}, "
          f"平均自注意(对角线)权重={diag_weight:.3f}")

print(f"\n  注意力熵越低 → 注意力越集中（关注少数几个位置）")
print(f"  注意力熵越高 → 注意力越分散（均匀关注所有位置）")


# ====================================================================
# 第8部分：BERT / ViT 级别的参数量估算
# ====================================================================
#
# 实际模型的参数量，帮助你建立对"规模"的直觉。

print("\n\n" + "=" * 60)
print("第8部分：实际模型参数量估算")
print("=" * 60)


def estimate_encoder_params(vocab_size, d_model, n_heads, d_ff, n_layers):
    """
    估算 Transformer Encoder 的参数量。

    参数:
        vocab_size : 词表大小
        d_model    : 模型维度
        n_heads    : 注意力头数（不影响参数量，但影响计算效率）
        d_ff       : FFN 中间层维度
        n_layers   : 层数

    返回:
        params_dict : 各部分参数量的字典
    """
    # 词嵌入
    embedding = vocab_size * d_model

    # 单个 Encoder Block:
    #   多头注意力: 4 * (d_model * d_model + d_model)  (W_Q, W_K, W_V, W_O 各有权重+偏置)
    attn = 4 * (d_model * d_model + d_model)
    #   FFN: 2 * (d_model * d_ff + d_ff 或 d_model)
    ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    #   LayerNorm x2: 2 * (2 * d_model)  (每个 LN 有 gamma 和 beta)
    layer_norm = 2 * (2 * d_model)
    #   单层总计
    per_layer = attn + ffn + layer_norm

    # 最终 LayerNorm
    final_ln = 2 * d_model

    total = embedding + per_layer * n_layers + final_ln

    return {
        "词嵌入": embedding,
        "单层注意力": attn,
        "单层FFN": ffn,
        "单层LayerNorm": layer_norm,
        "单层总计": per_layer,
        "所有层总计": per_layer * n_layers,
        "最终LayerNorm": final_ln,
        "总参数量": total,
    }


# ---- 估算几个著名模型 ----
models = {
    "本节小模型": (1000, 64, 8, 256, 4),
    "Transformer-Base (论文)": (32000, 512, 8, 2048, 6),
    "BERT-Base": (30522, 768, 12, 3072, 12),
    "BERT-Large": (30522, 1024, 16, 4096, 24),
    "ViT-Base/16": (1000, 768, 12, 3072, 12),  # ViT 的 "词表" 是 patch 数
}

print(f"\n  {'模型':<28s}  {'参数量':>12s}  {'约':>10s}")
print(f"  {'-' * 55}")

for name, (vs, dm, nh, dff, nl) in models.items():
    params = estimate_encoder_params(vs, dm, nh, dff, nl)
    total = params["总参数量"]
    if total >= 1e9:
        approx = f"{total / 1e9:.1f}B"
    elif total >= 1e6:
        approx = f"{total / 1e6:.1f}M"
    elif total >= 1e3:
        approx = f"{total / 1e3:.1f}K"
    else:
        approx = str(total)
    print(f"  {name:<28s}  {total:>12,}  {approx:>10s}")

# 详细拆解 BERT-Base
print(f"\n  --- BERT-Base 参数量拆解 ---")
bert_params = estimate_encoder_params(30522, 768, 12, 3072, 12)
for key, val in bert_params.items():
    pct = val / bert_params["总参数量"] * 100 if key != "总参数量" else 100.0
    print(f"    {key:<16s}: {val:>12,}  ({pct:5.1f}%)")


# ====================================================================
# 第9部分：思考题
# ====================================================================

print("\n\n" + "=" * 60)
print("思考题")
print("=" * 60)

questions = [
    {
        "q": "Pre-Norm 和 Post-Norm 有什么区别？为什么现代模型倾向于使用 Pre-Norm？",
        "hint": (
            "提示: Post-Norm 是 x = LayerNorm(x + Sublayer(x))，"
            "Pre-Norm 是 x = x + Sublayer(LayerNorm(x))。"
            "想想深层网络中梯度传播的路径。"
        ),
        "answer": (
            "Pre-Norm 中残差连接直接从输入到输出，形成'梯度高速公路'，\n"
            "    梯度可以不经过 LayerNorm 直接回传到任意一层，训练更稳定。\n"
            "    Post-Norm 的梯度必须穿过 LayerNorm，深层网络容易出现\n"
            "    梯度消失/爆炸，需要精心调整学习率和 warm-up。\n"
            "    代价: Pre-Norm 的最终表示没有经过 LayerNorm，\n"
            "    所以需要在 Encoder 末尾加一个额外的 LayerNorm。"
        ),
    },
    {
        "q": (
            "如果所有 Encoder Block 共享权重（即只有1套参数，重复使用 N 次），\n"
            "   参数量会减少多少？效果会怎样？"
        ),
        "hint": (
            "提示: ALBERT (A Lite BERT) 就是这么做的。\n"
            "想想参数共享对模型容量和训练效率的影响。"
        ),
        "answer": (
            "参数量减少到约 1/N (N=层数)。ALBERT 用这种方式将 BERT 的\n"
            "    参数量减少了约 80%，但性能仅轻微下降。\n"
            "    原因: 虽然参数相同，但每一层接收的输入不同（上一层的输出），\n"
            "    所以每次通过都在做不同的计算。可以理解为一种'迭代精炼'。\n"
            "    缺点: 模型容量降低，在大规模数据上效果不如独立参数。"
        ),
    },
    {
        "q": (
            "为什么要把嵌入向量乘以 sqrt(d_model)？\n"
            "   如果不乘会怎样？"
        ),
        "hint": (
            "提示: Embedding 层的权重通常用均值0、方差约 1/d_model 的分布初始化。\n"
            "位置编码的值域在 [-1, 1] 之间。想想两者直接相加会怎样。"
        ),
        "answer": (
            "Embedding 的权重初始化方差约为 1/d_model，向量的 L2 范数约为 1。\n"
            "    而位置编码的值域是 [-1, 1]，L2 范数约为 sqrt(d_model/2)。\n"
            "    如果不缩放，位置编码会淹没词嵌入的语义信息。\n"
            "    乘以 sqrt(d_model) 后，嵌入向量的范数提升到 sqrt(d_model)，\n"
            "    与位置编码在同一量级，两者的信息都能被保留。"
        ),
    },
    {
        "q": (
            "在 Padding Mask 中，我们只遮挡了 Key 方向的 padding。\n"
            "   需不需要同时遮挡 Query 方向的 padding？为什么？"
        ),
        "hint": (
            "提示: 想想 padding 位置作为 Query 时，\n"
            "计算出的注意力输出会被用在什么地方。"
        ),
        "answer": (
            "严格来说不需要。Padding 位置作为 Query 时确实会计算出注意力输出，\n"
            "    但这些输出在后续的 loss 计算中会被忽略\n"
            "    （分类任务只用 [CLS]，序列标注只看有效位置的 loss）。\n"
            "    遮挡 Key 方向才是关键——防止有效 token '看到' padding。\n"
            "    但某些实现为了计算效率或数值稳定性，也会同时遮挡 Query 方向。"
        ),
    },
    {
        "q": (
            "Transformer Encoder 的输出中，每个位置的向量都包含了整个序列的信息。\n"
            "   这是通过什么机制实现的？如果堆叠更多层，信息传播有什么变化？"
        ),
        "hint": (
            "提示: 自注意力让每个位置可以直接与所有其他位置交互。\n"
            "一层就能看到全部，那多层的意义是什么？"
        ),
        "answer": (
            "自注意力机制让每个位置可以直接 attend 到序列中的所有位置，\n"
            "    所以仅一层就能实现全局信息交互。但一层只能学到'浅层'的关系。\n"
            "    多层堆叠的意义在于：\n"
            "      - 第1层: 每个位置看到其他位置的原始特征\n"
            "      - 第2层: 每个位置看到'已经包含上下文的'特征\n"
            "      - 第N层: 特征经过 N 次'信息融合'，能捕捉更复杂的关系\n"
            "    类似 CNN 中浅层学边缘、深层学语义的逐层抽象过程。"
        ),
    },
]

for i, item in enumerate(questions, 1):
    print(f"\n思考题 {i}：{item['q']}")
    print(f"  {item['hint']}")
    print(f"\n  参考答案：{item['answer']}")


# ====================================================================
# 总结
# ====================================================================
print("\n\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
  1. Encoder Block = LayerNorm + Self-Attention + 残差连接
                   + LayerNorm + FFN + 残差连接
     - Pre-Norm 版本：先 LN 再计算，训练更稳定
     - 每个子层都有残差连接，保证梯度顺畅传播

  2. TransformerEncoder = Embedding + PosEnc + [EncoderBlock x N] + FinalLN
     - 每层权重独立（不共享）
     - 输出的每个位置包含整个序列的上下文信息

  3. Padding Mask 的作用：
     - 将 pad 位置的注意力分数设为 -inf
     - softmax 后权重为 0，完全忽略 padding

  4. 维度变化追踪：
     input_ids (B, S)
       → Embedding (B, S, d_model)
       → + PosEnc  (B, S, d_model)
       → EncoderBlock x N (B, S, d_model)  形状始终不变
       → output (B, S, d_model)

  5. 参数量估算（以 BERT-Base 为例）：
     - 约 110M 参数
     - FFN 占大头（约 2/3），Attention 约 1/3
     - 词嵌入也占相当比例

  下一节预告: 第7章 · 第5节 · Decoder Block 与因果掩码
  （从 Encoder 到 Decoder，理解 GPT 的核心结构）
""")

print("=" * 60)
print("第7章 · 第4节 完成！")
print("=" * 60)
