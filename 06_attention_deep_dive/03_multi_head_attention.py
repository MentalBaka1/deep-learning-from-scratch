"""
====================================================================
第6章 · 第3节 · 多头注意力
====================================================================

【一句话总结】
多头注意力让模型同时从多个"角度"关注信息——一个头看语法关系，
一个头看语义相似性，一个头看位置关系……

【为什么深度学习需要这个？】
- 单头注意力只有一种"关注方式"，太单一
- 多头注意力 = 多个子空间并行做注意力 → 捕获更丰富的关系
- Transformer 的每一层都用多头注意力（通常 8 或 12 个头）
- GPT-3 用 96 个头，GPT-4 更多

【核心概念】

1. 为什么需要多个头？
   - 类比：看一幅画，你可能同时注意颜色、构图、光影、笔触
   - 单头注意力 = 只用一种方式看
   - 多头注意力 = 用多种方式看，然后综合
   - 不同的头自动学会关注不同类型的关系

2. 多头注意力的实现
   - 输入 X: (batch, seq_len, d_model)
   - 投影到 Q, K, V: 各 (batch, seq_len, d_model)
   - 分成 h 个头: (batch, h, seq_len, d_k)  其中 d_k = d_model / h
   - 每个头独立做 scaled dot-product attention
   - 拼接所有头: (batch, seq_len, d_model)
   - 最后一个线性变换 W_O: (batch, seq_len, d_model)

3. 维度关系
   - d_model = 512, heads = 8 → d_k = d_v = 64
   - 总计算量和单头差不多（因为每个头的维度缩小了）
   - 但表达能力更强（多个子空间）

4. 不同头学到了什么？
   - 实验表明不同的头会自动分工：
     * 某些头关注相邻词
     * 某些头关注句法关系（主语-谓语）
     * 某些头关注长距离依赖
   - 这种自动分工是训练出来的，不是人为设计的

5. 输出投影 W_O 的作用
   - 各头拼接后维度虽然对了，但需要 W_O 来"融合"不同头的信息
   - 没有 W_O，各头的信息就只是简单拼接，没有交互

【前置知识】
第6章第1-2节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
torch.manual_seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 第1部分：从单头到多头 —— 核心思想演示
# ════════════════════════════════════════════════════════════════════
#
# 上一节我们实现了 Scaled Dot-Product Attention:
#   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# 问题来了：Q, K, V 都在同一个向量空间里做注意力，
# 只能学到"一种"关注模式。就好比用一种颜色的眼镜看世界。
#
# 多头注意力的核心思想：
#   1. 把 Q, K, V 各自线性投影到 h 个不同的子空间
#   2. 在每个子空间独立做注意力（每个子空间是一个"头"）
#   3. 拼接所有头的输出，再做一次线性投影
#
# 直觉：多副"有色眼镜"，每副看到不同的关系，最后综合判断

print("=" * 60)
print("第1部分：从单头到多头 —— 核心思想演示")
print("=" * 60)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力（单头版本，复习）。

    参数:
        Q: (batch, seq_len, d_k) 或 (batch, h, seq_len, d_k)
        K: 同 Q
        V: 同 Q
        mask: 可选掩码

    返回:
        output: 注意力加权后的值
        weights: 注意力权重矩阵
    """
    d_k = Q.size(-1)
    # Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V)
    return output, weights


# ---- 单头 vs 多头的直觉演示 ----
# 假设有一个序列，嵌入维度 d_model = 8
batch_size = 1
seq_len = 4
d_model = 8

# 随机输入序列
X = torch.randn(batch_size, seq_len, d_model)
print(f"\n输入 X 形状: {X.shape}  (batch=1, seq_len=4, d_model=8)")

# 方案A：单头注意力 —— 直接用 d_model=8 做注意力
print("\n--- 方案A：单头注意力 ---")
W_Q_single = torch.randn(d_model, d_model)
W_K_single = torch.randn(d_model, d_model)
W_V_single = torch.randn(d_model, d_model)

Q_single = X @ W_Q_single  # (1, 4, 8)
K_single = X @ W_K_single
V_single = X @ W_V_single
out_single, attn_single = scaled_dot_product_attention(Q_single, K_single, V_single)
print(f"  Q/K/V 形状:  {Q_single.shape}")
print(f"  注意力权重:  {attn_single.shape}  → 只有1种关注模式")
print(f"  输出形状:    {out_single.shape}")

# 方案B：多头注意力 —— 把 d_model=8 分成 h=2 个头，每头 d_k=4
print("\n--- 方案B：多头注意力 (h=2, d_k=4) ---")
h = 2
d_k = d_model // h  # 8 // 2 = 4

# 关键操作：把 (batch, seq_len, d_model) 拆成 (batch, h, seq_len, d_k)
Q_multi = Q_single.view(batch_size, seq_len, h, d_k).transpose(1, 2)
K_multi = K_single.view(batch_size, seq_len, h, d_k).transpose(1, 2)
V_multi = V_single.view(batch_size, seq_len, h, d_k).transpose(1, 2)
print(f"  分头后 Q 形状: {Q_multi.shape}  (batch, heads, seq_len, d_k)")

# 每个头独立做注意力
out_multi, attn_multi = scaled_dot_product_attention(Q_multi, K_multi, V_multi)
print(f"  注意力权重:    {attn_multi.shape}  → 2种关注模式！")
print(f"  每头输出形状:  {out_multi.shape}")

# 拼接所有头
out_concat = out_multi.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
print(f"  拼接后形状:    {out_concat.shape}  → 恢复原始维度")

print("\n核心对比:")
print(f"  单头: 1 个 {d_model}x{d_model} 的注意力空间 → 1 种关注模式")
print(f"  多头: {h} 个 {d_k}x{d_k} 的注意力子空间 → {h} 种关注模式")
print(f"  总维度相同! {h} x {d_k} = {h * d_k} = {d_model}")


# ════════════════════════════════════════════════════════════════════
# 第2部分：MultiHeadAttention 完整实现
# ════════════════════════════════════════════════════════════════════
#
# 这是 Transformer 中最核心的模块。
# 论文原文: MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
#   其中 head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
#
# 实现技巧：不需要真的创建 h 个小矩阵 W_Q_i，
#   而是用一个大矩阵 W_Q (d_model x d_model) 做投影，然后 reshape 分头。
#   这样可以用矩阵乘法一步完成所有头的投影，效率更高。

print("\n\n" + "=" * 60)
print("第2部分：MultiHeadAttention 完整实现")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """
    多头注意力的完整实现。

    参数:
        d_model : 模型维度（输入/输出维度）
        n_heads : 注意力头数
        dropout : dropout 比率（默认 0.0）

    前向输入:
        query  : (batch, seq_len_q, d_model)
        key    : (batch, seq_len_k, d_model)
        value  : (batch, seq_len_k, d_model)
        mask   : 可选掩码 (batch, 1, 1, seq_len_k) 或 (batch, 1, seq_len_q, seq_len_k)

    前向输出:
        output : (batch, seq_len_q, d_model)
        attn_weights : (batch, n_heads, seq_len_q, seq_len_k)
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        # 基本检查：d_model 必须能被 n_heads 整除
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除！"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 四个线性变换层（不需要偏置也可以，这里加上偏置更通用）
        # W_Q, W_K, W_V 各自将 d_model → d_model
        # 实际上等价于 h 个 d_model → d_k 的小矩阵拼在一起
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 输出投影 W_O: 把拼接后的多头输出融合回 d_model 维
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 保存注意力权重（用于可视化）
        self.attn_weights = None

    def forward(self, query, key, value, mask=None):
        """
        前向传播的五个步骤:
        1. 线性投影得到 Q, K, V
        2. 分成 h 个头
        3. 每个头独立做 scaled dot-product attention
        4. 拼接所有头
        5. 输出投影 W_O
        """
        batch_size = query.size(0)

        # ---- 步骤1: 线性投影 ----
        # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # ---- 步骤2: 分成 h 个头 ----
        # (batch, seq_len, d_model) → (batch, seq_len, h, d_k) → (batch, h, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ---- 步骤3: Scaled Dot-Product Attention ----
        # scores: (batch, h, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 保存注意力权重（可视化用）
        self.attn_weights = attn_weights.detach()

        # (batch, h, seq_len_q, d_k)
        context = torch.matmul(attn_weights, V)

        # ---- 步骤4: 拼接所有头 ----
        # (batch, h, seq_len, d_k) → (batch, seq_len, h, d_k) → (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # ---- 步骤5: 输出投影 ----
        output = self.W_O(context)

        return output, self.attn_weights


# 实例化并测试
d_model = 64
n_heads = 8
mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
print(f"\n模型结构:")
print(f"  d_model = {d_model}")
print(f"  n_heads = {n_heads}")
print(f"  d_k = d_model / n_heads = {d_model // n_heads}")
print(f"\n模块详情:")
print(mha)

# 测试前向传播
test_input = torch.randn(2, 10, d_model)  # batch=2, seq_len=10
output, weights = mha(test_input, test_input, test_input)
print(f"\n前向传播测试:")
print(f"  输入形状:     {test_input.shape}")
print(f"  输出形状:     {output.shape}")
print(f"  注意力权重:   {weights.shape}")
print(f"  权重每行之和: {weights[0, 0, 0].sum().item():.4f}  (应该 = 1.0)")


# ════════════════════════════════════════════════════════════════════
# 第3部分：维度变换详解 —— 逐步追踪张量形状
# ════════════════════════════════════════════════════════════════════
#
# 这部分是理解多头注意力的关键。
# 我们用一个具体的例子，打印每一步的张量形状。

print("\n\n" + "=" * 60)
print("第3部分：维度变换详解（逐步追踪张量形状）")
print("=" * 60)

# 使用一个具体的配置
batch, seq, d_model, n_heads = 2, 6, 32, 4
d_k = d_model // n_heads

print(f"\n配置: batch={batch}, seq_len={seq}, d_model={d_model}, "
      f"n_heads={n_heads}, d_k={d_k}")
print("-" * 60)

# 创建模型和输入
mha_demo = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
X = torch.randn(batch, seq, d_model)

print(f"\n步骤0: 输入")
print(f"  X 形状:        {X.shape}  (batch, seq_len, d_model)")

# 手动执行每个步骤，打印形状
print(f"\n步骤1: 线性投影 (W_Q, W_K, W_V 各自做一次矩阵乘法)")
Q = mha_demo.W_Q(X)
K = mha_demo.W_K(X)
V = mha_demo.W_V(X)
print(f"  Q = X @ W_Q:   {Q.shape}  (batch, seq_len, d_model)")
print(f"  K = X @ W_K:   {K.shape}  (batch, seq_len, d_model)")
print(f"  V = X @ W_V:   {V.shape}  (batch, seq_len, d_model)")

print(f"\n步骤2: 分成 {n_heads} 个头 (reshape + transpose)")
print(f"  view 操作:")
Q_reshaped = Q.view(batch, seq, n_heads, d_k)
print(f"    Q.view(batch, seq, h, d_k):       {Q_reshaped.shape}")
Q_heads = Q_reshaped.transpose(1, 2)
K_heads = K.view(batch, seq, n_heads, d_k).transpose(1, 2)
V_heads = V.view(batch, seq, n_heads, d_k).transpose(1, 2)
print(f"    Q.transpose(1, 2):                {Q_heads.shape}  (batch, h, seq, d_k)")
print(f"  解读: 第2维从 seq_len 变成了 n_heads，第3维从 n_heads 变成了 seq_len")
print(f"         每个头拥有 seq_len x d_k 的 Q/K/V 矩阵")

print(f"\n步骤3: 计算注意力分数")
scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / math.sqrt(d_k)
print(f"  Q @ K^T:       {Q_heads.shape} @ {K_heads.transpose(-2, -1).shape}")
print(f"  scores 形状:   {scores.shape}  (batch, h, seq_q, seq_k)")
print(f"  除以 sqrt(d_k) = sqrt({d_k}) = {math.sqrt(d_k):.2f}")

print(f"\n步骤4: Softmax 得到注意力权重")
attn_weights = F.softmax(scores, dim=-1)
print(f"  weights 形状:  {attn_weights.shape}  (batch, h, seq_q, seq_k)")
print(f"  每个头独立有一个 {seq}x{seq} 的注意力矩阵！")

print(f"\n步骤5: 加权求和得到每个头的输出")
context = torch.matmul(attn_weights, V_heads)
print(f"  weights @ V:   {attn_weights.shape} @ {V_heads.shape}")
print(f"  context 形状:  {context.shape}  (batch, h, seq_len, d_k)")

print(f"\n步骤6: 拼接所有头 (transpose + reshape)")
concat = context.transpose(1, 2).contiguous()
print(f"  transpose:     {concat.shape}  (batch, seq_len, h, d_k)")
concat = concat.view(batch, seq, d_model)
print(f"  view/reshape:  {concat.shape}  (batch, seq_len, d_model)")
print(f"  拼接: {n_heads} 个头 x d_k={d_k} = d_model={d_model}")

print(f"\n步骤7: 输出投影 W_O")
output = mha_demo.W_O(concat)
print(f"  output = concat @ W_O:  {output.shape}  (batch, seq_len, d_model)")
print(f"  W_O 的作用: 融合不同头的信息，让它们"交流"")

print(f"\n完整流程总结:")
print(f"  输入  → {X.shape}")
print(f"  投影  → Q/K/V: {Q.shape}")
print(f"  分头  → {Q_heads.shape}")
print(f"  注意力 → weights: {attn_weights.shape}, context: {context.shape}")
print(f"  拼接  → {concat.shape}")
print(f"  输出  → {output.shape}")
print(f"  维度始终保持: 输入 d_model = 输出 d_model = {d_model}")


# ════════════════════════════════════════════════════════════════════
# 第4部分：不同头的注意力模式可视化
# ════════════════════════════════════════════════════════════════════
#
# 不同的头学到了不同的关注模式。这里用一个小实验演示：
# 构造一个简单句子，观察不同头的注意力权重分布。
#
# 虽然随机初始化的权重还没有训练，但我们可以用人工设置的权重
# 来模拟"训练后"不同头的分工效果。

print("\n\n" + "=" * 60)
print("第4部分：不同头的注意力模式可视化")
print("=" * 60)

# 模拟一个简短的句子（5个词）
tokens = ["我", "喜欢", "学习", "深度", "学习"]
seq_len = len(tokens)
d_model_vis = 16
n_heads_vis = 4
d_k_vis = d_model_vis // n_heads_vis  # = 4

# 创建模型
mha_vis = MultiHeadAttention(d_model=d_model_vis, n_heads=n_heads_vis)

# 人工设置不同头的权重，模拟训练后的分工效果
# 这里我们直接操纵 Q/K 的投影权重来控制不同头的行为
with torch.no_grad():
    # 让不同头关注不同的"特征维度"
    # 头0: 大权重在前几维 → 可能关注语义相似性
    # 头1: 大权重在后几维 → 可能关注位置关系
    # 头2, 头3: 随机权重 → 混合模式

    # 重新初始化为较小的值
    nn.init.normal_(mha_vis.W_Q.weight, 0, 0.3)
    nn.init.normal_(mha_vis.W_K.weight, 0, 0.3)
    nn.init.normal_(mha_vis.W_V.weight, 0, 0.3)

    # 让头0的 Q/K 权重更集中 → 注意力更"尖锐"
    mha_vis.W_Q.weight[:d_k_vis, :d_k_vis] = torch.eye(d_k_vis) * 2.0
    mha_vis.W_K.weight[:d_k_vis, :d_k_vis] = torch.eye(d_k_vis) * 2.0

    # 让头1的 Q/K 对位置信息更敏感
    for i in range(d_k_vis):
        mha_vis.W_Q.weight[d_k_vis + i, d_k_vis + i] = 3.0
        mha_vis.W_K.weight[d_k_vis + i, d_k_vis + i] = 3.0

# 构造输入嵌入 —— 让语义相似的词嵌入也相似
X_vis = torch.randn(1, seq_len, d_model_vis) * 0.5
with torch.no_grad():
    # "学习"出现两次(index 2 和 4)，让它们的嵌入接近
    X_vis[0, 4] = X_vis[0, 2] + torch.randn(d_model_vis) * 0.1
    # "深度"和"学习"(index 3 和 4)语义相关
    X_vis[0, 3, :d_k_vis] = X_vis[0, 4, :d_k_vis] + torch.randn(d_k_vis) * 0.2

# 前向传播，获取每个头的注意力权重
mha_vis.eval()
with torch.no_grad():
    _, attn_vis = mha_vis(X_vis, X_vis, X_vis)

# 可视化：4个头的注意力矩阵并排显示
fig, axes = plt.subplots(1, n_heads_vis, figsize=(16, 4))

for head_idx in range(n_heads_vis):
    ax = axes[head_idx]
    # 取出这个头的注意力权重矩阵
    attn_matrix = attn_vis[0, head_idx].numpy()  # (seq_len, seq_len)

    im = ax.imshow(attn_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # 设置刻度标签为词语
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(tokens, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(tokens, fontsize=9)

    # 在格子里写权重数值
    for i in range(seq_len):
        for j in range(seq_len):
            val = attn_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(f"Head {head_idx}", fontsize=12, fontweight="bold")
    if head_idx == 0:
        ax.set_ylabel("Query (查询)", fontsize=10)

fig.suptitle("不同头的注意力模式对比\n"
             "（每个头用不同的"视角"关注序列中的关系）",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=axes, shrink=0.8, label="注意力权重")
plt.tight_layout()
plt.show()

print("观察要点:")
print("  - 不同的头产生了不同的注意力分布")
print("  - 有的头可能更关注相邻位置（局部关系）")
print("  - 有的头可能更关注语义相似的词（如两个'学习'）")
print("  - 有的头可能分布更均匀（全局关系）")
print("  - 训练后这种分工会更加明显")


# ════════════════════════════════════════════════════════════════════
# 第5部分：参数量分析 —— 多头 vs 单头
# ════════════════════════════════════════════════════════════════════
#
# 直觉上你可能担心: 多个头 → 参数更多 → 计算更慢？
# 实际上: 总参数量几乎一样！因为每个头的维度按比例缩小了。

print("\n\n" + "=" * 60)
print("第5部分：参数量分析（多头 vs 单头）")
print("=" * 60)


def count_parameters(module):
    """统计一个 nn.Module 的参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_mha_params(d_model, n_heads):
    """
    详细分析多头注意力的参数量。

    参数:
        d_model : 模型维度
        n_heads : 注意力头数
    """
    d_k = d_model // n_heads
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    # 逐层计算参数量
    w_q_params = sum(p.numel() for p in mha.W_Q.parameters())
    w_k_params = sum(p.numel() for p in mha.W_K.parameters())
    w_v_params = sum(p.numel() for p in mha.W_V.parameters())
    w_o_params = sum(p.numel() for p in mha.W_O.parameters())
    total = count_parameters(mha)

    print(f"\n  d_model={d_model}, n_heads={n_heads}, d_k={d_k}")
    print(f"  {'层':15s} {'权重形状':25s} {'参数量':>10s}")
    print(f"  {'-' * 55}")
    print(f"  {'W_Q':15s} {'(' + str(d_model) + ', ' + str(d_model) + ') + bias':25s} {w_q_params:>10,d}")
    print(f"  {'W_K':15s} {'(' + str(d_model) + ', ' + str(d_model) + ') + bias':25s} {w_k_params:>10,d}")
    print(f"  {'W_V':15s} {'(' + str(d_model) + ', ' + str(d_model) + ') + bias':25s} {w_v_params:>10,d}")
    print(f"  {'W_O':15s} {'(' + str(d_model) + ', ' + str(d_model) + ') + bias':25s} {w_o_params:>10,d}")
    print(f"  {'-' * 55}")
    print(f"  {'总计':15s} {'':25s} {total:>10,d}")

    return total


# 对比不同头数的参数量
print("\n比较: 固定 d_model=512, 改变 n_heads")
configs = [
    (512, 1, "单头"),
    (512, 8, "标准 Transformer"),
    (512, 16, "更多头"),
    (512, 64, "很多头"),
]

param_counts = []
for d, h, desc in configs:
    print(f"\n{'─' * 60}")
    print(f"  配置: {desc} (d_model={d}, n_heads={h})")
    total = analyze_mha_params(d, h)
    param_counts.append((desc, h, total))

# 打印对比结论
print(f"\n{'═' * 60}")
print("参数量对比总结:")
print(f"{'═' * 60}")
for desc, h, total in param_counts:
    print(f"  {desc:20s} (h={h:>2d}): {total:>10,d} 参数")

print(f"""
关键发现:
  - 无论用多少个头，参数量完全一样！
  - 原因: W_Q, W_K, W_V, W_O 的形状都是 (d_model, d_model)
  - 与头数无关！头数只影响 reshape 的方式，不影响参数量
  - 计算量也基本相同（矩阵乘法总量不变）
  - 但多头的表达能力更强（多个子空间 > 单个大空间）

公式:
  参数量 = 4 * (d_model * d_model + d_model)
         = 4 * d_model * (d_model + 1)
  对于 d_model=512: 4 * 512 * 513 = {4 * 512 * 513:,d}
""")


# ════════════════════════════════════════════════════════════════════
# 第6部分：与 PyTorch 内置 nn.MultiheadAttention 对比
# ════════════════════════════════════════════════════════════════════
#
# PyTorch 提供了 nn.MultiheadAttention，接口稍有不同：
# - 默认输入格式是 (seq_len, batch, d_model)（注意不是 batch_first！）
# - 可以通过 batch_first=True 改为 (batch, seq_len, d_model)
# - 返回 (output, attn_weights)
#
# 我们来验证我们的实现和官方的行为是否一致。

print("\n\n" + "=" * 60)
print("第6部分：与 PyTorch 内置 nn.MultiheadAttention 对比")
print("=" * 60)

d_model_cmp = 64
n_heads_cmp = 8
seq_len_cmp = 10
batch_size_cmp = 2

# ---- 1. 创建我们的实现 ----
our_mha = MultiHeadAttention(d_model=d_model_cmp, n_heads=n_heads_cmp)

# ---- 2. 创建 PyTorch 内置实现 ----
# batch_first=True 让输入格式和我们一样: (batch, seq_len, d_model)
pt_mha = nn.MultiheadAttention(
    embed_dim=d_model_cmp,
    num_heads=n_heads_cmp,
    batch_first=True  # 重要！否则默认是 (seq_len, batch, d_model)
)

# ---- 3. 对比接口 ----
print("\n接口对比:")
print(f"  {'':20s} {'我们的实现':25s} {'PyTorch 内置':25s}")
print(f"  {'-' * 70}")
print(f"  {'输入格式':20s} {'(batch, seq, d_model)':25s} {'(batch, seq, d_model)*':25s}")
print(f"  {'mask 位置':20s} {'第4个参数':25s} {'key_padding_mask/attn_mask':25s}")
print(f"  {'输出格式':20s} {'(output, weights)':25s} {'(output, weights)':25s}")
print(f"  {'权重形状':20s} {'(B, h, seq, seq)':25s} {'(B, seq, seq)**':25s}")
print(f"\n  * PyTorch 需要 batch_first=True，否则是 (seq, batch, d_model)")
print(f"  ** PyTorch 默认返回平均后的权重；average_attn_weights=False 返回逐头权重")

# ---- 4. 用相同权重测试 ----
# 把我们的权重复制到 PyTorch 内置模型
# PyTorch 的 nn.MultiheadAttention 内部合并了 W_Q, W_K, W_V 为一个大矩阵 in_proj_weight
with torch.no_grad():
    # PyTorch 的 in_proj_weight 是 (3*d_model, d_model)
    # 前 d_model 行 = W_Q, 中间 = W_K, 后面 = W_V
    pt_mha.in_proj_weight[:d_model_cmp] = our_mha.W_Q.weight.clone()
    pt_mha.in_proj_weight[d_model_cmp:2*d_model_cmp] = our_mha.W_K.weight.clone()
    pt_mha.in_proj_weight[2*d_model_cmp:] = our_mha.W_V.weight.clone()

    pt_mha.in_proj_bias[:d_model_cmp] = our_mha.W_Q.bias.clone()
    pt_mha.in_proj_bias[d_model_cmp:2*d_model_cmp] = our_mha.W_K.bias.clone()
    pt_mha.in_proj_bias[2*d_model_cmp:] = our_mha.W_V.bias.clone()

    pt_mha.out_proj.weight.copy_(our_mha.W_O.weight)
    pt_mha.out_proj.bias.copy_(our_mha.W_O.bias)

# 用相同输入测试
X_cmp = torch.randn(batch_size_cmp, seq_len_cmp, d_model_cmp)

our_mha.eval()
pt_mha.eval()

with torch.no_grad():
    our_out, our_weights = our_mha(X_cmp, X_cmp, X_cmp)
    pt_out, pt_weights = pt_mha(X_cmp, X_cmp, X_cmp, average_attn_weights=False)

# 比较输出
output_diff = (our_out - pt_out).abs().max().item()
weight_diff = (our_weights - pt_weights).abs().max().item()

print(f"\n数值对比（使用相同权重和输入）:")
print(f"  输出最大差异:     {output_diff:.2e}")
print(f"  注意力权重差异:   {weight_diff:.2e}")

if output_diff < 1e-5 and weight_diff < 1e-5:
    print(f"  结论: 我们的实现与 PyTorch 内置完全一致！")
else:
    print(f"  存在微小数值差异（浮点精度导致），行为上等价")

# ---- 5. 参数量对比 ----
our_params = count_parameters(our_mha)
pt_params = count_parameters(pt_mha)
print(f"\n参数量对比:")
print(f"  我们的实现: {our_params:,d} 参数")
print(f"  PyTorch 内置: {pt_params:,d} 参数")
print(f"  差异: {abs(our_params - pt_params)} "
      f"({'相同！' if our_params == pt_params else '不同'})")

# ---- 6. PyTorch 实现的关键差异 ----
print(f"""
PyTorch nn.MultiheadAttention 的实现细节:
  1. 内部把 W_Q, W_K, W_V 合并成一个大矩阵 in_proj_weight
     形状 (3*d_model, d_model)，一次矩阵乘法完成三个投影
  2. 支持 key_padding_mask（填充位掩码）和 attn_mask（因果掩码等）
  3. 支持 add_zero_attn（在 K, V 末尾补零，帮助注意力学习"不关注"）
  4. average_attn_weights 参数控制是否对头取平均
  5. 底层使用高度优化的 C++/CUDA 实现，速度更快
""")


# ════════════════════════════════════════════════════════════════════
# 第7部分：综合演示 —— 模拟多头注意力的"分工"
# ════════════════════════════════════════════════════════════════════
#
# 为了更直观地展示不同头的分工，我们构造一个简单任务:
# 给定一个句子，人工设置权重使得:
#   - 头0: 关注相邻位置（局部模式）
#   - 头1: 关注语义相似词（语义模式）
#   - 头2: 关注距离最远的词（长距离依赖）
#   - 头3: 均匀关注所有词（全局平均）

print("\n\n" + "=" * 60)
print("第7部分：综合演示 —— 模拟不同头的分工")
print("=" * 60)

tokens_demo = ["the", "cat", "sat", "on", "the", "mat"]
n_tokens = len(tokens_demo)
d_model_demo = 16
n_heads_demo = 4
d_k_demo = d_model_demo // n_heads_demo

# 构造人工注意力权重来模拟不同头的行为
attn_patterns = torch.zeros(1, n_heads_demo, n_tokens, n_tokens)

# 头0: 局部注意力 —— 每个词主要关注自己和左右邻居
for i in range(n_tokens):
    for j in range(n_tokens):
        dist = abs(i - j)
        if dist == 0:
            attn_patterns[0, 0, i, j] = 3.0
        elif dist == 1:
            attn_patterns[0, 0, i, j] = 1.5
        else:
            attn_patterns[0, 0, i, j] = 0.1
# 归一化
attn_patterns[0, 0] = F.softmax(attn_patterns[0, 0], dim=-1)

# 头1: 语义相似性 —— "the"关注"the"，"cat"关注"mat"（都是名词）
semantic_scores = torch.zeros(n_tokens, n_tokens)
# "the"(0) 和 "the"(4) 互相关注
semantic_scores[0, 4] = 3.0; semantic_scores[4, 0] = 3.0
# "cat"(1) 和 "mat"(5) 互相关注（名词）
semantic_scores[1, 5] = 3.0; semantic_scores[5, 1] = 3.0
# "sat"(2) 和 "on"(3) 关注（动词-介词）
semantic_scores[2, 3] = 2.0; semantic_scores[3, 2] = 2.0
# 自注意力
for i in range(n_tokens):
    semantic_scores[i, i] = 1.0
attn_patterns[0, 1] = F.softmax(semantic_scores, dim=-1)

# 头2: 长距离依赖 —— 越远关注越多
for i in range(n_tokens):
    for j in range(n_tokens):
        dist = abs(i - j)
        attn_patterns[0, 2, i, j] = dist * 0.5 + 0.1
attn_patterns[0, 2] = F.softmax(attn_patterns[0, 2], dim=-1)

# 头3: 均匀注意力 —— 全局平均
attn_patterns[0, 3] = 1.0 / n_tokens

# 可视化
head_names = [
    "Head 0: 局部关注\n(关注相邻词)",
    "Head 1: 语义关注\n(关注语义相似词)",
    "Head 2: 长距离关注\n(关注远处的词)",
    "Head 3: 全局均匀\n(平等关注所有词)"
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for h_idx in range(n_heads_demo):
    ax = axes[h_idx]
    attn = attn_patterns[0, h_idx].numpy()

    im = ax.imshow(attn, cmap="Blues", vmin=0, aspect="equal")

    ax.set_xticks(range(n_tokens))
    ax.set_yticks(range(n_tokens))
    ax.set_xticklabels(tokens_demo, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(tokens_demo, fontsize=9)

    for i in range(n_tokens):
        for j in range(n_tokens):
            val = attn[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title(head_names[h_idx], fontsize=10, fontweight="bold")
    if h_idx == 0:
        ax.set_ylabel("Query (查询词)", fontsize=10)

fig.suptitle('不同头的"分工"模拟\n'
             '（真实模型中这些模式是自动学到的，这里是人工设置用于教学）',
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=axes, shrink=0.8, label="注意力权重")
plt.tight_layout()
plt.show()

print("""
分工总结:
  Head 0 (局部):    每个词主要关注自己和邻居 → 捕捉 n-gram 特征
  Head 1 (语义):    "the"↔"the", "cat"↔"mat" → 捕捉语义关系
  Head 2 (长距离):  关注距离远的词 → 捕捉长距离依赖
  Head 3 (全局):    均匀分配注意力 → 全局语境建模

  综合所有头的信息（通过 W_O 融合），模型就能同时建模多种关系！
  这就是多头注意力比单头注意力强大的原因。
""")


# ════════════════════════════════════════════════════════════════════
# 第8部分：完整总结与思考题
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了多头注意力的核心知识:

  1. 为什么需要多头:
     - 单头只有一种关注模式，多头 = 多种关注模式并行
     - 不同头自动学会关注不同类型的关系（语法/语义/位置...）

  2. 实现要点:
     - W_Q, W_K, W_V: 各 (d_model, d_model) 的线性投影
     - 分头: view + transpose → (batch, n_heads, seq_len, d_k)
     - 每个头独立做 scaled dot-product attention
     - 拼接: transpose + view → (batch, seq_len, d_model)
     - W_O: 输出投影，融合不同头的信息

  3. 参数量:
     - 总参数 = 4 * d_model * (d_model + 1)
     - 与头数无关！头数只影响 reshape 方式
     - 多头不增加参数量，但增强了表达能力

  4. PyTorch 内置:
     - nn.MultiheadAttention: 合并 W_Q/W_K/W_V 为 in_proj_weight
     - 注意 batch_first 参数
     - 底层有高度优化的实现

下一节将学习完整的 Transformer Encoder 层！
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【头数选择】
   d_model = 768 时，用 8 个头和 12 个头有什么区别？
   d_k 分别是多少？为什么 BERT-base 选择了 12 个头？
   提示: 768 / 8 = 96, 768 / 12 = 64。
   更多头 = 更多子空间 = 更丰富的关注模式，
   但每个头的维度更小 = 每个子空间的表达能力更弱。
   要在"子空间数量"和"子空间质量"之间取得平衡。

2. 【去掉输出投影 W_O】
   如果去掉 W_O，直接把拼接后的结果作为输出，会怎样？
   提示: 没有 W_O，各头的信息只是简单拼接，没有"融合"。
   就好比多个专家各自给出意见，但没有人来综合判断。
   实验表明去掉 W_O 会导致性能明显下降。

3. 【参数量计算】
   GPT-3 (d_model=12288, n_heads=96) 的多头注意力层有多少参数？
   提示: 4 * 12288 * (12288 + 1) = 604,012,032 ≈ 6 亿参数。
   注意: 这只是一层的注意力！GPT-3 有 96 层。
   这也说明了为什么大模型需要大量 GPU 内存。

4. 【Multi-Query Attention (MQA)】
   GQA 和 MQA 是什么？为什么现代模型倾向于用它们？
   提示: MQA 让所有头共享同一组 K, V，只有 Q 分头。
   参数量从 4*d^2 降到 2*d^2 + 2*d*(d/h)。
   GQA 是折中方案: 把头分成若干组，每组共享 K, V。

5. 【因果掩码】
   在 GPT 等自回归模型中，多头注意力需要加因果掩码
   (causal mask)，防止"看到未来"。
   请思考: 掩码应该是什么形状？怎样和多头注意力配合？
   提示: 掩码形状 (1, 1, seq_len, seq_len)，
   上三角为 0（被屏蔽），下三角和对角线为 1（可见）。
   广播机制让它自动适用于所有 batch 和所有头。
""")

print("下一节预告: 第6章 · 第4节 · 完整 Transformer Encoder 层")
