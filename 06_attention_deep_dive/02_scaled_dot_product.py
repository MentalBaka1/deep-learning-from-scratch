"""
====================================================================
第6章 · 第2节 · 缩放点积注意力
====================================================================

【一句话总结】
Attention(Q,K,V) = softmax(QK^T / √d_k) V —— 这一行公式就是
Transformer 的核心，理解它的每个部分至关重要。

【为什么深度学习需要这个？】
- 这是 "Attention Is All You Need" 论文的核心公式
- 理解为什么要除以 √d_k（不除会怎样？）
- 理解 softmax 在这里的作用（为什么不用其他归一化？）
- 这个公式从 GPT-1 到 GPT-4 一直在用，理解一次受益终身

【核心概念】

1. 点积注意力（Dot-Product Attention）
   - score = Q · K^T
   - Q: (seq_len, d_k), K: (seq_len, d_k) → score: (seq_len, seq_len)
   - score[i][j] = Q 的第 i 个位置对 K 的第 j 个位置的"关注程度"
   - 点积越大 → 越相关 → 注意力权重越高

2. 为什么要缩放（Scale）？
   - 假设 Q 和 K 的每个元素都是均值0、方差1的随机变量
   - 点积 = Σ q_i × k_i，有 d_k 项相加
   - 点积的方差 = d_k（项数越多，方差越大）
   - d_k=512 时，点积可能到 ±30 → softmax 进入饱和区 → 梯度消失
   - 除以 √d_k → 方差变为1 → softmax 输出更"软"→ 梯度更健康

3. Softmax 的角色
   - 将任意实数分数转为概率分布（和为1，非负）
   - 让最大分数的位置获得最多的注意力
   - softmax 是"可微分的 argmax"

4. 温度参数（Temperature）
   - softmax(x/T)
   - T 小 → 更"尖锐"（接近 hard attention）
   - T 大 → 更"均匀"（平等关注所有位置）
   - GPT 生成文本时调的"温度"就是这个！

5. 注意力矩阵的含义
   - attn_weights[i][j] = "位置 i 对位置 j 的关注程度"
   - 每一行是一个概率分布（和为1）
   - 可以可视化为热力图 → 看到模型"在看哪里"

【前置知识】
第6章第1节 - 注意力直觉
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════
# 第一部分：点积注意力 —— 逐步手写实现
# ════════════════════════════════════════════════════════════════════
# 把 Attention(Q,K,V) = softmax(QK^T / √d_k) V 拆成四步：
#   第1步：计算原始分数  scores = Q @ K^T
#   第2步：缩放          scores = scores / √d_k
#   第3步：归一化        weights = softmax(scores)
#   第4步：加权求和      output  = weights @ V
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第一部分：点积注意力 —— 逐步手写实现")
print("=" * 60)

# 假设有一个长度为 4 的序列，每个位置的嵌入维度 d_k = 8
seq_len = 4
d_k = 8

# 随机生成 Q, K, V（实际中它们是输入经过线性变换得到的）
Q = torch.randn(seq_len, d_k)  # (4, 8) —— 每行是一个"查询"向量
K = torch.randn(seq_len, d_k)  # (4, 8) —— 每行是一个"键"向量
V = torch.randn(seq_len, d_k)  # (4, 8) —— 每行是一个"值"向量

print(f"\n输入形状：Q={list(Q.shape)}, K={list(K.shape)}, V={list(V.shape)}")
print(f"序列长度 seq_len={seq_len}, 注意力维度 d_k={d_k}")

# ---- 第1步：计算原始分数 ----
# score[i][j] = Q[i] 和 K[j] 的点积 = 两个向量的相似度
scores = Q @ K.T  # (seq_len, seq_len) = (4, 4)
print(f"\n第1步：原始分数 scores = Q @ K^T")
print(f"  scores 形状: {list(scores.shape)}")
print(f"  scores =\n{scores.numpy().round(3)}")
print(f"  含义：scores[i][j] = 位置 i 对位置 j 的'关注分数'")

# ---- 第2步：缩放 ----
# 为什么要除以 √d_k？后面的实验会详细解释
scale = math.sqrt(d_k)
scaled_scores = scores / scale
print(f"\n第2步：缩放 scores / √d_k = scores / {scale:.2f}")
print(f"  缩放前方差: {scores.var().item():.3f}")
print(f"  缩放后方差: {scaled_scores.var().item():.3f}")
print(f"  缩放让分数的数值范围更合理，softmax 不会饱和")

# ---- 第3步：Softmax 归一化 ----
# 每一行做 softmax → 每一行变成概率分布（和为1）
attn_weights = F.softmax(scaled_scores, dim=-1)  # (4, 4)
print(f"\n第3步：注意力权重 = softmax(scaled_scores)")
print(f"  attn_weights =\n{attn_weights.numpy().round(3)}")
print(f"  每行之和: {attn_weights.sum(dim=-1).numpy().round(6)}")
print(f"  每一行是一个概率分布：位置 i 对所有位置的注意力分配")

# ---- 第4步：加权求和 ----
# 用注意力权重对 V 加权求和 → 输出
output = attn_weights @ V  # (seq_len, d_k) = (4, 8)
print(f"\n第4步：输出 = attn_weights @ V")
print(f"  output 形状: {list(output.shape)}")
print(f"  每个位置的输出 = 所有位置的 V 的加权平均，权重就是注意力")


def scaled_dot_product_attention_manual(Q, K, V, mask=None):
    """
    手写缩放点积注意力 —— 完整版。

    参数:
        Q    : (seq_len, d_k) 查询矩阵
        K    : (seq_len, d_k) 键矩阵
        V    : (seq_len, d_v) 值矩阵
        mask : (seq_len, seq_len) 可选掩码，True 的位置会被屏蔽

    返回:
        output       : (seq_len, d_v) 注意力输出
        attn_weights : (seq_len, seq_len) 注意力权重矩阵
    """
    d_k = Q.size(-1)

    # 第1步 + 第2步：计算缩放点积分数
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    # 可选：应用掩码（因果注意力等场景需要）
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # 第3步：Softmax 归一化
    attn_weights = F.softmax(scores, dim=-1)

    # 第4步：加权求和
    output = attn_weights @ V

    return output, attn_weights


# 验证手写函数与分步结果一致
output_manual, weights_manual = scaled_dot_product_attention_manual(Q, K, V)
assert torch.allclose(output, output_manual, atol=1e-6), "手写实现与分步结果不一致！"
print(f"\n手写函数验证通过！与分步计算结果一致。")


# ════════════════════════════════════════════════════════════════════
# 第二部分：缩放的必要性 —— 不除以 √d_k 会怎样？
# ════════════════════════════════════════════════════════════════════
# 核心实验：对比有无缩放时 softmax 输出的分布。
# d_k 越大 → 没有缩放的点积方差越大 → softmax 越"尖锐" → 梯度消失
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第二部分：缩放的必要性 —— 不除以 √d_k 会怎样？")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
d_k_values = [64, 256, 512]

for col, d_k_test in enumerate(d_k_values):
    # 生成随机 Q, K（均值0，方差1）
    Q_test = torch.randn(32, d_k_test)
    K_test = torch.randn(32, d_k_test)

    # ---- 不缩放 ----
    scores_unscaled = Q_test @ K_test.T  # 方差 ≈ d_k
    weights_unscaled = F.softmax(scores_unscaled, dim=-1)

    # ---- 缩放 ----
    scores_scaled = scores_unscaled / math.sqrt(d_k_test)  # 方差 ≈ 1
    weights_scaled = F.softmax(scores_scaled, dim=-1)

    # 上排：不缩放的注意力权重分布
    ax = axes[0, col]
    ax.hist(weights_unscaled.flatten().numpy(), bins=50,
            color="#e74c3c", alpha=0.7, edgecolor="white", density=True)
    ax.set_title(f"不缩放, d_k={d_k_test}\n分数方差={scores_unscaled.var():.1f}",
                 fontsize=12)
    ax.set_xlabel("注意力权重值")
    ax.set_ylabel("密度")
    ax.set_xlim(0, 0.3)

    # 下排：缩放后的注意力权重分布
    ax = axes[1, col]
    ax.hist(weights_scaled.flatten().numpy(), bins=50,
            color="#2ecc71", alpha=0.7, edgecolor="white", density=True)
    ax.set_title(f"缩放后, d_k={d_k_test}\n分数方差={scores_scaled.var():.1f}",
                 fontsize=12)
    ax.set_xlabel("注意力权重值")
    ax.set_ylabel("密度")
    ax.set_xlim(0, 0.3)

    # 打印数值统计
    max_unscaled = weights_unscaled.max(dim=-1).values.mean().item()
    max_scaled = weights_scaled.max(dim=-1).values.mean().item()
    print(f"\n  d_k = {d_k_test}:")
    print(f"    不缩放 → 分数方差={scores_unscaled.var():.1f}, "
          f"最大权重均值={max_unscaled:.4f}")
    print(f"    缩放后 → 分数方差={scores_scaled.var():.1f}, "
          f"最大权重均值={max_scaled:.4f}")

axes[0, 0].set_ylabel("不缩放\n密度", fontsize=13, fontweight="bold")
axes[1, 0].set_ylabel("缩放后\n密度", fontsize=13, fontweight="bold")
plt.suptitle("缩放的必要性：不缩放时 softmax 接近 one-hot（梯度消失！）",
             fontsize=14)
plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n结论：d_k 越大，不缩放的后果越严重 ——")
print("  softmax 输出几乎变成 one-hot → 梯度几乎为零 → 训练失败")
print("  除以 √d_k 能让方差回到 1 → softmax 输出更均匀 → 梯度健康")


# ════════════════════════════════════════════════════════════════════
# 第三部分：方差分析 —— 实验验证点积方差 = d_k
# ════════════════════════════════════════════════════════════════════
# 理论推导：
#   若 q_i, k_i ~ N(0,1) 且独立，则 q_i * k_i 的方差 = 1
#   点积 = Σ_{i=1}^{d_k} q_i * k_i，有 d_k 项独立求和
#   点积的方差 = d_k * 1 = d_k
#   除以 √d_k 后：方差 = d_k / d_k = 1
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第三部分：方差分析 —— 实验验证点积方差 = d_k")
print("=" * 60)

d_k_range = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_samples = 10000  # 每个 d_k 采样多少次

empirical_variances = []  # 实测方差
theoretical_variances = []  # 理论方差 = d_k
scaled_variances = []  # 缩放后的实测方差

for d_k_test in d_k_range:
    # 随机生成 n_samples 对 (q, k) 向量，计算它们的点积
    q_samples = torch.randn(n_samples, d_k_test)
    k_samples = torch.randn(n_samples, d_k_test)

    # 逐对计算点积：dot[i] = q_samples[i] · k_samples[i]
    dot_products = (q_samples * k_samples).sum(dim=-1)

    # 统计
    var_raw = dot_products.var().item()
    var_scaled = (dot_products / math.sqrt(d_k_test)).var().item()

    empirical_variances.append(var_raw)
    theoretical_variances.append(float(d_k_test))
    scaled_variances.append(var_scaled)

    if d_k_test in [8, 64, 512]:
        print(f"  d_k = {d_k_test:4d} │ "
              f"理论方差 = {d_k_test:6.1f} │ "
              f"实测方差 = {var_raw:8.1f} │ "
              f"缩放后方差 = {var_scaled:.3f}")

# 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：原始方差 vs 理论值
ax1.plot(d_k_range, empirical_variances, "ro-", markersize=8,
         linewidth=2, label="实测方差")
ax1.plot(d_k_range, theoretical_variances, "b--", linewidth=2,
         label="理论方差 = d_k")
ax1.set_xlabel("d_k（注意力维度）", fontsize=12)
ax1.set_ylabel("点积方差", fontsize=12)
ax1.set_title("点积方差随 d_k 线性增长", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xscale("log", base=2)
ax1.set_yscale("log", base=2)

# 右图：缩放后方差稳定在 1
ax2.plot(d_k_range, scaled_variances, "go-", markersize=8,
         linewidth=2, label="缩放后方差")
ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=2,
            label="目标方差 = 1")
ax2.set_xlabel("d_k（注意力维度）", fontsize=12)
ax2.set_ylabel("缩放后点积方差", fontsize=12)
ax2.set_title("除以 √d_k 后方差稳定在 1", fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xscale("log", base=2)
ax2.set_ylim(0.5, 1.5)

plt.tight_layout()
plt.savefig("variance_analysis.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n实验验证了理论推导：")
print("  - 点积的方差 ≈ d_k（完美线性关系）")
print("  - 除以 √d_k 后，方差 ≈ 1（与 d_k 大小无关）")
print("  - 这就是'缩放'的数学原理：控制 softmax 输入的数值范围")


# ════════════════════════════════════════════════════════════════════
# 第四部分：温度实验 —— softmax 的"温度"参数
# ════════════════════════════════════════════════════════════════════
# softmax(x / T) 中的 T 就是温度：
#   T → 0：输出趋近 one-hot（只关注最相关的位置）
#   T = 1：标准 softmax
#   T → ∞：输出趋近均匀分布（平等关注所有位置）
# GPT 生成文本时的"温度"调节的就是这个！
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第四部分：温度实验 —— softmax 的'温度'参数")
print("=" * 60)

# 一组固定的注意力分数
logits = torch.tensor([2.0, 1.0, 0.5, -0.5, -1.0])
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
positions = list(range(len(logits)))
pos_labels = [f"位置{i}" for i in positions]

for idx, T in enumerate(temperatures):
    ax = axes[idx // 3, idx % 3]
    probs = F.softmax(logits / T, dim=-1).numpy()

    bars = ax.bar(positions, probs, color=plt.cm.RdYlGn(probs / probs.max()),
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(pos_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"T = {T}", fontsize=14, fontweight="bold")
    ax.set_ylabel("注意力权重")

    # 在柱状图上标注概率值
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{p:.3f}", ha="center", fontsize=9)

    # 计算熵（衡量分布的"均匀程度"）
    entropy = -(probs * np.log(probs + 1e-10)).sum()
    max_entropy = np.log(len(probs))
    ax.text(0.02, 0.92, f"熵={entropy:.2f}/{max_entropy:.2f}",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("温度对 softmax 输出的影响\n"
             "（原始分数: [2.0, 1.0, 0.5, -0.5, -1.0]）",
             fontsize=14)
plt.tight_layout()
plt.savefig("temperature_experiment.png", dpi=100, bbox_inches="tight")
plt.show()

print("\n温度实验结论：")
print("  T=0.1 → 几乎只关注最高分的位置（hard attention）")
print("  T=1.0 → 标准 softmax，分布有明显倾向但不极端")
print("  T=10  → 接近均匀分布，所有位置被平等关注")
print("\n  GPT 生成文本时：")
print("    低温度 → 保守，选最可能的词 → 确定性强，但无聊")
print("    高温度 → 冒险，给低概率词更多机会 → 多样性好，但可能胡说")
print("  注意：缩放因子 1/√d_k 本质上就是一种固定温度 T=√d_k")


# ════════════════════════════════════════════════════════════════════
# 第五部分：注意力矩阵可视化 —— 一个句子在"看"哪里？
# ════════════════════════════════════════════════════════════════════
# 可视化注意力矩阵是理解 Transformer 的最直观方式。
# attn_weights[i][j] = 位置 i 对位置 j 的关注程度
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第五部分：注意力矩阵可视化")
print("=" * 60)

# 模拟一个中文句子的自注意力
# 假设每个字已经被嵌入为一个向量
sentence = ["我", "喜欢", "深度", "学习", "这门", "课"]
n_tokens = len(sentence)
d_model = 32

# 模拟 Q, K, V（实际中来自嵌入向量的线性变换）
torch.manual_seed(123)
Q_sent = torch.randn(n_tokens, d_model)
K_sent = torch.randn(n_tokens, d_model)
V_sent = torch.randn(n_tokens, d_model)

# 人工制造一些有意义的注意力模式：
# 让"喜欢"更关注"深度学习"（动词关注宾语）
# 让"学习"更关注"深度"（修饰关系）
K_sent[2] = Q_sent[1] * 0.8 + torch.randn(d_model) * 0.3  # "深度"与"喜欢"相似
K_sent[3] = Q_sent[1] * 0.6 + torch.randn(d_model) * 0.4  # "学习"也与"喜欢"有关
K_sent[2] = K_sent[2] * 0.5 + Q_sent[3] * 0.5              # "深度"也与"学习"有关

# 计算注意力
output_sent, attn_sent = scaled_dot_product_attention_manual(Q_sent, K_sent, V_sent)

# 可视化注意力热力图
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(attn_sent.numpy(), cmap="Blues", vmin=0)

# 设置坐标轴标签
ax.set_xticks(range(n_tokens))
ax.set_yticks(range(n_tokens))
ax.set_xticklabels(sentence, fontsize=13)
ax.set_yticklabels(sentence, fontsize=13)
ax.set_xlabel("Key（被关注的位置）", fontsize=12)
ax.set_ylabel("Query（发起关注的位置）", fontsize=12)

# 在每个格子里写上权重值
for i in range(n_tokens):
    for j in range(n_tokens):
        weight = attn_sent[i, j].item()
        color = "white" if weight > 0.3 else "black"
        ax.text(j, i, f"{weight:.2f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.8, label="注意力权重")
ax.set_title("自注意力热力图：每个字在'看'哪些字？\n"
             "（每行之和 = 1.0）", fontsize=13)
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

print(f"\n注意力矩阵形状: {list(attn_sent.shape)}")
print("解读方式：")
print("  - 第 i 行 = 位置 i 的注意力分布")
print("  - 颜色越深 = 注意力权重越大 = 越关注这个位置")
print("  - 每行之和 = 1（概率分布）")


# ════════════════════════════════════════════════════════════════════
# 第六部分：完整 ScaledDotProductAttention 类
# ════════════════════════════════════════════════════════════════════
# 用 PyTorch nn.Module 封装成可复用的模块。
# 支持：batch 维度、多头（外部切分）、因果掩码、dropout。
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第六部分：完整 ScaledDotProductAttention 类")
print("=" * 60)


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块。

    实现公式：Attention(Q, K, V) = softmax(QK^T / √d_k) V

    支持特性：
    - 任意 batch 大小
    - 可选因果掩码（用于自回归生成，如 GPT）
    - 可选 dropout（训练时随机丢弃部分注意力权重，防止过拟合）
    - 可选温度参数（控制注意力分布的尖锐程度）

    参数:
        dropout_p   : float, dropout 概率，默认 0.0
        temperature : float, 温度参数，默认 1.0
    """

    def __init__(self, dropout_p=0.0, temperature=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.temperature = temperature

    def forward(self, Q, K, V, mask=None):
        """
        前向传播。

        参数:
            Q    : (batch, seq_len_q, d_k) 查询
            K    : (batch, seq_len_k, d_k) 键
            V    : (batch, seq_len_k, d_v) 值
            mask : (seq_len_q, seq_len_k) 可选掩码，True 表示需要屏蔽的位置

        返回:
            output       : (batch, seq_len_q, d_v) 注意力输出
            attn_weights : (batch, seq_len_q, seq_len_k) 注意力权重
        """
        d_k = Q.size(-1)

        # 第1步 + 第2步：计算缩放点积分数
        # Q: (B, Lq, dk), K^T: (B, dk, Lk) → scores: (B, Lq, Lk)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (math.sqrt(d_k) * self.temperature)

        # 可选掩码（因果注意力：禁止看到"未来"的信息）
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # 第3步：Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)

        # Dropout（训练时随机屏蔽一些注意力连接）
        attn_weights = self.dropout(attn_weights)

        # 第4步：加权求和
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# ---- 测试：基本功能 ----
print("\n--- 基本功能测试 ---")
batch_size = 2
seq_len = 6
d_k = 16
d_v = 16

attn_module = ScaledDotProductAttention(dropout_p=0.0)
Q_batch = torch.randn(batch_size, seq_len, d_k)
K_batch = torch.randn(batch_size, seq_len, d_k)
V_batch = torch.randn(batch_size, seq_len, d_v)

output_batch, weights_batch = attn_module(Q_batch, K_batch, V_batch)
print(f"  输入: Q={list(Q_batch.shape)}, K={list(K_batch.shape)}, V={list(V_batch.shape)}")
print(f"  输出: output={list(output_batch.shape)}, weights={list(weights_batch.shape)}")
print(f"  权重每行之和（应为1.0）: {weights_batch[0].sum(dim=-1).tolist()}")

# ---- 测试：因果掩码 ----
print("\n--- 因果掩码测试（GPT 式自回归） ---")
# 生成上三角掩码：位置 i 只能看到位置 0..i，看不到 i+1..n
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
print(f"  因果掩码（True=屏蔽）:\n{causal_mask.int().numpy()}")

output_causal, weights_causal = attn_module(Q_batch, K_batch, V_batch, mask=causal_mask)
print(f"\n  应用掩码后的注意力权重（第一个样本，位置3的分布）:")
w = weights_causal[0, 3].tolist()
print(f"    {['%.3f' % x for x in w]}")
print(f"    位置3只看到位置0-3（后面的权重=0）: "
      f"后两位={w[4]:.6f}, {w[5]:.6f}")

# ---- 测试：温度参数 ----
print("\n--- 温度参数测试 ---")
for temp in [0.5, 1.0, 2.0]:
    attn_temp = ScaledDotProductAttention(temperature=temp)
    _, w_temp = attn_temp(Q_batch[:1], K_batch[:1], V_batch[:1])
    max_w = w_temp.max(dim=-1).values.mean().item()
    min_w = w_temp.min(dim=-1).values.mean().item()
    print(f"  T={temp:.1f} → 最大权重均值={max_w:.4f}, 最小权重均值={min_w:.4f}")


# ════════════════════════════════════════════════════════════════════
# 第七部分：与 PyTorch 内置实现对比
# ════════════════════════════════════════════════════════════════════
# PyTorch 2.0+ 提供了 torch.nn.functional.scaled_dot_product_attention
# 它内部使用 FlashAttention 等优化技术，比手写的快很多。
# 我们验证两者在数值上等价。
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("第七部分：与 PyTorch 内置实现对比")
print("=" * 60)

# 准备测试数据
batch_size = 4
seq_len = 32
d_k = 64

Q_cmp = torch.randn(batch_size, seq_len, d_k)
K_cmp = torch.randn(batch_size, seq_len, d_k)
V_cmp = torch.randn(batch_size, seq_len, d_k)

# ---- 我们的实现 ----
our_attn = ScaledDotProductAttention(dropout_p=0.0)
our_output, our_weights = our_attn(Q_cmp, K_cmp, V_cmp)

# ---- PyTorch 内置实现 ----
# 注意：F.scaled_dot_product_attention 不返回权重
# 需要手动使用 _math 实现（sdp_kernel）
with torch.no_grad():
    # PyTorch 的接口
    pt_output = F.scaled_dot_product_attention(Q_cmp, K_cmp, V_cmp)

# 对比结果
output_match = torch.allclose(our_output, pt_output, atol=1e-5)
max_diff = (our_output - pt_output).abs().max().item()

print(f"\n  输入形状: Q={list(Q_cmp.shape)}")
print(f"  我们的输出形状: {list(our_output.shape)}")
print(f"  PyTorch 输出形状: {list(pt_output.shape)}")
print(f"  数值一致性: {output_match}")
print(f"  最大绝对差异: {max_diff:.2e}")

# ---- 带掩码的对比 ----
print("\n--- 带因果掩码的对比 ---")
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
)

our_output_masked, _ = our_attn(Q_cmp, K_cmp, V_cmp, mask=causal_mask)

# PyTorch 内置支持因果掩码的简便写法
with torch.no_grad():
    pt_output_masked = F.scaled_dot_product_attention(
        Q_cmp, K_cmp, V_cmp, is_causal=True
    )

mask_match = torch.allclose(our_output_masked, pt_output_masked, atol=1e-5)
mask_diff = (our_output_masked - pt_output_masked).abs().max().item()
print(f"  因果掩码数值一致性: {mask_match}")
print(f"  因果掩码最大差异: {mask_diff:.2e}")

# ---- 性能说明 ----
print("\n--- 性能对比说明 ---")
print("  我们的实现：")
print("    - 显式计算 (seq_len x seq_len) 注意力矩阵")
print("    - 内存 O(n^2)，seq_len=4096 时需要 ~64MB（float32）")
print("    - 优点：可以返回注意力权重，方便可视化和调试")
print("\n  PyTorch 内置 (FlashAttention 等)：")
print("    - 分块计算，不显式存储完整注意力矩阵")
print("    - 内存 O(n)，速度快 2-4 倍")
print("    - 缺点：不返回注意力权重（为了省内存）")
print("    - 实际训练大模型时必须用这个！")


# ════════════════════════════════════════════════════════════════════
# 第八部分：思考题
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【缩放因子的直觉】
   假设 d_k = 10000（一个很大的注意力维度），不做缩放时
   点积的标准差大约是 √10000 = 100。这意味着 softmax 的
   输入可能分布在 [-300, +300] 的范围。
   (a) softmax([300, 0, 0, 0]) 的输出大约是什么？
   (b) 这时候 softmax 的梯度（对输入的 Jacobian）大约是多少？
   (c) 为什么说这会导致"训练停滞"？
   提示：softmax 输出接近 one-hot 时，梯度 ≈ diag(p) - pp^T ≈ 0。

2. 【如果不用 softmax，用其他归一化呢？】
   有人提出用 L1 归一化（weights = |scores| / Σ|scores|）
   代替 softmax，这样做有什么优缺点？
   提示：考虑负分数的处理、梯度特性、以及"赢者通吃"的效果。
   扩展：实际上 "Linear Attention" 就是用核函数替代 softmax，
   这是当前研究的热点。

3. 【温度与知识蒸馏】
   在知识蒸馏（Knowledge Distillation）中，老师模型的输出
   会先除以一个较大的温度 T（比如 T=20），然后再让学生模型学习。
   (a) 为什么要用高温度？直接学习 softmax 输出不行吗？
   (b) 高温度的 softmax 输出包含了什么"暗知识"（dark knowledge）？
   提示：T=1 时，"猫"=0.99, "虎"=0.005, "狗"=0.004；
         T=20 时，"猫"=0.40, "虎"=0.31, "狗"=0.29 ——
         "虎比狗更像猫"这个信息在高温下才明显。

4. 【注意力的计算复杂度】
   标准缩放点积注意力的时间和空间复杂度都是 O(n^2·d)，
   其中 n 是序列长度，d 是维度。
   (a) 当 n=100K（10万个token）时，注意力矩阵需要多少显存？
   (b) 为什么说这是 Transformer 处理长文本的主要瓶颈？
   (c) FlashAttention 是如何在不改变结果的前提下降低显存的？
   提示：(100000)^2 × 4 bytes = 40 GB。

5. 【动手实验】
   修改本节的 ScaledDotProductAttention 类，实现以下变体：
   (a) 加性注意力（Additive Attention）：
       score(q, k) = v^T · tanh(W_q·q + W_k·k)
       对比它和点积注意力在 d_k 不同时的表现。
   (b) 相对位置编码：
       在注意力分数中加入相对位置偏置 bias[i-j]，
       让模型知道两个 token 之间的距离。
       这是 ALiBi（Attention with Linear Biases）的核心思想。
""")


# ════════════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("总结")
print("=" * 60)
print("""
本节你深入理解了 Transformer 的核心公式：

  Attention(Q, K, V) = softmax(QK^T / √d_k) · V

  1. 点积 QK^T：计算每对位置的相似度分数
  2. 缩放 / √d_k：控制分数的方差为 1，避免 softmax 饱和
  3. Softmax：将分数转为概率分布（和为1，非负）
  4. 加权求和 @V：用注意力权重混合所有位置的信息

  关键实验结论：
  - 不缩放时，d_k 越大 → softmax 越"尖锐" → 梯度消失
  - 点积方差 = d_k，除以 √d_k 后方差稳定为 1
  - 温度控制注意力的"集中度"：低温尖锐，高温均匀
  - 我们的实现与 PyTorch 内置实现数值一致

下一节预告: 第6章 · 第3节 · 多头注意力（Multi-Head Attention）
  —— 一个注意力头只能关注一种模式，多个头可以同时关注
  语法、语义、位置等不同维度的信息。
""")
