"""
====================================================================
第6章 · 第1节 · 注意力的直觉与本质
====================================================================

【一句话总结】
注意力机制的本质是"加权信息检索"——给定一个查询(Query)，
从一组键值对(Key-Value)中找到最相关的信息并聚合。

【为什么深度学习需要这个？】
- 注意力是 Transformer 的灵魂，没有它就没有 GPT、BERT、LLaMA
- 它解决了 RNN 的两个致命问题：无法并行 + 长距离依赖
- 理解 QKV 的本质，是理解所有现代大模型的前提

【核心概念】

1. 注意力的信息检索类比
   - 想象你在图书馆找一本书：
     * Query（查询）：你想找的内容描述——"关于深度学习的入门书"
     * Key（键）：每本书的标题/标签——"机器学习基础"、"深度学习导论"...
     * Value（值）：书的实际内容
   - 步骤：拿 Query 和每个 Key 比较相关度 → 相关度高的 Value 权重大 → 加权求和得到结果

2. 从数据库查询到神经网络
   - 传统数据库：SELECT value WHERE key = query（精确匹配，0或1）
   - 注意力机制：对所有 key 计算"软匹配"分数 → softmax 归一化 → 加权求和
   - Soft Attention：所有位置都参与，权重连续（可微分，可训练）
   - Hard Attention：只选最相关的位置（不可微，需要强化学习）

3. QKV 的本质
   - Q, K, V 不是三个独立的东西，而是同一个输入的三种"视角"
   - 通过不同的线性变换 W_Q, W_K, W_V 投影得到
   - Q 和 K 用来计算"相关度"，V 是真正要聚合的信息
   - 为什么 Q 和 K 分开？因为"我在找什么"和"我能提供什么"是不同的视角

4. 注意力分数的计算
   - score(Q, K) = Q · K^T（点积，衡量相似度）
   - 权重 = softmax(score)（归一化为概率分布）
   - 输出 = 权重 × V（加权信息聚合）

5. 注意力的三种类型（预告）
   - 自注意力（Self-Attention）：Q, K, V 来自同一序列
   - 交叉注意力（Cross-Attention）：Q 来自一个序列，K, V 来自另一个
   - 因果注意力（Causal Attention）：只能看到之前的位置（GPT 用）

【前置知识】
第4章第3节 - Seq2Seq+Attention，第5章 - PyTorch基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
torch.manual_seed(42)  # 固定随机种子，保证结果可复现
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════
# 第1部分：信息检索类比 —— 图书馆找书
# ════════════════════════════════════════════════════════════════════
#
# 注意力机制本质上就是"信息检索"：
#   1. 你有一个查询（Query）："我想找深度学习的入门书"
#   2. 图书馆有很多书，每本书有标签（Key）和内容（Value）
#   3. 拿 Query 和每个 Key 比较 → 得到相关度分数
#   4. 用分数加权 Value → 得到最终结果
#
# 我们用一个简单的字符串匹配来模拟这个过程。
#

print("=" * 60)
print("第1部分：信息检索类比 —— 图书馆找书")
print("=" * 60)

# 模拟图书馆：每本书有 Key（标签关键词）和 Value（内容摘要）
library = {
    "keys": [
        "深度学习 入门 神经网络",
        "机器学习 统计 回归",
        "计算机视觉 图像 CNN",
        "自然语言处理 文本 RNN",
        "强化学习 决策 智能体",
    ],
    "values": [
        "本书系统介绍深度学习基础，涵盖前馈网络、反向传播等核心概念",
        "本书讲解经典机器学习算法，包括线性回归、决策树、SVM等",
        "本书聚焦计算机视觉，详解卷积神经网络及其在图像识别中的应用",
        "本书介绍自然语言处理技术，从词向量到序列模型再到注意力机制",
        "本书探讨强化学习理论，包括Q-learning、策略梯度等方法",
    ],
}


def compute_keyword_score(query, key):
    """
    计算查询与键之间的关键词匹配分数。

    简单策略：统计 query 中的词在 key 中出现了多少个。

    参数:
        query : 查询字符串
        key   : 键字符串（书的标签）

    返回:
        score : 匹配分数（命中词数）
    """
    query_words = set(query.split())
    key_words = set(key.split())
    return len(query_words & key_words)  # 交集大小


def book_search_attention(query, library):
    """
    用注意力的方式检索图书馆。

    步骤：
      1. 计算 Query 与每个 Key 的匹配分数
      2. Softmax 归一化为权重
      3. 打印加权结果（哪些书最相关）

    参数:
        query   : 查询字符串
        library : 包含 keys 和 values 的字典
    """
    keys = library["keys"]
    values = library["values"]

    # 第一步：计算原始匹配分数
    scores = [compute_keyword_score(query, k) for k in keys]
    scores_tensor = torch.tensor(scores, dtype=torch.float32)

    # 第二步：Softmax 归一化 → 注意力权重
    weights = F.softmax(scores_tensor, dim=0)

    print(f"\n查询 (Query): \"{query}\"")
    print("-" * 50)
    for i, (key, val, s, w) in enumerate(zip(keys, values, scores, weights)):
        marker = " <<<" if w == weights.max() else ""
        print(f"  书{i+1} Key: \"{key}\"")
        print(f"       匹配分数={s}, 注意力权重={w:.3f}{marker}")
    print(f"\n注意力权重分布: {[f'{w:.3f}' for w in weights.tolist()]}")
    print(f"权重之和 = {weights.sum():.3f} (softmax 保证归一化)")


# 测试不同的查询
book_search_attention("深度学习 入门 基础", library)
book_search_attention("图像 CNN 视觉", library)
book_search_attention("文本 RNN 注意力", library)

print("\n核心洞察：")
print("  注意力 = 计算相关度 + softmax 归一化 + 加权聚合")
print("  神经网络中，'关键词匹配'被替换为'向量点积'")


# ════════════════════════════════════════════════════════════════════
# 第2部分：Hard vs Soft Attention —— 硬注意力 vs 软注意力
# ════════════════════════════════════════════════════════════════════
#
# 两种注意力策略：
#   - Hard Attention（硬注意力）：argmax 只选最相关的一个位置
#     * 优点：计算高效（只取一个）
#     * 缺点：不可微分，无法用反向传播训练，需要强化学习（REINFORCE）
#
#   - Soft Attention（软注意力）：softmax 分配连续权重给所有位置
#     * 优点：可微分，可以端到端训练
#     * 缺点：计算量大（所有位置都参与）
#     * 这是目前主流方法（Transformer 用的就是 Soft Attention）
#

print("\n\n" + "=" * 60)
print("第2部分：Hard vs Soft Attention 对比")
print("=" * 60)

# 模拟一个序列，5个位置各有一个特征向量
seq_len = 5
d_model = 4  # 特征维度

# 5个位置的 Value 向量
values_demo = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # 位置0
    [0.0, 1.0, 0.0, 0.0],  # 位置1
    [0.0, 0.0, 1.0, 0.0],  # 位置2 ← 假设这个最相关
    [0.0, 0.0, 0.0, 1.0],  # 位置3
    [0.5, 0.5, 0.0, 0.0],  # 位置4
])

# 原始注意力分数（假设已经计算好）
raw_scores = torch.tensor([0.5, 1.2, 3.8, 0.9, 1.0])

# ---- Hard Attention: argmax ----
hard_idx = torch.argmax(raw_scores)
hard_weights = torch.zeros(seq_len)
hard_weights[hard_idx] = 1.0  # 只有一个位置权重为1
hard_output = hard_weights @ values_demo  # 直接取该位置的 Value

# ---- Soft Attention: softmax ----
soft_weights = F.softmax(raw_scores, dim=0)
soft_output = soft_weights @ values_demo  # 加权求和所有位置的 Value

print(f"\n原始分数:     {raw_scores.tolist()}")
print(f"\nHard Attention:")
print(f"  权重 (argmax): {hard_weights.tolist()}")
print(f"  输出:          {hard_output.tolist()}")
print(f"  特点: 只取位置{hard_idx.item()}的 Value，信息丢失严重")
print(f"\nSoft Attention:")
print(f"  权重 (softmax): {[f'{w:.3f}' for w in soft_weights.tolist()]}")
print(f"  输出:           {[f'{v:.3f}' for v in soft_output.tolist()]}")
print(f"  特点: 所有位置都参与，权重连续可微")

# ---- 可视化对比 ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左：原始分数
axes[0].bar(range(seq_len), raw_scores.numpy(), color="steelblue", alpha=0.8)
axes[0].set_title("原始注意力分数", fontsize=12)
axes[0].set_xlabel("位置")
axes[0].set_ylabel("分数")
axes[0].set_xticks(range(seq_len))

# 中：Hard Attention 权重
colors_hard = ["#e74c3c" if i == hard_idx else "#cccccc" for i in range(seq_len)]
axes[1].bar(range(seq_len), hard_weights.numpy(), color=colors_hard, alpha=0.9)
axes[1].set_title("Hard Attention (argmax)\n只选最大的一个", fontsize=12)
axes[1].set_xlabel("位置")
axes[1].set_ylabel("权重")
axes[1].set_xticks(range(seq_len))
axes[1].set_ylim(0, 1.1)

# 右：Soft Attention 权重
axes[2].bar(range(seq_len), soft_weights.numpy(), color="#2ecc71", alpha=0.8)
axes[2].set_title("Soft Attention (softmax)\n所有位置都参与", fontsize=12)
axes[2].set_xlabel("位置")
axes[2].set_ylabel("权重")
axes[2].set_xticks(range(seq_len))
axes[2].set_ylim(0, 1.1)

plt.suptitle("Hard vs Soft Attention：硬选择 vs 软加权",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("06_01_hard_vs_soft.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 06_01_hard_vs_soft.png")

print("\n关键区别：")
print("  Hard: argmax 不可微 → 不能反向传播 → 需要 REINFORCE 等策略")
print("  Soft: softmax 可微 → 可以端到端训练 → Transformer 的选择")


# ════════════════════════════════════════════════════════════════════
# 第3部分：QKV 手动演示 —— 一步步计算注意力
# ════════════════════════════════════════════════════════════════════
#
# 注意力的核心公式：
#   Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
#
# 步骤拆解：
#   1. Q · K^T → 计算每对 (query, key) 的点积 → 相似度矩阵
#   2. / √d_k → 缩放（防止点积过大导致 softmax 饱和）
#   3. softmax → 归一化为概率分布
#   4. × V → 加权聚合 Value 信息
#

print("\n\n" + "=" * 60)
print("第3部分：QKV 手动演示")
print("=" * 60)

# 模拟输入：3个词的序列，每个词 d_model=4 维
# 假设这是句子 "猫 坐 垫" 的嵌入表示
seq_len = 3
d_k = 4  # Q, K 的维度（也是 V 的维度，简化版）

# 假设已经通过线性变换得到 Q, K, V
Q = torch.tensor([
    [1.0, 0.5, 0.0, 0.2],   # "猫" 的 Query：我在找什么？
    [0.2, 0.1, 0.8, 0.5],   # "坐" 的 Query
    [0.5, 0.3, 0.1, 0.9],   # "垫" 的 Query
])

K = torch.tensor([
    [0.9, 0.6, 0.1, 0.3],   # "猫" 的 Key：我能提供什么？
    [0.1, 0.2, 0.7, 0.4],   # "坐" 的 Key
    [0.4, 0.2, 0.0, 0.8],   # "垫" 的 Key
])

V = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],   # "猫" 的 Value：我的信息内容
    [0.0, 1.0, 0.0, 0.0],   # "坐" 的 Value
    [0.0, 0.0, 1.0, 0.0],   # "垫" 的 Value
])

print(f"\n输入形状: Q={list(Q.shape)}, K={list(K.shape)}, V={list(V.shape)}")
print(f"Q (各词的查询向量):\n{Q}")
print(f"K (各词的键向量):\n{K}")
print(f"V (各词的值向量):\n{V}")

# 第一步：Q · K^T → 注意力分数矩阵
print(f"\n--- 第1步: 计算 Q · K^T (点积相似度) ---")
scores = Q @ K.T  # (3, 3)
print(f"scores = Q @ K^T =\n{scores}")
print(f"scores[i][j] = 第i个词的 Query 与第j个词的 Key 的点积")
print(f"例: scores[0][0] = Q[0]·K[0] = {(Q[0] * K[0]).sum():.2f} (猫查询自己)")

# 第二步：缩放（除以 √d_k）
print(f"\n--- 第2步: 缩放 scores / √d_k ---")
scale = d_k ** 0.5
scaled_scores = scores / scale
print(f"d_k = {d_k}, √d_k = {scale:.2f}")
print(f"缩放后的分数:\n{scaled_scores}")
print(f"为什么缩放？防止 d_k 很大时点积太大 → softmax 趋近 one-hot → 梯度消失")

# 第三步：Softmax 归一化
print(f"\n--- 第3步: Softmax 归一化 (每行独立) ---")
attn_weights = F.softmax(scaled_scores, dim=-1)  # 每行归一化
print(f"注意力权重矩阵:\n{attn_weights}")
print(f"每行之和: {attn_weights.sum(dim=-1).tolist()} (都是1.0)")
print(f"含义: attn_weights[i][j] = 词i对词j分配的注意力权重")

# 第四步：加权聚合 Value
print(f"\n--- 第4步: 注意力输出 = 权重 × V ---")
output = attn_weights @ V  # (3, 4)
print(f"输出:\n{output}")
print(f"每个词的输出 = 所有词的 Value 的加权组合")
print(f"例: 输出[0] = {attn_weights[0][0]:.3f}*V[猫] + "
      f"{attn_weights[0][1]:.3f}*V[坐] + {attn_weights[0][2]:.3f}*V[垫]")


# ════════════════════════════════════════════════════════════════════
# 第4部分：注意力权重可视化 —— 热力图
# ════════════════════════════════════════════════════════════════════
#
# 注意力权重矩阵可以画成热力图，直观展示"谁在关注谁"。
# 这是分析 Transformer 行为的核心工具。
#

print("\n\n" + "=" * 60)
print("第4部分：注意力权重可视化 (热力图)")
print("=" * 60)

# 用一个更长的句子来演示
words = ["我", "喜欢", "深度", "学习", "因为", "它", "很", "强大"]
n_words = len(words)
d_model_demo = 16

# 模拟一组嵌入向量（实际应该是词嵌入 + 位置编码）
torch.manual_seed(42)
embeddings = torch.randn(n_words, d_model_demo)

# 模拟 Q, K, V 线性变换（小维度演示）
W_Q = torch.randn(d_model_demo, d_model_demo) * 0.3
W_K = torch.randn(d_model_demo, d_model_demo) * 0.3
W_V = torch.randn(d_model_demo, d_model_demo) * 0.3

Q_demo = embeddings @ W_Q  # (8, 16)
K_demo = embeddings @ W_K  # (8, 16)
V_demo = embeddings @ W_V  # (8, 16)

# 计算注意力权重
scores_demo = Q_demo @ K_demo.T / (d_model_demo ** 0.5)
attn_weights_demo = F.softmax(scores_demo, dim=-1)

# 绘制热力图
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(attn_weights_demo.detach().numpy(), cmap="Blues", aspect="auto")

# 添加数值标注
for i in range(n_words):
    for j in range(n_words):
        val = attn_weights_demo[i, j].item()
        color = "white" if val > 0.3 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color)

ax.set_xticks(range(n_words))
ax.set_yticks(range(n_words))
ax.set_xticklabels(words, fontsize=11)
ax.set_yticklabels(words, fontsize=11)
ax.set_xlabel("Key (被关注的词)", fontsize=12)
ax.set_ylabel("Query (发出关注的词)", fontsize=12)
ax.set_title("自注意力权重热力图\n每行 = 该词对所有词的注意力分布",
             fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax, label="注意力权重")
plt.tight_layout()
plt.savefig("06_01_attention_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 06_01_attention_heatmap.png")

print(f"\n热力图解读：")
print(f"  - 每一行是一个 Query 词的注意力分布（和为1）")
print(f"  - 颜色越深 = 注意力权重越大 = 越关注那个词")
print(f"  - 对角线较亮 = 每个词多少会关注自己（但不是全部注意力）")


# ════════════════════════════════════════════════════════════════════
# 第5部分：为什么需要学习 Q, K, V？—— 学习投影 vs 原始输入
# ════════════════════════════════════════════════════════════════════
#
# 核心问题：为什么不直接用原始嵌入做注意力，而要通过 W_Q, W_K, W_V 投影？
#
# 答案：原始嵌入只有一个"视角"，而不同的线性投影可以让模型
# 从不同角度理解"我在找什么"(Q) 和"我能提供什么"(K)。
#
# 实验：对比 (1) 直接用原始嵌入 vs (2) 通过可学习投影
# 在一个简单任务上，看哪个注意力模式更合理。
#

print("\n\n" + "=" * 60)
print("第5部分：为什么需要学习 Q, K, V？")
print("=" * 60)

# 构造一个有明确语义结构的例子
# 句子："小猫 在 垫子 上 睡觉"
# 语义上，"小猫"应该关注"睡觉"（主谓关系），"垫子"应该关注"上"（位置关系）
sentence = ["小猫", "在", "垫子", "上", "睡觉"]
n = len(sentence)
d = 8

# 模拟嵌入：让语义相关的词有相似的嵌入
torch.manual_seed(123)
# 精心设计的嵌入，使得相关词之间有一定相似度
embed_matrix = torch.randn(n, d)
# 让 "小猫"(0) 和 "睡觉"(4) 在某些维度相似
embed_matrix[4, :4] = embed_matrix[0, :4] * 0.8 + torch.randn(4) * 0.2
# 让 "垫子"(2) 和 "上"(3) 在某些维度相似
embed_matrix[3, 4:] = embed_matrix[2, 4:] * 0.8 + torch.randn(4) * 0.2

# ---- 方法1：直接用原始嵌入做注意力（不投影）----
scores_raw = embed_matrix @ embed_matrix.T / (d ** 0.5)
attn_raw = F.softmax(scores_raw, dim=-1)

# ---- 方法2：通过可学习的 W_Q, W_K 投影 ----
# 使用 nn.Linear（包含可学习参数），模拟训练后的投影矩阵
torch.manual_seed(77)
proj_q = nn.Linear(d, d, bias=False)
proj_k = nn.Linear(d, d, bias=False)

# 手动调整投影权重，模拟"训练后"的效果：
# 让投影后的 Q[0](小猫) 和 K[4](睡觉) 更相似
with torch.no_grad():
    # 让 W_Q 强调前4维（与主语相关的特征）
    proj_q.weight.copy_(torch.eye(d) + torch.randn(d, d) * 0.1)
    # 让 W_K 强调后4维（与语义角色相关的特征）并旋转空间
    w_k = torch.eye(d) + torch.randn(d, d) * 0.1
    w_k[:4, 4:] = torch.randn(4, 4) * 0.5  # 创建交叉维度的连接
    proj_k.weight.copy_(w_k)

Q_proj = proj_q(embed_matrix)
K_proj = proj_k(embed_matrix)
scores_proj = Q_proj @ K_proj.T / (d ** 0.5)
attn_proj = F.softmax(scores_proj, dim=-1)

# ---- 可视化对比 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, attn, title in zip(
    axes,
    [attn_raw.detach().numpy(), attn_proj.detach().numpy()],
    ["直接用原始嵌入 (无投影)\nQ=K=原始嵌入",
     "通过 W_Q, W_K 投影后\nQ=XW_Q, K=XW_K"]
):
    im = ax.imshow(attn, cmap="Oranges", aspect="auto", vmin=0, vmax=0.5)
    for i in range(n):
        for j in range(n):
            val = attn[i, j]
            color = "white" if val > 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sentence, fontsize=11)
    ax.set_yticklabels(sentence, fontsize=11)
    ax.set_xlabel("Key", fontsize=11)
    ax.set_ylabel("Query", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle("原始嵌入 vs 学习投影：为什么 QKV 需要独立的权重矩阵？",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("06_01_raw_vs_projected.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 06_01_raw_vs_projected.png")

print("\n关键结论：")
print("  1. 原始嵌入做注意力 → 注意力模式由嵌入相似度决定，不可控")
print("  2. 学习投影做注意力 → 模型可以学会从不同视角看同一个词")
print("  3. W_Q 决定'我在找什么'，W_K 决定'我能提供什么'")
print("  4. 分开投影让模型能学到不对称的关系（A 关注 B ≠ B 关注 A）")


# ════════════════════════════════════════════════════════════════════
# 第6部分：注意力的效果 —— 加权求和 vs 简单平均
# ════════════════════════════════════════════════════════════════════
#
# 用一个简单任务直观展示注意力的价值：
# 任务：给定一个序列，输出与特定 query 最相关的信息
#
# 对比三种策略：
#   1. 简单平均：所有位置均匀聚合（无注意力）
#   2. 注意力聚合：根据相关度加权聚合
#   3. 可学习注意力：通过训练学到最优的 QKV 投影
#

print("\n\n" + "=" * 60)
print("第6部分：注意力的效果 —— 简单序列任务")
print("=" * 60)

# 任务设计：
# 输入序列包含若干"信号"和"噪声"，目标是提取信号并忽略噪声
# 信号：值较大的向量；噪声：值较小的随机向量
# Query 指定我们要找的信号类型

torch.manual_seed(42)

# 创建一个包含信号和噪声的序列
print("\n--- 任务：从噪声中提取信号 ---")
signal_vec = torch.tensor([5.0, 5.0, 0.0, 0.0])    # 信号在前两维
noise_vecs = torch.randn(4, 4) * 0.5                # 4个噪声向量
sequence = torch.cat([signal_vec.unsqueeze(0), noise_vecs], dim=0)  # (5, 4)

# Query：我们要找前两维值大的向量
query = torch.tensor([[3.0, 3.0, 0.0, 0.0]])  # (1, 4) 与信号方向一致

print(f"序列 (5个位置, 4维):")
for i, v in enumerate(sequence):
    tag = " <-- 信号" if i == 0 else "     噪声"
    print(f"  位置{i}: {v.tolist()}{tag}")
print(f"查询 Query: {query.tolist()[0]}")

# 方法1：简单平均（无注意力）
avg_output = sequence.mean(dim=0)

# 方法2：注意力聚合
attn_scores = query @ sequence.T / (4 ** 0.5)  # (1, 5)
attn_w = F.softmax(attn_scores, dim=-1)         # (1, 5)
attn_output = (attn_w @ sequence).squeeze(0)     # (4,)

print(f"\n方法1 - 简单平均 (无注意力):")
print(f"  输出: {[f'{v:.3f}' for v in avg_output.tolist()]}")
print(f"  与信号的余弦相似度: "
      f"{F.cosine_similarity(avg_output.unsqueeze(0), signal_vec.unsqueeze(0)).item():.3f}")

print(f"\n方法2 - 注意力聚合:")
print(f"  注意力权重: {[f'{w:.3f}' for w in attn_w.squeeze().tolist()]}")
print(f"  输出: {[f'{v:.3f}' for v in attn_output.tolist()]}")
print(f"  与信号的余弦相似度: "
      f"{F.cosine_similarity(attn_output.unsqueeze(0), signal_vec.unsqueeze(0)).item():.3f}")

# ---- 可学习注意力的训练演示 ----
print(f"\n--- 方法3: 可学习注意力 (训练演示) ---")


class SimpleAttention(nn.Module):
    """
    最简单的可学习注意力模块。

    包含 W_Q 和 W_K 两个可学习投影矩阵。

    参数:
        d_in  : 输入特征维度
        d_attn: 注意力空间维度
    """
    def __init__(self, d_in, d_attn):
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_attn, bias=False)
        self.W_K = nn.Linear(d_in, d_attn, bias=False)
        self.scale = d_attn ** 0.5

    def forward(self, query, keys, values):
        """
        前向传播：计算注意力并聚合 Value。

        参数:
            query  : (1, d_in)  查询向量
            keys   : (n, d_in)  键向量序列
            values : (n, d_in)  值向量序列

        返回:
            output : (1, d_in)  注意力聚合结果
            weights: (1, n)     注意力权重
        """
        q = self.W_Q(query)                         # (1, d_attn)
        k = self.W_K(keys)                           # (n, d_attn)
        scores = q @ k.T / self.scale                # (1, n)
        weights = F.softmax(scores, dim=-1)          # (1, n)
        output = weights @ values                    # (1, d_in)
        return output, weights


# 训练可学习注意力
model = SimpleAttention(d_in=4, d_attn=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 目标：让注意力输出尽可能接近信号向量
target = signal_vec.unsqueeze(0)  # (1, 4)
losses = []

for step in range(100):
    out, w = model(query, sequence, sequence)
    loss = F.mse_loss(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

final_out, final_w = model(query, sequence, sequence)
print(f"  训练 100 步后:")
print(f"  注意力权重: {[f'{w:.3f}' for w in final_w.squeeze().detach().tolist()]}")
print(f"  输出: {[f'{v:.3f}' for v in final_out.squeeze().detach().tolist()]}")
print(f"  与信号的余弦相似度: "
      f"{F.cosine_similarity(final_out.detach(), signal_vec.unsqueeze(0)).item():.3f}")
print(f"  损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f}")

# 可视化训练过程
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图：三种方法的输出对比
methods = ["信号\n(目标)", "简单平均\n(无注意力)", "点积注意力\n(无学习)", "学习注意力\n(训练后)"]
outputs = [signal_vec, avg_output, attn_output, final_out.squeeze().detach()]
colors_bar = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

x_pos = np.arange(4)  # 4个维度
bar_width = 0.2
for idx, (method, out, c) in enumerate(zip(methods, outputs, colors_bar)):
    axes[0].bar(x_pos + idx * bar_width, out.numpy(), bar_width,
                label=method, color=c, alpha=0.85)

axes[0].set_xlabel("特征维度", fontsize=11)
axes[0].set_ylabel("值", fontsize=11)
axes[0].set_title("三种聚合方法的输出对比", fontsize=12)
axes[0].set_xticks(x_pos + 1.5 * bar_width)
axes[0].set_xticklabels([f"d{i}" for i in range(4)])
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.2, axis="y")

# 右图：训练损失曲线
axes[1].plot(losses, color="#9b59b6", linewidth=2)
axes[1].set_xlabel("训练步数", fontsize=11)
axes[1].set_ylabel("MSE 损失", fontsize=11)
axes[1].set_title("可学习注意力的训练过程", fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.suptitle("注意力的价值：精准提取信号，忽略噪声",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("06_01_attention_effect.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 06_01_attention_effect.png")

print("\n核心结论：")
print("  - 简单平均：信号被噪声稀释，信息丢失严重")
print("  - 点积注意力：利用相似度找到信号，效果好很多")
print("  - 可学习注意力：通过训练优化 QK 投影，效果最好")
print("  - 注意力的本质价值 = 自适应地聚合相关信息")


# ════════════════════════════════════════════════════════════════════
# 第7部分：完整总结与思考题
# ════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了注意力机制的直觉与本质：

  1. 信息检索类比:  Query 找 Key → 相关度 → 加权聚合 Value
  2. Hard vs Soft:  argmax(不可微) vs softmax(可微，主流)
  3. QKV 手动演示:  score = QK^T/√d → softmax → ×V → 输出
  4. 权重可视化:    热力图展示"谁在关注谁"
  5. 为什么学 QKV:  不同投影 = 不同视角，让模型学会不对称关注
  6. 注意力效果:    比简单平均强得多，能精准提取信号

注意力的一句话总结：
  给定 Query，在所有 Key 中找到最相关的，
  然后对对应的 Value 做加权求和。
  这就是 Transformer 的灵魂。

下一节将深入 Scaled Dot-Product Attention 和 Multi-Head Attention！
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【缩放因子 √d_k 的作用】
   在注意力公式中，为什么要除以 √d_k？
   试试去掉缩放因子（即直接用 QK^T 做 softmax），
   当 d_k=64 和 d_k=512 时，softmax 输出有什么变化？
   提示：d_k 越大，点积的方差越大，softmax 输出越趋向 one-hot，
   梯度越小，训练越困难。

2. 【注意力的计算复杂度】
   对于长度为 n 的序列，计算 QK^T 的时间复杂度是多少？
   这对处理长文本（如 n=100,000）有什么影响？
   你能想到什么办法降低复杂度吗？
   提示：O(n^2·d_k)，这就是为什么长文本需要 FlashAttention、
   稀疏注意力等技巧。

3. 【Q=K 时会发生什么？】
   如果不使用独立的 W_Q 和 W_K，而是让 Q=K（即共享投影矩阵），
   注意力矩阵会有什么特殊性质？这样做有什么缺点？
   提示：QK^T 会变成对称矩阵，而实际的注意力关系通常是不对称的——
   "猫关注睡觉"不等于"睡觉关注猫"。

4. 【Soft Attention 的温度控制】
   在 softmax 之前乘以一个温度参数 τ：softmax(scores / τ)
   当 τ→0 时，Soft Attention 趋近什么？当 τ→∞ 时呢？
   实际训练中，如何利用温度来控制注意力的"锐利程度"？
   提示：τ→0 趋近 Hard Attention (one-hot)，
        τ→∞ 趋近均匀分布（所有位置等权重）。

5. 【从注意力到 Transformer】
   本节介绍的是单头注意力。真正的 Transformer 使用"多头注意力"
   (Multi-Head Attention)，即把 Q, K, V 分成多个头独立计算再拼接。
   你觉得多头注意力的好处是什么？一个头学到的和多个头学到的有什么区别？
   提示：不同的头可以关注不同类型的关系——有的头关注语法结构，
   有的头关注语义相似性，有的头关注位置关系。
""")

print("下一节预告: 第6章 · 第2节 · Scaled Dot-Product Attention 与 Multi-Head Attention")
