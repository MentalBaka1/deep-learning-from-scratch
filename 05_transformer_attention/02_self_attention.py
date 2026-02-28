"""
==============================================================
第5章 第2节：自注意力 —— 句子的自我审视
==============================================================

【为什么需要它？】
在普通注意力中，Q 和 K/V 可以来自不同序列（交叉注意力）。
自注意力（Self-Attention）是一种特殊情况：
  Q、K、V 全部来自同一个序列

直觉："句子自我审视"
  每个词都问：在这个句子里，哪些词和我最相关？
  例："The animal didn't cross the street because it was too tired"
    "it" 应该关注 "animal"（而不是 "street"）
    → 自注意力能学到这种长距离依赖！

多头注意力（Multi-Head Attention）：
  类比：用多个"阅读角度"同时阅读
  一个头可能关注"语法关系"（主语-谓语）
  另一个头可能关注"语义相关性"（名词-形容词）
  最后把所有头的输出拼接，综合所有视角

【存在理由】
解决问题：单一注意力视角可能不够，不同类型的关系需要不同的表示子空间
核心思想：并行运行多个独立的注意力头，从不同子空间捕获不同关系
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights

# ============================================================
# Part 1: 单头自注意力
# ============================================================
print("=" * 50)
print("Part 1: 单头自注意力")
print("=" * 50)

"""
自注意力的实现：
  1. 输入 X: (batch, seq_len, d_model)
  2. 三个线性投影：
     Q = X @ W_Q  （学习"我想查询什么"）
     K = X @ W_K  （学习"我有什么可以被查询"）
     V = X @ W_V  （学习"我的实际内容是什么"）
  3. 用 attention(Q, K, V) 计算输出

关键：W_Q, W_K, W_V 是可学习的参数！
  不同的任务会学到不同的投影方式
"""

class SingleHeadSelfAttention:
    def __init__(self, d_model, d_k=None, d_v=None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        # 可学习的投影矩阵
        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, self.d_k) * scale
        self.W_K = np.random.randn(d_model, self.d_k) * scale
        self.W_V = np.random.randn(d_model, self.d_v) * scale

        self.cache = None

    def forward(self, X, mask=None):
        """
        X: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_v)
        """
        # 投影到 Q, K, V 空间
        Q = X @ self.W_Q  # (batch, seq, d_k)
        K = X @ self.W_K  # (batch, seq, d_k)
        V = X @ self.W_V  # (batch, seq, d_v)

        output, weights = attention(Q, K, V, mask)
        self.cache = (X, Q, K, V, weights)

        return output, weights

    def backward(self, d_output):
        """反向传播"""
        X, Q, K, V, weights = self.cache
        batch, seq_len, d_k = Q.shape
        d_v = V.shape[-1]

        # attention 的反向
        d_weights = d_output @ V.transpose(0, 2, 1)  # (batch, seq, seq)
        d_V = weights.transpose(0, 2, 1) @ d_output  # (batch, seq, d_v)

        # softmax 反向（对每行）
        d_scores = np.zeros_like(d_weights)
        for b in range(batch):
            for i in range(seq_len):
                w = weights[b, i]  # (seq,)
                d_w = d_weights[b, i]  # (seq,)
                # Jacobian of softmax: diag(w) - w*w.T
                d_scores[b, i] = w * (d_w - np.sum(d_w * w))

        d_scores /= np.sqrt(d_k)  # 除以 √d_k

        # Q, K 的梯度
        d_Q = d_scores @ K  # (batch, seq, d_k)
        d_K = d_scores.transpose(0, 2, 1) @ Q  # (batch, seq, d_k)

        # 投影矩阵的梯度
        self.d_WQ = X.transpose(0, 2, 1).reshape(-1, self.d_model).T.reshape(batch, self.d_model, seq_len)
        # 简化：只计算关键梯度
        self.d_WQ = np.sum(X.transpose(0, 2, 1) @ d_Q, axis=0)
        self.d_WK = np.sum(X.transpose(0, 2, 1) @ d_K, axis=0)
        self.d_WV = np.sum(X.transpose(0, 2, 1) @ d_V, axis=0)

        # 传回 X 的梯度
        d_X = d_Q @ self.W_Q.T + d_K @ self.W_K.T + d_V @ self.W_V.T
        return d_X

# ============================================================
# Part 2: 多头注意力 —— 多个视角
# ============================================================
print("Part 2: 多头注意力（Multi-Head Attention）")
print("=" * 50)

"""
多头注意力：
  d_model = d_k * n_heads  （把 d_model 等分给 n_heads 个头）
  每个头独立做自注意力，使用不同的 W_Q, W_K, W_V
  所有头的输出拼接，再通过 W_O 投影回 d_model

  MultiHead(X) = Concat(head_1, ..., head_h) @ W_O

为什么多头？
  - 不同头可以关注不同类型的关系
  - 在"猫跑了"中：
    一个头关注"谁做了动作"（猫→跑）
    另一个头关注"时态"（跑了→过去时）
  - 集成多个视角，表达能力更强

实现技巧：不需要 n_heads 个独立的矩阵
  把 W_Q 设计为 (d_model, d_model)，把结果 reshape 成 n_heads 个头
  这样一次矩阵乘法就能计算所有头
"""

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        scale = 1.0 / np.sqrt(d_model)
        # 一次性计算所有头的 Q, K, V（合并成大矩阵）
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

        self.cache = None

    def forward(self, X, mask=None):
        """
        X: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = X.shape
        H = self.n_heads
        d_k = self.d_k

        # 投影 + 分头
        Q = X @ self.W_Q  # (batch, seq, d_model)
        K = X @ self.W_K
        V = X @ self.W_V

        # Reshape 分成 n_heads 个头
        # (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        def split_heads(x):
            return x.reshape(batch, seq_len, H, d_k).transpose(0, 2, 1, 3)

        Q = split_heads(Q)  # (batch, H, seq, d_k)
        K = split_heads(K)
        V = split_heads(V)

        # 每个头独立做注意力
        # 合并 batch 和 heads 维度，用 attention 批量计算
        Q_flat = Q.reshape(batch * H, seq_len, d_k)
        K_flat = K.reshape(batch * H, seq_len, d_k)
        V_flat = V.reshape(batch * H, seq_len, d_k)

        head_outputs, all_weights = attention(Q_flat, K_flat, V_flat, mask)

        # 还原形状：(batch, H, seq, d_k) → (batch, seq, d_model)
        head_outputs = head_outputs.reshape(batch, H, seq_len, d_k)
        head_outputs = head_outputs.transpose(0, 2, 1, 3)  # (batch, seq, H, d_k)
        concat = head_outputs.reshape(batch, seq_len, d_model)  # 拼接所有头

        # 最终输出投影
        output = concat @ self.W_O

        all_weights = all_weights.reshape(batch, H, seq_len, seq_len)
        self.cache = (X, concat, all_weights)

        return output, all_weights

# ============================================================
# Part 3: 测试多头注意力
# ============================================================
print("Part 3: 多头注意力测试")
print("=" * 50)

batch, seq_len, d_model = 2, 6, 64
n_heads = 8
X_test = np.random.randn(batch, seq_len, d_model)

mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
output, weights = mha.forward(X_test)

print(f"输入形状：{X_test.shape}")
print(f"输出形状：{output.shape}  （和输入相同！）")
print(f"注意力权重形状：{weights.shape}  （batch, n_heads, seq, seq）")
print(f"\n每个头的注意力权重（对第一个样本）：")
for head in range(n_heads):
    max_attention = weights[0, head].max(axis=1).mean()  # 平均最大注意力
    print(f"  头 {head+1}: 平均最大注意力权重 = {max_attention:.3f}")

# ============================================================
# Part 4: 可视化不同头的注意力模式
# ============================================================
print("\nPart 4: 可视化多头注意力")
print("=" * 50)

# 用一个句子演示（用词 ID 模拟）
words = ["I", "love", "deep", "learning", "so", "much"]
seq_len = len(words)
d_model = 16
n_heads = 4

np.random.seed(42)
X = np.random.randn(1, seq_len, d_model)
mha_demo = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
_, weights_demo = mha_demo.forward(X)

fig, axes = plt.subplots(2, n_heads, figsize=(16, 8))
fig.suptitle(f'多头注意力：{n_heads}个头的注意力模式对比', fontsize=13)

for h in range(n_heads):
    ax = axes[0][h]
    w = weights_demo[0, h]  # (seq, seq)
    im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max())
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(words, rotation=45, fontsize=8)
    ax.set_yticklabels(words, fontsize=8)
    ax.set_title(f'头 {h+1}', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1][h]
    # 每个词"最关注"的词（argmax）
    max_attention_pos = w.argmax(axis=1)
    ax2.barh(range(seq_len), w.max(axis=1), color='steelblue', alpha=0.7)
    ax2.set_yticks(range(seq_len))
    ax2.set_yticklabels(words, fontsize=8)
    ax2.set_xlabel('最大注意力权重')
    ax2.set_title(f'头{h+1}每个词的最大注意力', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    for i, (pos, val) in enumerate(zip(max_attention_pos, w.max(axis=1))):
        ax2.text(val + 0.01, i, words[pos], va='center', fontsize=7, color='red')

plt.tight_layout()
plt.savefig('05_transformer_attention/multi_head_attention.png', dpi=80, bbox_inches='tight')
print("图片已保存：05_transformer_attention/multi_head_attention.png")
plt.show()

print("\n观察：不同头关注的模式不同，这正是多头的优势！")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【计算量分析】
   对于 seq_len=512，d_model=512，n_heads=8：
   - Q @ K.T 的计算量（FLOPs）= ?
   - 注意力计算总量（所有头）= ?
   - 这就是为什么 Transformer 对长序列二次方复杂 O(n²d)！
   - 现代方法（Flash Attention, Sparse Attention）如何解决？

2. 【头的特化】
   对一个训练好的 BERT 模型，研究人员发现：
   - 某些头专门关注"相邻词"（语法结构）
   - 某些头专门关注"语义相关词"（同义词/相关概念）
   - 某些头关注"句子边界"
   如果你发现某个头的注意力权重几乎总是对角矩阵（每个词只关注自己），
   这个头有用吗？能把它"剪掉"吗？（提示：Attention Head Pruning）

3. 【实现 Masked Self-Attention】
   给 MultiHeadAttention.forward() 添加因果掩码支持：
   - 生成 (1, 1, seq, seq) 的因果掩码（可以广播到所有头）
   - 验证：位置0只能看自己；位置3可以看0,1,2,3
   这是 GPT 使用的解码器自注意力。
""")
