"""
==============================================================
第5章 第1节：注意力机制 QKV —— 用英文阅读理解来理解
==============================================================

【为什么需要它？】
RNN/LSTM 的问题：
  1. 串行计算（必须一步一步），无法并行，速度慢
  2. 长序列中，早期信息仍然难以保存（即使是 LSTM）
  3. 固定大小的隐状态（bottleneck）压缩了所有信息

注意力机制的解决方案：
  - 任意两个位置可以直接"对话"，不需要经过中间状态
  - 可以并行计算（对 GPU 友好）
  - 信息不需要经过压缩瓶颈

【完整类比：英文阅读理解】

想象你在做英文阅读理解题：
  文章 = 一组句子，每个句子有"关键词"和"内容"
  Key  = 每个句子的关键词摘要（索引/目录）
  Value = 每个句子的完整内容
  Query = 你要回答的问题

做题过程：
  1. 读题目（Query）
  2. 扫一遍文章各句子的关键词（Key），找最相关的
  3. 根据相关度（打分），加权阅读各句子的内容（Value）
  4. 综合得到答案

注意力机制 = 这个"读题-找关键词-综合内容"的过程

具体实现：
  1. score = Q @ K.T / √d_k   （每个query和每个key的相似度）
  2. weights = softmax(score)  （把分数变成权重，和为1）
  3. output = weights @ V      （加权求和所有value）

【存在理由】
解决问题：RNN 的串行性和长距离依赖问题
核心思想：动态、基于内容的加权聚合，任意位置可以直接交互
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(42)

# ============================================================
# Part 1: 手动演示 —— 用数字模拟阅读理解
# ============================================================
print("=" * 50)
print("Part 1: 用具体数字理解 QKV")
print("=" * 50)

"""
场景：3个句子（3个 Key-Value 对），1个问题（Query）

模拟：
  句子1：主题是"天气"，内容是 [2, 0, 0, 1]
  句子2：主题是"食物"，内容是 [0, 3, 0, 0]
  句子3：主题是"运动"，内容是 [0, 0, 4, 0]

  Keys（主题摘要）：
    k1 = [1, 0, 0]  （天气）
    k2 = [0, 1, 0]  （食物）
    k3 = [0, 0, 1]  （运动）

  Query（问题）：
    q = [0.8, 0.2, 0.0]  （主要问天气，稍微问食物）

期望结果：注意力权重 ≈ [0.8, 0.2, 0.0]
         输出 ≈ 0.8*句子1内容 + 0.2*句子2内容
"""

# Keys: 3个句子的摘要向量
K = np.array([
    [1.0, 0.0, 0.0],  # 句子1的key：天气
    [0.0, 1.0, 0.0],  # 句子2的key：食物
    [0.0, 0.0, 1.0],  # 句子3的key：运动
])

# Values: 3个句子的内容
V = np.array([
    [2.0, 0.0, 0.0, 1.0],  # 句子1的内容
    [0.0, 3.0, 0.0, 0.0],  # 句子2的内容
    [0.0, 0.0, 4.0, 0.0],  # 句子3的内容
])

# Query: 问题向量
Q = np.array([[0.8, 0.2, 0.0]])  # (1, 3)

# ===== 注意力计算 =====
d_k = K.shape[-1]  # Key 的维度

# Step 1: 计算 Query 和每个 Key 的相似度
scores = Q @ K.T  # (1, 3) @ (3, 3) = (1, 3)
print("Step 1: Query 和每个 Key 的相似度（点积）")
print(f"  scores = Q @ K.T = {scores}")

# Step 2: 缩放（为什么要 ÷√d_k？见下文）
scaled_scores = scores / np.sqrt(d_k)
print(f"\nStep 2: 缩放（÷√d_k = ÷√{d_k} = {np.sqrt(d_k):.2f}）")
print(f"  scaled_scores = {scaled_scores}")

# Step 3: Softmax → 权重
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scaled_scores)
print(f"\nStep 3: Softmax → 注意力权重（和为1）")
print(f"  weights = {weights}")
print(f"  权重之和 = {weights.sum():.2f}")

# Step 4: 加权求和 Values
output = weights @ V  # (1, 3) @ (3, 4) = (1, 4)
print(f"\nStep 4: 加权求和 Values → 输出")
print(f"  output = weights @ V = {output}")
print(f"\n  解释：")
print(f"    = {weights[0,0]:.2f} × 句子1内容 + {weights[0,1]:.2f} × 句子2内容 + {weights[0,2]:.2f} × 句子3内容")
print(f"    = {weights[0,0]:.2f} × {V[0]} + {weights[0,1]:.2f} × {V[1]} + {weights[0,2]:.2f} × {V[2]}")

# ============================================================
# Part 2: 为什么要除以 √d_k？
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 为什么要缩放 ÷√d_k？")
print("=" * 50)

"""
当 d_k（Key的维度）很大时：
  点积 Q @ K.T 的量级 = O(d_k)（每个元素平均贡献 1）
  → 点积值很大！
  → softmax 的输入很大
  → softmax 输出非常"尖锐"（接近 one-hot）
  → 梯度消失！（softmax 在极端值时梯度趋近0）

除以 √d_k 使点积的方差保持为 1（归一化效果）
"""

d_k_values = [4, 16, 64, 256]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for d_k in d_k_values:
    # 生成随机 Q, K
    Q_rand = np.random.randn(1, d_k)
    K_rand = np.random.randn(10, d_k)

    # 无缩放的 softmax 输出
    scores_no_scale = (Q_rand @ K_rand.T).ravel()
    scores_scaled = scores_no_scale / np.sqrt(d_k)

    weights_no_scale = softmax(scores_no_scale.reshape(1,-1)).ravel()
    weights_scaled = softmax(scores_scaled.reshape(1,-1)).ravel()

    ax.plot(range(10), weights_no_scale, '--', label=f'd_k={d_k}（无缩放）', alpha=0.6)

ax.set_title('不缩放的 Softmax 输出\n（d_k大时，几乎只有一个权重≈1）')
ax.set_xlabel('位置')
ax.set_ylabel('注意力权重')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1]
for d_k in d_k_values:
    Q_rand = np.random.randn(1, d_k)
    K_rand = np.random.randn(10, d_k)
    scores_scaled = (Q_rand @ K_rand.T / np.sqrt(d_k)).ravel()
    weights_scaled = softmax(scores_scaled.reshape(1,-1)).ravel()
    ax.plot(range(10), weights_scaled, '-o', markersize=4,
           label=f'd_k={d_k}（÷√d_k）', alpha=0.8)

ax.set_title('缩放后的 Softmax 输出\n（分布更均匀，梯度不消失）')
ax.set_xlabel('位置')
ax.set_ylabel('注意力权重')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_transformer_attention/scaling_effect.png', dpi=100, bbox_inches='tight')
print("图片已保存：05_transformer_attention/scaling_effect.png")
plt.show()

# ============================================================
# Part 3: 注意力函数完整实现
# ============================================================
print("Part 3: 注意力函数实现")
print("=" * 50)

def attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Q: (batch, seq_q, d_k)   —— 查询
    K: (batch, seq_k, d_k)   —— 键
    V: (batch, seq_k, d_v)   —— 值
    mask: (batch, seq_q, seq_k)  可选，-inf 会被 softmax 抹掉

    返回：
      output: (batch, seq_q, d_v)  —— 加权求和后的输出
      weights: (batch, seq_q, seq_k)  —— 注意力权重（用于可视化）
    """
    d_k = K.shape[-1]

    # (batch, seq_q, d_k) @ (batch, d_k, seq_k) = (batch, seq_q, seq_k)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask  # mask 的值是 -inf（大负数）

    weights = softmax(scores)  # (batch, seq_q, seq_k)

    # (batch, seq_q, seq_k) @ (batch, seq_k, d_v) = (batch, seq_q, d_v)
    output = weights @ V

    return output, weights

# ============================================================
# Part 4: 因果掩码（Causal Mask）—— 解码器用
# ============================================================
print("Part 4: 因果掩码（Decoder 用）")
print("=" * 50)

"""
在生成文本时（如 GPT），每个位置只能看到自己和之前的位置
（不能"看未来"！）

因果掩码：上三角矩阵，把"未来"位置设为 -∞
  softmax(-∞) = 0，这样未来位置的注意力权重为0
"""

def make_causal_mask(seq_len):
    """
    创建因果掩码（下三角为0，上三角为-inf）
    """
    mask = np.zeros((seq_len, seq_len))
    mask = np.triu(mask + (-1e9), k=1)  # 上三角（不含对角）设为极大负数
    return mask  # (seq_len, seq_len)

seq_len = 5
causal_mask = make_causal_mask(seq_len)

print("因果掩码（0=可以看，-∞=不能看未来）：")
print(causal_mask)
print("\n对应的 softmax 权重（0行=只看自己，4行=能看所有）：")
for i in range(seq_len):
    row_weights = softmax(causal_mask[i].reshape(1, -1)).ravel()
    print(f"  位置{i}: {row_weights.round(3)}")

# ============================================================
# Part 5: 位置编码 —— 让 Transformer 知道顺序
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 位置编码 —— 给序列加上位置信息")
print("=" * 50)

"""
注意力机制本身是"位置无关"的：
  同一个词，不管在句子哪里，注意力计算方式完全相同
  → Transformer 不知道词序！

解决方案：位置编码（Positional Encoding）
  在词向量中加入位置信息

Transformer 原论文使用正弦/余弦位置编码：
  PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
  PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

直觉：
  - 不同频率的正弦/余弦波叠加，每个位置有独一无二的"指纹"
  - 低维度：低频变化（反映长距离位置差异）
  - 高维度：高频变化（反映短距离位置差异）
  - 连续且可外推：没见过的长度也能有合理位置编码
"""

def positional_encoding(max_seq_len, d_model):
    """
    Transformer 的正弦位置编码
    返回 (max_seq_len, d_model)
    """
    PE = np.zeros((max_seq_len, d_model))
    pos = np.arange(max_seq_len).reshape(-1, 1)    # (seq_len, 1)
    i = np.arange(0, d_model, 2).reshape(1, -1)    # (1, d_model/2)

    # 频率：1/10000^{2i/d_model}
    div_term = 1 / (10000 ** (i / d_model))

    PE[:, 0::2] = np.sin(pos * div_term)  # 偶数维度：sin
    PE[:, 1::2] = np.cos(pos * div_term)  # 奇数维度：cos

    return PE

PE = positional_encoding(max_seq_len=50, d_model=64)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 热力图
ax = axes[0]
im = ax.imshow(PE[:20, :], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('位置编码维度')
ax.set_ylabel('序列位置')
ax.set_title('位置编码热力图\n（每一行是一个位置的编码，每行都不同）')
plt.colorbar(im, ax=ax)

# 几个维度的波形
ax = axes[1]
for dim in [0, 2, 10, 30, 62]:
    ax.plot(PE[:, dim], label=f'维度 {dim}')
ax.set_xlabel('序列位置')
ax.set_ylabel('编码值')
ax.set_title('不同维度的位置编码波形\n（维度越高，频率越高）')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_transformer_attention/positional_encoding.png', dpi=100, bbox_inches='tight')
print("图片已保存：05_transformer_attention/positional_encoding.png")
plt.show()

# ============================================================
# Part 6: 注意力权重可视化
# ============================================================
print("\nPart 6: 注意力权重可视化")
print("=" * 50)

"""
用一个简单的例子可视化注意力：
  输入序列：["The", "cat", "sat", "on", "the", "mat"]
  每个词用随机向量表示（真实情况是词嵌入）
"""

words = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(words)
d_k = 8  # 每个词的向量维度

# 随机词嵌入（实际中是学习出来的）
np.random.seed(7)
word_embeddings = np.random.randn(1, seq_len, d_k)  # (batch=1, seq=6, d=8)

# Q, K, V 的投影矩阵（实际中也是学习的）
W_Q = np.random.randn(d_k, d_k) * 0.1
W_K = np.random.randn(d_k, d_k) * 0.1
W_V = np.random.randn(d_k, d_k) * 0.1

Q = word_embeddings @ W_Q.T
K = word_embeddings @ W_K.T
V = word_embeddings @ W_V.T

output, weights = attention(Q, K, V)
weights_vis = weights[0]  # (seq, seq) —— 去掉 batch 维度

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(weights_vis, cmap='Blues', vmin=0, vmax=weights_vis.max())
ax.set_xticks(range(seq_len))
ax.set_yticks(range(seq_len))
ax.set_xticklabels(words, rotation=45, fontsize=12)
ax.set_yticklabels(words, fontsize=12)
ax.set_title('注意力权重矩阵\n（行=Query，列=Key；颜色越深=注意力越强）')
ax.set_xlabel('Key（被关注的词）')
ax.set_ylabel('Query（正在看的词）')

for i in range(seq_len):
    for j in range(seq_len):
        ax.text(j, i, f'{weights_vis[i,j]:.2f}', ha='center', va='center',
               fontsize=8, color='black' if weights_vis[i,j] < 0.3 else 'white')

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('05_transformer_attention/attention_weights.png', dpi=100, bbox_inches='tight')
print("图片已保存：05_transformer_attention/attention_weights.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【自注意力 vs 交叉注意力】
   自注意力：Q, K, V 都来自同一个序列（句子内部对话）
   交叉注意力：Q 来自一个序列，K/V 来自另一个序列（翻译：中文Q看英文KV）
   在机器翻译中，Decoder 用交叉注意力读 Encoder 的输出。
   请解释：为什么翻译时需要这种设计？

2. 【温度缩放（Temperature Scaling）】
   有时把 softmax 改为 softmax(scores / τ)，τ 是"温度"。
   - τ < 1（低温）：权重更集中（更确定地关注某一个词）
   - τ > 1（高温）：权重更均匀（"关注"所有词）
   在 ChatGPT 等模型的采样中，temperature 参数控制的就是这个。
   代入不同的 τ（0.1, 0.5, 1.0, 2.0），可视化权重分布的变化。

3. 【位置编码的设计】
   为什么用 sin/cos 而不是直接用位置整数（0,1,2,...）？
   提示：考虑两点：
   a）整数编码在很长序列时值会很大（比词向量大很多）
   b）sin/cos 的差异性质：PE(pos+k) 可以用 PE(pos) 线性表示
      → 模型可以更容易地学习"相对位置"
""")
