"""
==============================================================
第5章 第3节：Transformer 编码器 —— 拼图拼在一起
==============================================================

【为什么需要它？】
前两节学了注意力和自注意力，但 Transformer 不只是注意力：
  - 只有注意力：模型只会"关注"，但没有复杂的特征变换能力
  - 只有 MLP：模型有特征变换，但无法捕捉序列内的位置关系

Transformer 编码器 = 自注意力 + 前馈网络 + 残差连接 + 归一化
  把这些积木组合起来，发挥 1+1>2 的效果

【组件清单】
1. LayerNorm（层归一化）：让每个位置的特征分布稳定
2. MultiHeadAttention：序列内的"信息聚合"（之前已实现）
3. FeedForward（FFN）：每个位置独立的"特征变换"
4. 残差连接：梯度高速公路（和 ResNet 一样的思路）
5. 位置编码：注意力没有位置感，需要额外注入

【完整 Encoder Layer 结构】
  输入 X
  → MHA(X) + X → LayerNorm  （子层1：注意力）
  → FFN(X) + X → LayerNorm  （子层2：前馈）
  → 输出

【存在理由】
解决问题：需要一个既能捕捉序列关系，又能变换特征表示的强大模块
核心思想：把注意力、非线性变换、残差、归一化组合成可叠加的积木块
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# ============================================================
# Part 1: LayerNorm —— 为什么不用 BatchNorm？
# ============================================================
print("=" * 50)
print("Part 1: LayerNorm vs BatchNorm")
print("=" * 50)

"""
BatchNorm（之前学过）：
  在 batch 维度上归一化：对每个特征，统计 batch 内所有样本的均值/方差
  μ[j] = mean(x[:, j])   （第 j 个特征在整个 batch 上的均值）

BatchNorm 对序列任务的问题：
  1. 序列长度可变：batch 内序列长度不同，很难统一处理
  2. 推理时用的是"训练集的运行统计"，如果序列长度分布和训练不同，效果差
  3. 每个位置的语义不同：位置1（主语）和位置5（动词）统一归一化意义不大

LayerNorm（层归一化）：
  在特征维度上归一化：对每个样本的每个位置，统计该位置所有特征的均值/方差
  μ[b, i] = mean(x[b, i, :])  （第 b 个样本第 i 个位置的特征均值）

好处：
  - 与 batch size、序列长度无关
  - 每个位置独立归一化（语义合理）
  - 推理时行为与训练时完全一致
"""

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        # 可学习的缩放和平移参数（初始化为"不改变分布"）
        self.gamma = np.ones(d_model)   # 缩放
        self.beta = np.zeros(d_model)   # 偏移
        self.cache = None

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        在最后一个维度（特征维度）上归一化
        """
        mu = x.mean(axis=-1, keepdims=True)      # (batch, seq, 1)
        var = x.var(axis=-1, keepdims=True)       # (batch, seq, 1)
        x_hat = (x - mu) / np.sqrt(var + self.eps)  # 归一化
        out = self.gamma * x_hat + self.beta      # 可学习的缩放平移

        self.cache = (x, x_hat, mu, var)
        return out

    def backward(self, d_out):
        x, x_hat, mu, var = self.cache
        N = x.shape[-1]  # d_model

        self.d_gamma = (d_out * x_hat).sum(axis=(0, 1))
        self.d_beta = d_out.sum(axis=(0, 1))

        dx_hat = d_out * self.gamma
        dvar = (dx_hat * (x - mu) * (-0.5) * (var + self.eps)**(-1.5)).sum(axis=-1, keepdims=True)
        dmu = (dx_hat * (-1 / np.sqrt(var + self.eps))).sum(axis=-1, keepdims=True)
        dx = dx_hat / np.sqrt(var + self.eps) + 2 * dvar * (x - mu) / N + dmu / N
        return dx

# 验证 LayerNorm
print("验证 LayerNorm 的效果：")
x_demo = np.array([[[1., 2., 3., 4.],
                     [10., 20., 30., 40.]]])  # (1, 2, 4)
ln = LayerNorm(d_model=4)
out = ln.forward(x_demo)
print(f"  输入第0位置均值={x_demo[0,0].mean():.1f}, 方差={x_demo[0,0].var():.2f}")
print(f"  输入第1位置均值={x_demo[0,1].mean():.1f}, 方差={x_demo[0,1].var():.2f}")
print(f"  归一化后第0位置均值≈{out[0,0].mean():.6f}, 方差≈{out[0,0].var():.4f}")
print(f"  归一化后第1位置均值≈{out[0,1].mean():.6f}, 方差≈{out[0,1].var():.4f}")
print(f"  ✓ 无论原始分布如何，归一化后均值≈0，方差≈1")

# ============================================================
# Part 2: FeedForward 网络 —— 每个位置的特征变换
# ============================================================
print("\nPart 2: 前馈网络（FFN）")
print("=" * 50)

"""
注意力做了什么：把序列中各位置的信息"混合"在一起
FFN 做了什么：对每个位置的特征做非线性变换（但不混合位置）

FFN 结构：
  Linear(d_model → d_ff)  → ReLU → Linear(d_ff → d_model)

  其中 d_ff = 4 * d_model（通常是 4 倍扩展）

为什么要 4 倍扩展？
  - 在高维空间中有更多的"特征组合可能性"
  - 然后压缩回 d_model，强迫模型学到最重要的变换
  - 这是 Transformer 参数量的主要来源！

FFN 和注意力的分工：
  注意力 = "在哪里找信息"（关注哪些位置）
  FFN    = "如何变换信息"（对已找到的信息做特征变换）
"""

class FeedForward:
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model  # 默认 4 倍扩展

        # He 初始化
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / self.d_ff)
        self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)
        self.cache = None

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        对每个位置独立做两层线性变换（逐位置应用，不跨位置交互）
        """
        z1 = x @ self.W1 + self.b1  # (batch, seq, d_ff)
        a1 = np.maximum(0, z1)       # ReLU 激活
        z2 = a1 @ self.W2 + self.b2  # (batch, seq, d_model)

        self.cache = (x, z1, a1)
        return z2

    def backward(self, d_out):
        x, z1, a1 = self.cache

        # 第二层反向
        self.dW2 = a1.reshape(-1, self.d_ff).T @ d_out.reshape(-1, self.d_model)
        self.db2 = d_out.sum(axis=(0, 1))
        d_a1 = d_out @ self.W2.T

        # ReLU 反向
        d_z1 = d_a1 * (z1 > 0)

        # 第一层反向
        self.dW1 = x.reshape(-1, self.d_model).T @ d_z1.reshape(-1, self.d_ff)
        self.db1 = d_z1.sum(axis=(0, 1))
        dx = d_z1 @ self.W1.T

        return dx

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

print("FFN 结构：")
d_model, d_ff = 64, 256
print(f"  输入：d_model = {d_model}")
print(f"  中间：d_ff = {d_ff}（4倍扩展）")
print(f"  输出：d_model = {d_model}")
print(f"  FFN 参数量：{d_model*d_ff + d_ff + d_ff*d_model + d_model:,}")
print(f"  MHA 参数量（相同d_model，8头）：{4*(d_model*d_model):,}")
print(f"  → FFN 参数量 ≈ MHA 的 {(d_model*d_ff*2)/(4*d_model*d_model):.1f}x")

# ============================================================
# Part 3: 完整的 Encoder Layer
# ============================================================
print("\nPart 3: 组装完整 Encoder Layer")
print("=" * 50)

"""
Transformer Encoder Layer 的完整结构：

  输入 X
      ↓
  ┌─── 残差连接 ────────────────────┐
  │   MultiHeadAttention(X, X, X)  │
  │   + X                          │ → LayerNorm → 中间输出
  └────────────────────────────────┘
      ↓
  ┌─── 残差连接 ────────────────────┐
  │   FeedForward(中间输出)         │
  │   + 中间输出                    │ → LayerNorm → 最终输出
  └────────────────────────────────┘

注意：残差连接在 LayerNorm 之前加（Pre-LN 变体）或之后（Post-LN）
  原始论文是 Post-LN（如上），但现代实现更多用 Pre-LN（更稳定训练）
  这里实现 Post-LN 以贴近原论文
"""

def attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = 1.0 / np.sqrt(d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        self.cache = None

    def forward(self, X, mask=None):
        batch, seq_len, d_model = X.shape
        H, d_k = self.n_heads, self.d_k

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        def split_heads(x):
            return x.reshape(batch, seq_len, H, d_k).transpose(0, 2, 1, 3)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        Q_flat = Q.reshape(batch * H, seq_len, d_k)
        K_flat = K.reshape(batch * H, seq_len, d_k)
        V_flat = V.reshape(batch * H, seq_len, d_k)

        head_outputs, all_weights = attention(Q_flat, K_flat, V_flat, mask)
        head_outputs = head_outputs.reshape(batch, H, seq_len, d_k)
        head_outputs = head_outputs.transpose(0, 2, 1, 3)
        concat = head_outputs.reshape(batch, seq_len, d_model)
        output = concat @ self.W_O

        all_weights = all_weights.reshape(batch, H, seq_len, seq_len)
        self.cache = (X, concat, all_weights)
        return output, all_weights

class TransformerEncoderLayer:
    """
    一个完整的 Transformer Encoder 层
    包含：自注意力 + 残差 + LayerNorm + FFN + 残差 + LayerNorm
    """
    def __init__(self, d_model, n_heads, d_ff=None):
        self.d_model = d_model
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, X, mask=None):
        """
        X: (batch, seq_len, d_model)
        """
        # 子层1：自注意力 + 残差连接 + LayerNorm
        attn_out, weights = self.attn.forward(X, mask)
        X2 = self.ln1.forward(X + attn_out)   # 残差 + 归一化

        # 子层2：前馈网络 + 残差连接 + LayerNorm
        ffn_out = self.ffn.forward(X2)
        X3 = self.ln2.forward(X2 + ffn_out)  # 残差 + 归一化

        return X3, weights

# 验证 Encoder Layer
batch, seq_len, d_model = 2, 8, 32
n_heads = 4

X_test = np.random.randn(batch, seq_len, d_model)
enc_layer = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads)
out, weights = enc_layer.forward(X_test)

print(f"输入形状：{X_test.shape}")
print(f"输出形状：{out.shape}  （维度不变！）")
print(f"注意力权重：{weights.shape}  （batch, heads, seq, seq）")

# ============================================================
# Part 4: 位置编码 + 完整 Encoder 堆叠
# ============================================================
print("\nPart 4: 位置编码 + 多层 Encoder 堆叠")
print("=" * 50)

"""
问题：自注意力是"无序的"
  attention(Q, K, V) 对词的顺序不敏感！
  "猫追狗" 和 "狗追猫" 会得到相同的注意力结构（如果词向量相同）

解决：位置编码（Positional Encoding）
  把位置信息编码成向量，加到词嵌入上
  模型通过读取这个额外信息来理解位置

Transformer 原始论文用的正弦/余弦位置编码：
  PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

为什么用三角函数？
  1. 每个位置有唯一的编码（不同频率的正弦波组合）
  2. 可以推广到比训练集更长的序列（三角函数是无限延伸的）
  3. 相对位置信息可以通过线性变换获得（数学上可以证明）
"""

def positional_encoding(seq_len, d_model):
    """
    生成 Transformer 原始论文中的正弦位置编码
    返回: (1, seq_len, d_model) 可以广播到任意 batch
    """
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)  # (seq, 1)
    # 每两个维度一组，用一个频率的 sin/cos
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)

    PE[:, 0::2] = np.sin(position / div_term)  # 偶数位：sin
    PE[:, 1::2] = np.cos(position / div_term)  # 奇数位：cos

    return PE[np.newaxis, :, :]  # (1, seq, d_model)

class TransformerEncoder:
    """
    完整的 Transformer Encoder
    = 位置编码 + N 个 EncoderLayer 堆叠
    """
    def __init__(self, d_model, n_heads, n_layers, d_ff=None):
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

    def forward(self, X, mask=None):
        """
        X: (batch, seq_len, d_model) 已经过词嵌入
        """
        # 加入位置编码
        seq_len = X.shape[1]
        PE = positional_encoding(seq_len, self.d_model)
        X = X + PE

        # 逐层传播
        all_weights = []
        for layer in self.layers:
            X, weights = layer.forward(X, mask)
            all_weights.append(weights)

        return X, all_weights

# 测试完整 Encoder
print("位置编码可视化：")
PE = positional_encoding(seq_len=50, d_model=64)
print(f"  位置编码形状：{PE.shape}")
print(f"  PE[0,0] (位置0，前4维) = {PE[0,0,:4].round(3)}")
print(f"  PE[0,1] (位置1，前4维) = {PE[0,1,:4].round(3)}")

encoder = TransformerEncoder(d_model=32, n_heads=4, n_layers=3)
X_input = np.random.randn(2, 10, 32)
encoder_out, all_weights = encoder.forward(X_input)
print(f"\n3层 Encoder：")
print(f"  输入形状：{X_input.shape}")
print(f"  输出形状：{encoder_out.shape}")
print(f"  每层注意力权重：{all_weights[0].shape}")

# ============================================================
# Part 5: 实战 —— 序列分类任务
# ============================================================
print("\nPart 5: 实战 —— 玩具序列分类")
print("=" * 50)

"""
任务：判断数字序列是否"单调递增"
  输入：序列 [x1, x2, ..., xT]（每个数字都嵌入成向量）
  输出：0（不单调）或 1（单调递增）

用 [CLS] token 策略（类似 BERT）：
  在序列开头加一个特殊的 [CLS] token
  Encoder 输出后，取 [CLS] 位置的特征做分类

  [CLS], x1, x2, x3, ... → Encoder → [CLS_out] → Linear → 分类
"""

np.random.seed(42)

# 生成玩具数据集
def make_monotone_dataset(n_samples=500, seq_len=8, vocab_size=10, embed_dim=16):
    """
    生成序列数据：
      正样本：单调递增序列
      负样本：非单调序列
    """
    # 词嵌入矩阵（每个数字对应一个向量）
    embedding = np.random.randn(vocab_size + 1, embed_dim) * 0.5  # +1 for [CLS]
    CLS_IDX = vocab_size  # [CLS] token 的 ID

    X_data = []
    y_data = []

    for _ in range(n_samples):
        if np.random.random() < 0.5:
            # 正样本：生成单调递增序列
            seq = np.sort(np.random.randint(0, vocab_size, seq_len))
            label = 1
        else:
            # 负样本：随机序列（大概率不单调）
            seq = np.random.randint(0, vocab_size, seq_len)
            label = 0

        # 加入 [CLS] token
        full_seq = np.concatenate([[CLS_IDX], seq])
        # 嵌入
        X_embed = embedding[full_seq]  # (seq_len+1, embed_dim)
        X_data.append(X_embed)
        y_data.append(label)

    return np.array(X_data), np.array(y_data), embedding

embed_dim = 16
X_all, y_all, embedding = make_monotone_dataset(n_samples=400, embed_dim=embed_dim)

# 划分训练/测试集
n_train = 320
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_test, y_test = X_all[n_train:], y_all[n_train:]

print(f"数据集：训练 {n_train} 样本，测试 {len(X_test)} 样本")
print(f"正样本比例：{y_all.mean():.2%}")
print(f"输入形状：{X_train.shape}  （batch, seq_len+1, embed_dim）")

# 简单的 Transformer 分类器
def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

class TransformerClassifier:
    """
    用 Transformer Encoder 做序列二分类
    取 [CLS] 位置的输出，接一个线性分类头
    """
    def __init__(self, d_model, n_heads, n_layers):
        self.encoder = TransformerEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        # 分类头
        scale = 1.0 / np.sqrt(d_model)
        self.W_cls = np.random.randn(d_model, 1) * scale
        self.b_cls = np.zeros(1)

    def forward(self, X):
        """X: (batch, seq+1, d_model)"""
        enc_out, weights = self.encoder.forward(X)
        # 取 [CLS] 位置（第0个位置）的输出
        cls_out = enc_out[:, 0, :]  # (batch, d_model)
        logit = cls_out @ self.W_cls + self.b_cls  # (batch, 1)
        prob = sigmoid(logit.ravel())  # (batch,)
        return prob, enc_out, weights

# 训练（简化版：只用分类头参数，Encoder 固定 — 展示流程）
model = TransformerClassifier(d_model=embed_dim, n_heads=4, n_layers=2)

# 仅训练分类头（Encoder 权重固定，减少计算量）
train_losses = []
n_epochs = 30
batch_size = 32
lr = 0.01

print("\n训练中（仅展示分类头训练）...")
for epoch in range(n_epochs):
    idx = np.random.permutation(n_train)
    epoch_loss = []

    for start in range(0, n_train, batch_size):
        batch_idx = idx[start:start+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        # 前向
        probs, enc_out, _ = model.forward(X_batch)
        eps = 1e-8
        loss = -np.mean(y_batch * np.log(probs + eps) + (1 - y_batch) * np.log(1 - probs + eps))
        epoch_loss.append(loss)

        # 反向（只更新分类头）
        cls_out = enc_out[:, 0, :]
        d_prob = (probs - y_batch) / len(y_batch)  # BCE + sigmoid 梯度
        d_W = cls_out.T @ d_prob.reshape(-1, 1)
        d_b = d_prob.sum()
        model.W_cls -= lr * d_W
        model.b_cls -= lr * d_b

    train_losses.append(np.mean(epoch_loss))

# 评估
def accuracy(probs, labels, threshold=0.5):
    return np.mean((probs >= threshold) == labels)

test_probs, _, _ = model.forward(X_test)
train_probs, _, _ = model.forward(X_train)
print(f"训练准确率：{accuracy(train_probs, y_train):.2%}")
print(f"测试准确率：{accuracy(test_probs, y_test):.2%}")

# ============================================================
# Part 6: 可视化
# ============================================================
print("\nPart 6: 可视化 Transformer 组件")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Transformer Encoder 完整解析', fontsize=14)

# 1. LayerNorm vs BatchNorm
ax = axes[0][0]
np.random.seed(0)
x_data = np.random.randn(8, 5, 32)  # (batch, seq, d_model)
# BatchNorm（在 batch 维度计算）
bn_means = x_data.mean(axis=0).mean(axis=0)  # (d_model,)
bn_stds = x_data.std(axis=0).mean(axis=0)
# LayerNorm（在特征维度计算）
ln_means = x_data.mean(axis=-1)  # (batch, seq)
ln_stds = x_data.std(axis=-1)

ax.bar(['BN均值\n(跨batch)', 'LN均值\n(跨特征)'],
       [np.abs(bn_means).mean(), np.abs(ln_means).mean()],
       color=['orange', 'steelblue'], alpha=0.7)
ax.set_ylabel('绝对均值')
ax.set_title('BatchNorm vs LayerNorm\n归一化方向对比')
ax.grid(True, alpha=0.3, axis='y')
ax.text(0, np.abs(bn_means).mean()+0.01, '跨batch统计', ha='center', fontsize=8)
ax.text(1, np.abs(ln_means).mean()+0.01, '跨特征统计', ha='center', fontsize=8)

# 2. 位置编码可视化
ax = axes[0][1]
PE_vis = positional_encoding(seq_len=20, d_model=32)
im = ax.imshow(PE_vis[0], cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('特征维度')
ax.set_ylabel('序列位置')
ax.set_title('位置编码（sin/cos）\n每行=一个位置的编码向量')
plt.colorbar(im, ax=ax, fraction=0.046)
ax.text(1, 22, '低频（慢变化）→', fontsize=7, color='black')

# 3. 训练损失曲线
ax = axes[0][2]
ax.plot(train_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE 损失')
ax.set_title(f'序列分类训练曲线\n测试准确率={accuracy(test_probs, y_test):.1%}')
ax.grid(True, alpha=0.3)

# 4. 注意力权重热力图（第1层第1头）
_, _, attn_weights = model.forward(X_test[:1])
ax = axes[1][0]
w = attn_weights[0][0, 0]  # 第1层第1头第1个样本
words = ['[CLS]'] + [f'x{i}' for i in range(w.shape[0]-1)]
im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max())
ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels(words, rotation=45, fontsize=8)
ax.set_yticklabels(words, fontsize=8)
ax.set_title('注意力权重热力图\n（第1层第1头）')
plt.colorbar(im, ax=ax, fraction=0.046)

# 5. Encoder 架构图（文字说明）
ax = axes[1][1]
ax.axis('off')
arch_text = """
Transformer Encoder Layer 结构：

  输入 X ────────────────────────┐
      ↓                          │
  MultiHeadAttention(X)          │ 残差
      ↓                          │
      + ←────────────────────────┘
      ↓
  LayerNorm
      ↓ ─────────────────────────┐
  FeedForward                    │ 残差
      ↓                          │
      + ←────────────────────────┘
      ↓
  LayerNorm
      ↓
  输出（形状不变）

N 层堆叠 = 完整 Encoder
"""
ax.text(0.05, 0.95, arch_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Encoder Layer 完整结构')

# 6. FFN 维度扩展示意
ax = axes[1][2]
d_model_val = 64
d_ff_val = 256
layers_h = [('输入', d_model_val), ('FFN扩展', d_ff_val), ('FFN压缩', d_model_val)]
colors = ['lightblue', 'orange', 'lightblue']
y_pos = [3, 2, 1]
for (name, dim), y, c in zip(layers_h, y_pos, colors):
    width = dim / d_ff_val
    ax.barh([y], [width], height=0.4, left=[0.5-width/2], color=c, edgecolor='black')
    ax.text(0.5, y, f'{name}\n({dim}维)', ha='center', va='center', fontsize=9)
    if y > 1:
        ax.annotate('', xy=(0.5, y-0.3), xytext=(0.5, y+0.3-0.6),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.set_xlim(0, 1)
ax.set_ylim(0.4, 3.6)
ax.axis('off')
ax.set_title(f'FFN 维度扩展（{d_model_val}→{d_ff_val}→{d_model_val}）\n中间层信息更丰富')

plt.tight_layout()
plt.savefig('05_transformer_attention/transformer_encoder.png', dpi=80, bbox_inches='tight')
print("图片已保存：05_transformer_attention/transformer_encoder.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【LayerNorm 位置】
   原始 Transformer（Post-LN）：X + SubLayer(X) → LayerNorm
   现代 Transformer（Pre-LN）：X + SubLayer(LayerNorm(X))
   Pre-LN 为什么训练更稳定？
   （提示：Pre-LN 保证了残差路径上的梯度不被归一化层影响）

2. 【位置编码的相对性】
   原始位置编码是绝对位置。
   RoPE（旋转位置编码）用旋转矩阵实现相对位置感知：
     让 q_pos 和 k_pos 的点积只依赖 (pos_q - pos_k)
   这为什么对泛化到更长序列有帮助？

3. 【FFN 的直觉】
   有研究表明，Transformer 的 FFN 层像是"键值记忆"：
   - FFN 的第一层（d_model→d_ff）是"键"：匹配输入特征
   - FFN 的第二层（d_ff→d_model）是"值"：输出对应的知识
   这和注意力的 K-V 机制有什么相似之处？
   （提示：ff_out = softmax(x @ W1) @ W2 和 attention 的形式）
""")
