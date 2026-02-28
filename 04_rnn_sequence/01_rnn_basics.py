"""
==============================================================
第4章 第1节：RNN 基础 —— 有记忆的神经网络
==============================================================

【为什么需要它？】
普通的全连接网络和 CNN 是"无记忆"的：
  - 每个输入独立处理，不考虑前后顺序
  - 对"今天的股价取决于昨天"这种序列关系，MLP 无能为力

现实中大量数据是序列（时间相关的）：
  - 文本："我爱" → 下一个词是？（取决于上下文！）
  - 股价：今天的价格依赖历史趋势
  - 语音：音频波形的时间序列
  - 视频：帧之间有关联

RNN 的解决方案：网络有"隐状态"（记忆），每一步都依赖上一步

【生活类比】
读一本书：
  读每个字时，你不只看当前这个字，还记得之前读了什么。
  隐状态 h_t = 你读到第 t 个字时脑子里保留的"记忆"。
  每读一个字：记忆更新 = 旧记忆 × 遗忘 + 新信息

RNN = 每步都有"记忆状态"的循环网络

【存在理由】
解决问题：传统网络无法处理变长序列和时间依赖
核心思想：循环隐状态 h_t = f(h_{t-1}, x_t)，保持历史信息
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: RNN 单步计算
# ============================================================
print("=" * 50)
print("Part 1: RNN 的计算方式")
print("=" * 50)

"""
RNN 在每个时间步 t 的计算：
  h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
  y_t = W_hy * h_t + b_y

  h_t：当前隐状态（"记忆"）
  x_t：当前输入
  h_{t-1}：上一步隐状态

参数在所有时间步共享！（参数共享 = 权重复用，类似卷积）
  - W_hh：隐状态到隐状态的权重（"记忆的转移"）
  - W_xh：输入到隐状态的权重（"新信息的影响"）
  - W_hy：隐状态到输出的权重

关键：W_hh、W_xh、W_hy 对所有时间步都是同一套参数！
  → 不管序列多长，参数数量不变
  → 可以处理任意长度的序列
"""

class RNNCell:
    """单个 RNN 时间步的计算"""

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # 参数初始化（正交初始化对 RNN 更稳定）
        scale = 0.1
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)

        self.W_hy = np.random.randn(hidden_size, output_size) * scale
        self.b_y = np.zeros(output_size)

        self.cache = None

    def forward(self, x, h_prev):
        """
        x: (batch, input_size) —— 当前时间步的输入
        h_prev: (batch, hidden_size) —— 上一步的隐状态

        h_t = tanh(x @ W_xh + h_prev @ W_hh + b_h)
        """
        # 线性变换（输入 + 上一步记忆）
        z = x @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h_t = np.tanh(z)  # 激活，输出 (-1, 1)
        y_t = h_t @ self.W_hy + self.b_y  # 输出层

        self.cache = (x, h_prev, z, h_t)
        return h_t, y_t

    def backward(self, d_h_t, d_y_t):
        """反向传播：计算梯度，传回 d_h_prev"""
        x, h_prev, z, h_t = self.cache

        # 输出层的梯度
        self.d_Why = h_t.T @ d_y_t
        self.d_by = d_y_t.sum(axis=0)
        d_h_from_y = d_y_t @ self.W_hy.T

        # 合并来自下一步 h_t 的梯度 + 来自输出 y_t 的梯度
        d_h = d_h_t + d_h_from_y

        # tanh 的反向：tanh'(z) = 1 - tanh(z)²
        d_z = d_h * (1 - h_t ** 2)

        # 参数梯度
        self.d_Wxh = x.T @ d_z
        self.d_Whh = h_prev.T @ d_z
        self.d_bh = d_z.sum(axis=0)

        # 传回上一步的隐状态梯度
        d_h_prev = d_z @ self.W_hh.T
        d_x = d_z @ self.W_xh.T

        return d_h_prev, d_x

# ============================================================
# Part 2: 完整 RNN（多步展开）
# ============================================================
print("Part 2: 完整 RNN —— 时间步展开")
print("=" * 50)

"""
RNN 的训练 = BPTT（Back-Propagation Through Time，时间反向传播）

前向：把序列展开，一步一步计算
  h_0 → h_1 → h_2 → ... → h_T

反向：从最后一步开始，反向传播穿越所有时间步
  dL/dh_T → dL/dh_{T-1} → ... → dL/dh_1

梯度在时间步上累积，然后更新共享参数。
"""

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size, output_size)
        self.cache_sequence = None

    def forward(self, X_seq, y_seq=None):
        """
        X_seq: (batch, seq_len, input_size)
        y_seq: (batch, seq_len)  可选的标签
        """
        batch, seq_len, input_size = X_seq.shape
        h = np.zeros((batch, self.hidden_size))  # 初始隐状态

        outputs = []
        self.cache_sequence = []

        for t in range(seq_len):
            x_t = X_seq[:, t, :]  # 当前时间步输入
            h, y_t = self.cell.forward(x_t, h)
            outputs.append(y_t)
            self.cache_sequence.append(self.cell.cache)

        outputs = np.stack(outputs, axis=1)  # (batch, seq_len, output_size)

        if y_seq is not None:
            # 多步预测损失（这里用 MSE 举例）
            loss = np.mean((outputs - y_seq) ** 2)
            return outputs, loss

        return outputs

    def backward(self, d_outputs):
        """
        d_outputs: (batch, seq_len, output_size)
        梯度从最后一个时间步开始反向传播
        """
        seq_len = d_outputs.shape[1]
        d_h_next = np.zeros((d_outputs.shape[0], self.hidden_size))

        # 累积梯度（所有时间步共享参数）
        self.d_Wxh = np.zeros_like(self.cell.W_xh)
        self.d_Whh = np.zeros_like(self.cell.W_hh)
        self.d_bh = np.zeros_like(self.cell.b_h)
        self.d_Why = np.zeros_like(self.cell.W_hy)
        self.d_by = np.zeros_like(self.cell.b_y)

        for t in reversed(range(seq_len)):
            # 恢复该时间步的缓存
            self.cell.cache = self.cache_sequence[t]

            d_h_next, _ = self.cell.backward(d_h_next, d_outputs[:, t, :])

            # 累积所有时间步的参数梯度
            self.d_Wxh += self.cell.d_Wxh
            self.d_Whh += self.cell.d_Whh
            self.d_bh += self.cell.d_bh
            self.d_Why += self.cell.d_Why
            self.d_by += self.cell.d_by

        # 梯度裁剪（防止梯度爆炸）
        max_norm = 5.0
        for grad in [self.d_Wxh, self.d_Whh, self.d_bh, self.d_Why, self.d_by]:
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                grad *= max_norm / norm

    def update(self, lr):
        self.cell.W_xh -= lr * self.d_Wxh
        self.cell.W_hh -= lr * self.d_Whh
        self.cell.b_h -= lr * self.d_bh
        self.cell.W_hy -= lr * self.d_Why
        self.cell.b_y -= lr * self.d_by

# ============================================================
# Part 3: 梯度消失的可视化
# ============================================================
print("Part 3: RNN 的梯度消失 —— 长序列的痛点")
print("=" * 50)

"""
RNN 的核心问题：梯度在时间步上连续相乘

从最后时间步反传到第 t 步的梯度：
  dL/dh_t = dL/dh_T × Π_{k=t}^{T} dh_k/dh_{k-1}
           = dL/dh_T × Π_{k=t}^{T} (tanh'(z_k) × W_hh)

  如果 |W_hh 的特征值| < 1 且 |tanh'| < 1
  → T-t 次相乘后，梯度趋近于 0
  → 网络忘记了很早之前的信息！

如何量化？
  对于序列长度 T，第 0 步的梯度幅度：
  ≈ (σ_max(W_hh) × max(tanh'(z)))^T
  当 T = 50，如果这个值 = 0.9，则 0.9^50 ≈ 0.005（几乎0！）
"""

# 模拟梯度流动
T_values = range(1, 51)
wh_eigenvalues = [0.7, 0.9, 1.0, 1.1]
colors = ['red', 'orange', 'green', 'blue']
labels = ['|λ|=0.7（消失）', '|λ|=0.9（慢消失）', '|λ|=1.0（稳定）', '|λ|=1.1（爆炸）']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for eig, color, label in zip(wh_eigenvalues, colors, labels):
    grad_magnitudes = [eig**t * 0.25**t for t in T_values]  # 乘以 tanh'≈0.25 估计
    ax.semilogy(list(T_values), grad_magnitudes, color=color, linewidth=2, label=label)

ax.axhline(1e-3, color='gray', linestyle='--', label='梯度消失阈值')
ax.set_xlabel('时间距离（当前步到最早步的步数）')
ax.set_ylabel('梯度幅度（对数）')
ax.set_title('RNN 梯度随时间步距离的衰减\n（tanh 版，W_hh 的特征值不同）')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 实际展示：长序列任务中 RNN 的记忆能力
# 任务：回忆序列开头的信息（"看到1就记住，50步后告诉我"）
ax = axes[1]
# 生成"复制任务"：输入序列开头是1或0，其余是噪声，要求输出开头的值
memory_lengths = [5, 10, 20, 50]
rnn_success = []

for mem_len in memory_lengths:
    # 简单模拟：能记住 mem_len 步
    # 实际 RNN 能力随序列长度下降
    estimated_acc = max(0.5, 1.0 - 0.02 * mem_len)  # 简化模型
    rnn_success.append(estimated_acc)

ax.bar(range(len(memory_lengths)), rnn_success, color=['green' if a > 0.7 else 'red' for a in rnn_success])
ax.set_xticks(range(len(memory_lengths)))
ax.set_xticklabels([f'{m}步' for m in memory_lengths])
ax.set_ylabel('复制任务准确率（估计）')
ax.set_title('RNN 的记忆能力随序列长度下降\n→ 这就是为什么需要 LSTM！')
ax.axhline(0.7, color='gray', linestyle='--', label='可接受阈值')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('04_rnn_sequence/rnn_gradient_vanishing.png', dpi=100, bbox_inches='tight')
print("图片已保存：04_rnn_sequence/rnn_gradient_vanishing.png")
plt.show()

# ============================================================
# Part 4: 用 RNN 生成文本（字符级）
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 字符级语言模型 —— 用 RNN 生成文本")
print("=" * 50)

"""
任务：给 RNN 一段文本，让它学会"下一个字符是什么"
  输入：当前字符序列
  输出：下一个字符（预测）

这是最简单的语言模型！
"""

text = "hello world this is a simple rnn example for learning deep learning"
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

vocab_size = len(chars)
seq_len = 20
hidden_size = 64

print(f"训练文本：'{text}'")
print(f"词汇表大小：{vocab_size} 个字符")
print(f"字符列表：{chars}")

# 准备数据（one-hot编码）
def text_to_onehot(text, seq_len):
    """把文本切成序列，one-hot 编码"""
    X_list, y_list = [], []
    for i in range(len(text) - seq_len):
        x_chars = text[i:i+seq_len]
        y_chars = text[i+1:i+seq_len+1]

        x_onehot = np.zeros((seq_len, vocab_size))
        y_indices = np.zeros(seq_len, dtype=int)

        for t, (cx, cy) in enumerate(zip(x_chars, y_chars)):
            x_onehot[t, char_to_idx[cx]] = 1.0
            y_indices[t] = char_to_idx[cy]

        X_list.append(x_onehot)
        y_list.append(y_indices)

    return np.array(X_list), np.array(y_list)  # (N, seq_len, vocab_size), (N, seq_len)

X_data, y_data = text_to_onehot(text, seq_len)
print(f"数据形状：X={X_data.shape}, y={y_data.shape}")

# 简化的字符级 RNN（不用上面的通用类，方便理解）
np.random.seed(0)
W_xh = np.random.randn(vocab_size, hidden_size) * 0.01
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
b_h = np.zeros(hidden_size)
W_hy = np.random.randn(hidden_size, vocab_size) * 0.01
b_y = np.zeros(vocab_size)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def rnn_forward_char(x_seq, h_init):
    """字符级 RNN 前向传播"""
    h = h_init
    hs, zs = [h], []
    ys = []

    for t in range(seq_len):
        z = x_seq[t] @ W_xh + h @ W_hh + b_h
        h = np.tanh(z)
        y = h @ W_hy + b_y
        hs.append(h)
        zs.append(z)
        ys.append(y)

    return np.array(ys), np.array(hs), np.array(zs)

def sample_text(seed_char, n_chars=40, temperature=1.0):
    """从训练好的模型生成文本"""
    h = np.zeros(hidden_size)
    result = seed_char

    # 当前字符
    x = np.zeros(vocab_size)
    x[char_to_idx[seed_char]] = 1.0

    for _ in range(n_chars):
        z = x @ W_xh + h @ W_hh + b_h
        h = np.tanh(z)
        y = h @ W_hy + b_y

        # Temperature 采样（temperature=1=正常，<1=更确定，>1=更随机）
        probs = softmax(y / temperature)
        next_char_idx = np.random.choice(vocab_size, p=probs)
        next_char = idx_to_char[next_char_idx]

        result += next_char
        x = np.zeros(vocab_size)
        x[next_char_idx] = 1.0

    return result

# 训练几个 epoch
print("\n训练字符级 RNN...")
losses = []
lr = 0.01

for epoch in range(100):
    epoch_loss = 0
    for xi, yi in zip(X_data, y_data):
        # 前向
        ys, hs, zs = rnn_forward_char(xi, np.zeros(hidden_size))
        probs = softmax(ys)

        # 损失
        loss = -np.mean(np.log(probs[range(seq_len), yi] + 1e-8))
        epoch_loss += loss

        # 简化反向传播（BPTT）
        d_hy = probs.copy()
        d_hy[range(seq_len), yi] -= 1
        d_hy /= seq_len

        dW_hy = hs[1:].reshape(-1, hidden_size).T @ d_hy.reshape(-1, vocab_size)
        db_y = d_hy.sum(axis=0)
        d_h = d_hy @ W_hy.T

        dW_xh = np.zeros_like(W_xh)
        dW_hh = np.zeros_like(W_hh)
        db_h = np.zeros_like(b_h)
        d_h_next = np.zeros(hidden_size)

        for t in reversed(range(seq_len)):
            dh_t = d_h[t] + d_h_next
            dz_t = dh_t * (1 - hs[t+1]**2)
            dW_xh += np.outer(xi[t], dz_t)
            dW_hh += np.outer(hs[t], dz_t)
            db_h += dz_t
            d_h_next = dz_t @ W_hh.T

        # 梯度裁剪
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -5, 5, out=grad)

        W_xh -= lr * dW_xh
        W_hh -= lr * dW_hh
        b_h -= lr * db_h
        W_hy -= lr * dW_hy
        b_y -= lr * db_y

    avg_loss = epoch_loss / len(X_data)
    losses.append(avg_loss)

    if epoch % 20 == 0:
        sample = sample_text('h', n_chars=30)
        print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, 生成：'{sample}'")

print(f"\n最终生成（temperature=0.5）：'{sample_text('h', n_chars=50, temperature=0.5)}'")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【梯度裁剪实验】
   把梯度裁剪关掉（注释掉 np.clip 那行）。
   训练 100 个 epoch，观察 loss 是否出现 NaN 或爆炸。
   这说明了梯度裁剪的重要性。

2. 【隐状态尺寸实验】
   把 hidden_size 从 64 改为 16 和 256。
   - hidden_size=16：总参数量是多少？能学会吗？
   - hidden_size=256：总参数量是多少？会过拟合吗？

3. 【双向 RNN 直觉】
   普通 RNN 只看"之前的"信息。
   双向 RNN（BiRNN）同时从前向后和从后向前各运行一次。
   在"填空"任务中（"The [MASK] sat on the mat"，[MASK] 是 cat），
   为什么双向更好？（提示：BERT 就是双向 Transformer！）
""")
