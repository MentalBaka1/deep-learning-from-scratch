"""
==============================================================
第4章 第2节：LSTM/GRU —— 门控记忆管理
==============================================================

【为什么需要它？】
RNN 的致命弱点：梯度消失 → 记不住长距离依赖
  "The cat, which ate the fish that was caught yesterday, was full."
  （猫，已经吃了昨天抓的鱼，吃饱了。）
  "was full" 的主语是 "cat"，但两者相隔很远。
  普通 RNN 会在中间内容干扰下忘记 "cat"。

1997年，Hochreiter 和 Schmidhuber 提出 LSTM，
通过门控机制解决了长期依赖问题。

【生活类比】
RNN 的记忆 = 写在沙子上，浪一来就冲走了
LSTM 的记忆 = 有一个笔记本（cell state），可以主动决定：

  遗忘门（Forget Gate）= "这条信息我不需要了，擦掉"
  输入门（Input Gate）  = "这条新信息很重要，记下来"
  输出门（Output Gate） = "现在要输出什么？"

  笔记本（cell state c_t）在整个序列中流动，信息不会被梯度消失冲走

【存在理由】
解决问题：RNN 因梯度消失无法记住长距离依赖
核心思想：门控机制主动控制信息的存储、遗忘和输出
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: LSTM 单步计算
# ============================================================
print("=" * 50)
print("Part 1: LSTM 的四个门")
print("=" * 50)

"""
LSTM 在每个时间步 t 的计算（四个"门"）：

  拼接输入：combined = [h_{t-1}, x_t]（一个向量）

  遗忘门 f_t = sigmoid(W_f * combined + b_f)
    → 决定"忘掉"cell state 的哪些部分（0=完全忘，1=完全记）

  输入门 i_t = sigmoid(W_i * combined + b_i)
    → 决定"写入"哪些新信息（0=不写，1=完全写）

  候选值 g_t = tanh(W_g * combined + b_g)
    → 准备写入的新信息（-1到1）

  输出门 o_t = sigmoid(W_o * combined + b_o)
    → 决定输出哪些信息给隐状态 h_t

  cell 更新：c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    （⊙ 是逐元素乘法）
    → f_t 控制保留多少旧记忆，i_t*g_t 是新信息

  隐状态：h_t = o_t ⊙ tanh(c_t)
    → 输出门控制从 cell state 中读出多少
"""

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        # 四个门的参数（合并为一个大矩阵，提高效率）
        # W: (combined_size, 4 * hidden_size)
        # 顺序：[遗忘门|输入门|候选值|输出门]
        scale = 1.0 / np.sqrt(combined_size)
        self.W = np.random.randn(combined_size, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)
        # 初始化遗忘门偏置为1（更容易记住信息，实践中有效）
        self.b[hidden_size:2*hidden_size] = 1.0  # 遗忘门偏置=1

        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """
        x: (batch, input_size)
        h_prev, c_prev: (batch, hidden_size)
        """
        batch = x.shape[0]

        # 拼接输入和上一步隐状态
        combined = np.concatenate([h_prev, x], axis=1)  # (batch, combined_size)

        # 一次矩阵乘法计算所有四个门
        gates = combined @ self.W + self.b  # (batch, 4*hidden_size)

        H = self.hidden_size
        # 分割成四个门
        f = sigmoid(gates[:, :H])         # 遗忘门
        i = sigmoid(gates[:, H:2*H])      # 输入门
        g = np.tanh(gates[:, 2*H:3*H])    # 候选值
        o = sigmoid(gates[:, 3*H:])       # 输出门

        # Cell state 更新（核心！）
        c_t = f * c_prev + i * g

        # 隐状态
        h_t = o * np.tanh(c_t)

        self.cache = (combined, f, i, g, o, c_prev, c_t, h_t, gates)
        return h_t, c_t

    def backward(self, d_h, d_c):
        """
        LSTM 的反向传播
        d_h: 来自当前步输出的梯度
        d_c: 来自下一步 cell state 的梯度
        """
        combined, f, i, g, o, c_prev, c_t, h_t, gates = self.cache
        H = self.hidden_size

        # 输出门和 tanh(c_t)
        tanh_c = np.tanh(c_t)
        d_o = d_h * tanh_c
        d_c += d_h * o * (1 - tanh_c**2)  # tanh(c_t) 的梯度

        # Cell state 分支
        d_f = d_c * c_prev
        d_i = d_c * g
        d_g = d_c * i
        d_c_prev = d_c * f

        # 通过 gate 函数反向
        d_gates = np.zeros_like(gates)
        d_gates[:, :H] = d_f * f * (1 - f)         # sigmoid'(f) = f*(1-f)
        d_gates[:, H:2*H] = d_i * i * (1 - i)
        d_gates[:, 2*H:3*H] = d_g * (1 - g**2)      # tanh'(g) = 1-g²
        d_gates[:, 3*H:] = d_o * o * (1 - o)

        # 参数梯度
        self.dW = combined.T @ d_gates
        self.db = d_gates.sum(axis=0)

        # 传回上一步的梯度
        d_combined = d_gates @ self.W.T
        d_h_prev = d_combined[:, :H]
        d_x = d_combined[:, H:]

        return d_h_prev, d_x, d_c_prev

# ============================================================
# Part 2: LSTM 完整序列处理
# ============================================================
print("Part 2: LSTM 完整序列 + 梯度检验")
print("=" * 50)

class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros(output_size)

    def forward(self, X_seq):
        """
        X_seq: (batch, seq_len, input_size)
        返回: (batch, seq_len, output_size)
        """
        batch, seq_len, _ = X_seq.shape
        h = np.zeros((batch, self.hidden_size))
        c = np.zeros((batch, self.hidden_size))

        outputs = []
        self.seq_cache = []

        for t in range(seq_len):
            h, c = self.cell.forward(X_seq[:, t, :], h, c)
            y = h @ self.W_out + self.b_out
            outputs.append(y)
            self.seq_cache.append(self.cell.cache)

        return np.stack(outputs, axis=1)

# ============================================================
# Part 3: GRU —— LSTM 的简化版
# ============================================================
print("Part 3: GRU —— 两个门的简化 LSTM")
print("=" * 50)

"""
GRU（Gated Recurrent Unit，门控循环单元）：
  比 LSTM 少一个门，性能相近，参数更少

  更新门 z_t = sigmoid(W_z * [h_{t-1}, x_t])
    → 控制"更新"多少（结合了遗忘门和输入门）

  重置门 r_t = sigmoid(W_r * [h_{t-1}, x_t])
    → 控制"重置"历史隐状态的程度

  候选隐状态 h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t])
    → 基于部分历史的新候选

  新隐状态 h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    → 线性插值：保留多少旧的，接受多少新的
"""

class GRUCell:
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size

        scale = 1.0 / np.sqrt(combined_size)
        # 更新门和重置门的参数
        self.W_z = np.random.randn(combined_size, hidden_size) * scale
        self.b_z = np.zeros(hidden_size)
        self.W_r = np.random.randn(combined_size, hidden_size) * scale
        self.b_r = np.zeros(hidden_size)
        # 候选隐状态的参数
        self.W_h = np.random.randn(combined_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)

    def forward(self, x, h_prev):
        combined = np.concatenate([h_prev, x], axis=1)

        z = sigmoid(combined @ self.W_z + self.b_z)   # 更新门
        r = sigmoid(combined @ self.W_r + self.b_r)   # 重置门

        # 候选：先"重置"历史，再计算
        combined_reset = np.concatenate([r * h_prev, x], axis=1)
        h_tilde = np.tanh(combined_reset @ self.W_h + self.b_h)

        # 新隐状态：插值
        h_t = (1 - z) * h_prev + z * h_tilde

        return h_t

# ============================================================
# Part 4: LSTM vs GRU vs RNN 参数数量对比
# ============================================================
input_size = 10
hidden_size = 64
output_size = 1

rnn_params = (input_size + hidden_size) * hidden_size + hidden_size + hidden_size * output_size
lstm_params = (input_size + hidden_size) * 4 * hidden_size + 4 * hidden_size + hidden_size * output_size
gru_params = (input_size + hidden_size) * 3 * hidden_size + 3 * hidden_size + hidden_size * output_size

print(f"参数数量对比（input={input_size}, hidden={hidden_size}）：")
print(f"  RNN:  {rnn_params:,} 参数")
print(f"  GRU:  {gru_params:,} 参数  （比 RNN 多 {gru_params/rnn_params:.1f}x）")
print(f"  LSTM: {lstm_params:,} 参数  （比 RNN 多 {lstm_params/rnn_params:.1f}x）")

# ============================================================
# Part 5: 可视化 LSTM 门的行为
# ============================================================
print("\nPart 4: 可视化 LSTM 的门控行为")
print("=" * 50)

"""
让我们观察 LSTM 在不同输入下，各个门的激活情况
这帮助理解"门"在做什么
"""

# 手动设置一个演示用的 LSTM
demo_lstm = LSTMCell(input_size=1, hidden_size=8)

# 生成一个有明显模式的序列：0...0 1 0...0（在某个位置有脉冲）
seq_len = 30
pulse_pos = 10
x_seq = np.zeros((1, seq_len, 1))
x_seq[0, pulse_pos, 0] = 5.0  # 在位置10有一个强信号

h = np.zeros((1, 8))
c = np.zeros((1, 8))

forget_gates = []
input_gates = []
output_gates = []
cell_states = []
hidden_states = []

for t in range(seq_len):
    h, c = demo_lstm.forward(x_seq[0, t:t+1, :], h, c)
    combined, f, i, g, o, c_prev, c_t, h_t, gates = demo_lstm.cache
    forget_gates.append(f[0].mean())
    input_gates.append(i[0].mean())
    output_gates.append(o[0].mean())
    cell_states.append(c[0].mean())
    hidden_states.append(np.linalg.norm(h[0]))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('LSTM 门控行为可视化（脉冲在位置10）', fontsize=13)

t_range = range(seq_len)

ax = axes[0][0]
ax.plot(t_range, forget_gates, 'r-', linewidth=2, label='遗忘门（均值）')
ax.axvline(pulse_pos, color='gray', linestyle='--', alpha=0.5, label='脉冲位置')
ax.set_title('遗忘门：接近1=保留记忆，接近0=忘记')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

ax = axes[0][1]
ax.plot(t_range, input_gates, 'g-', linewidth=2, label='输入门（均值）')
ax.axvline(pulse_pos, color='gray', linestyle='--', alpha=0.5, label='脉冲位置')
ax.set_title('输入门：接近1=写入新信息，接近0=不写')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

ax = axes[1][0]
ax.plot(t_range, cell_states, 'b-', linewidth=2, label='Cell State（均值）')
ax.axvline(pulse_pos, color='gray', linestyle='--', alpha=0.5, label='脉冲位置')
ax.set_title('Cell State：脉冲后状态改变，之后保持（记忆！）')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1][1]
ax.plot(t_range, hidden_states, 'purple', linewidth=2, label='隐状态范数')
ax.axvline(pulse_pos, color='gray', linestyle='--', alpha=0.5, label='脉冲位置')
ax.set_title('隐状态：脉冲后激活，然后根据输出门控制')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_rnn_sequence/lstm_gates.png', dpi=100, bbox_inches='tight')
print("图片已保存：04_rnn_sequence/lstm_gates.png")
plt.show()

print("\n关键观察：")
print("  在脉冲位置（t=10）：输入门打开（允许写入），cell state 改变")
print("  之后：遗忘门保持高值（继续记着），cell state 保持稳定")
print("  这就是 LSTM 的'长期记忆'！")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【LSTM 梯度流动】
   为什么 LSTM 的梯度消失问题比 RNN 轻？
   提示：在 cell state 的更新公式 c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t 中，
   对 c_{t-1} 求导得到什么？
   当 f_t ≈ 1（遗忘门接近全开）时，梯度如何流动？

2. 【门控效果实验】
   修改 LSTMCell，手动将遗忘门设置为 f_t = 0（完全遗忘）。
   在一个长序列上，查看隐状态是否还能保持之前的信息。
   再试试 f_t = 1（完全不忘）。

3. 【GRU 和 LSTM 的比较】
   GRU 没有独立的 cell state（c_t），只有 h_t。
   LSTM 有两个"记忆"：h_t（工作记忆）和 c_t（长期记忆）。
   在实践中，什么场景下 GRU 可能优于 LSTM？
   （提示：参数少 → 数据量少时更好，训练更快）
""")
