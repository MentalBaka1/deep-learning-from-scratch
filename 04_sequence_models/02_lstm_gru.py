"""
====================================================================
第4章 · 第2节 · LSTM 与 GRU：门控记忆管理
====================================================================

【一句话总结】
LSTM 和 GRU 通过"门"机制解决了 RNN 的梯度消失问题——
门决定了哪些信息该记住、哪些该遗忘、哪些该输出。

【为什么深度学习需要这个？】
- RNN 的梯度消失让它无法学习长距离依赖
- LSTM 通过细胞状态（cell state）建立"信息高速公路"
- GRU 是 LSTM 的简化版，参数更少，效果相当
- 虽然 Transformer 已取代 LSTM 成为主流，但门控思想影响深远

【核心概念】

1. LSTM 的四个门
   - 遗忘门（Forget Gate）：f = σ(W_f·[h_{t-1}, x_t] + b_f)
     决定细胞状态中哪些信息要丢弃（0=全忘，1=全记）
   - 输入门（Input Gate）：i = σ(W_i·[h_{t-1}, x_t] + b_i)
     决定哪些新信息要写入细胞状态
   - 候选值：C̃ = tanh(W_C·[h_{t-1}, x_t] + b_C)
     生成候选的新信息
   - 输出门（Output Gate）：o = σ(W_o·[h_{t-1}, x_t] + b_o)
     决定细胞状态的哪些部分要输出

2. 细胞状态更新
   - C_t = f ⊙ C_{t-1} + i ⊙ C̃
   - 先遗忘旧信息（f⊙C），再加入新信息（i⊙C̃）
   - 类比：笔记本——擦掉不需要的，写入新的

3. 隐状态输出
   - h_t = o ⊙ tanh(C_t)
   - 细胞状态经过 tanh 压缩后，由输出门筛选

4. 为什么 LSTM 解决梯度消失？
   - 细胞状态是"信息高速公路"，梯度可以几乎无损地流过
   - ∂C_t/∂C_{t-1} = f_t（遗忘门的值，接近1时梯度几乎不衰减）
   - 类比：ResNet 的跳跃连接，异曲同工

5. GRU（Gated Recurrent Unit）
   - 将 LSTM 的3个门简化为2个：重置门(r)和更新门(z)
   - z = σ(W_z·[h_{t-1}, x_t])  （相当于合并了遗忘门和输入门）
   - r = σ(W_r·[h_{t-1}, x_t])  （决定如何使用历史信息）
   - h̃ = tanh(W·[r⊙h_{t-1}, x_t])
   - h_t = (1-z)⊙h_{t-1} + z⊙h̃
   - 没有细胞状态，直接用隐状态

【前置知识】
第4章第1节 - RNN基础
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示（根据系统情况可能需要调整）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

np.random.seed(42)


# ====================================================================
# 第一部分：LSTM Cell 实现
# ====================================================================
print("=" * 60)
print("第一部分：LSTM Cell 实现——四个门的完整推导")
print("=" * 60)


def sigmoid(x):
    """Sigmoid 激活函数：σ(x) = 1 / (1 + e^{-x})"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    """
    LSTM 单元的完整实现。

    核心思想：
        LSTM 用三个门（遗忘、输入、输出）和一条"细胞状态高速公路"
        来解决 RNN 的梯度消失问题。

    参数说明：
        input_size  : 输入向量的维度
        hidden_size : 隐藏状态/细胞状态的维度

    内部权重：
        W_f, b_f : 遗忘门参数  → 决定丢弃哪些旧信息
        W_i, b_i : 输入门参数  → 决定写入哪些新信息
        W_c, b_c : 候选值参数  → 生成候选的新信息
        W_o, b_o : 输出门参数  → 决定输出细胞状态的哪些部分
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size  # [h_{t-1}, x_t] 的拼接维度

        # Xavier 初始化：让每一层的输入输出方差大致相同
        scale = np.sqrt(2.0 / concat_size)

        # ── 遗忘门（Forget Gate）──
        # f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
        # 输出接近1 → 保留旧信息；接近0 → 遗忘旧信息
        self.W_f = np.random.randn(hidden_size, concat_size) * scale
        self.b_f = np.ones(hidden_size)  # 偏置初始化为1，倾向于"记住"

        # ── 输入门（Input Gate）──
        # i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
        # 控制多少新信息写入细胞状态
        self.W_i = np.random.randn(hidden_size, concat_size) * scale
        self.b_i = np.zeros(hidden_size)

        # ── 候选值（Candidate）──
        # C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
        # 生成可能写入细胞状态的新信息
        self.W_c = np.random.randn(hidden_size, concat_size) * scale
        self.b_c = np.zeros(hidden_size)

        # ── 输出门（Output Gate）──
        # o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
        # 控制细胞状态的哪些部分作为隐状态输出
        self.W_o = np.random.randn(hidden_size, concat_size) * scale
        self.b_o = np.zeros(hidden_size)

    def forward(self, x_t, h_prev, c_prev):
        """
        LSTM 的前向传播（单时间步）。

        参数：
            x_t    : 当前时间步输入，形状 (input_size,)
            h_prev : 上一时间步隐状态，形状 (hidden_size,)
            c_prev : 上一时间步细胞状态，形状 (hidden_size,)

        返回：
            h_t    : 当前隐状态
            c_t    : 当前细胞状态
            gates  : 字典，包含各门的值（用于可视化）
        """
        # 步骤 0：拼接输入和上一隐状态 → [h_{t-1}, x_t]
        concat = np.concatenate([h_prev, x_t])

        # 步骤 1：遗忘门 —— "哪些旧记忆要擦掉？"
        f_t = sigmoid(self.W_f @ concat + self.b_f)

        # 步骤 2：输入门 —— "哪些新信息要写入？"
        i_t = sigmoid(self.W_i @ concat + self.b_i)

        # 步骤 3：候选值 —— "新信息的内容是什么？"
        c_tilde = np.tanh(self.W_c @ concat + self.b_c)

        # 步骤 4：更新细胞状态 —— "擦掉旧的 + 写入新的"
        # C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        # 这就是 LSTM 的"信息高速公路"！
        c_t = f_t * c_prev + i_t * c_tilde

        # 步骤 5：输出门 —— "细胞状态的哪些部分要输出？"
        o_t = sigmoid(self.W_o @ concat + self.b_o)

        # 步骤 6：计算隐状态输出
        # h_t = o_t ⊙ tanh(C_t)
        h_t = o_t * np.tanh(c_t)

        # 保存门的值，用于分析和可视化
        gates = {
            "forget": f_t,     # 遗忘门
            "input": i_t,      # 输入门
            "candidate": c_tilde,  # 候选值
            "output": o_t,     # 输出门
        }
        return h_t, c_t, gates

    def forward_sequence(self, X):
        """
        对整个序列进行前向传播。

        参数：
            X : 输入序列，形状 (seq_len, input_size)

        返回：
            all_h     : 所有时间步的隐状态
            all_c     : 所有时间步的细胞状态
            all_gates : 所有时间步的门值
        """
        seq_len = X.shape[0]
        h_t = np.zeros(self.hidden_size)
        c_t = np.zeros(self.hidden_size)

        all_h, all_c, all_gates = [], [], []
        for t in range(seq_len):
            h_t, c_t, gates = self.forward(X[t], h_t, c_t)
            all_h.append(h_t.copy())
            all_c.append(c_t.copy())
            all_gates.append(gates)

        return np.array(all_h), np.array(all_c), all_gates

    def param_count(self):
        """计算 LSTM 的总参数数量"""
        # 4个门，每个门有 W (hidden_size x concat_size) + b (hidden_size)
        concat_size = self.input_size + self.hidden_size
        per_gate = self.hidden_size * concat_size + self.hidden_size
        return 4 * per_gate


# 演示 LSTM 前向传播
lstm = LSTMCell(input_size=3, hidden_size=4)
print(f"LSTM 参数: input_size={lstm.input_size}, hidden_size={lstm.hidden_size}")
print(f"总参数量: {lstm.param_count()}")
print(f"  每个门: W 形状 ({lstm.hidden_size}, {lstm.input_size + lstm.hidden_size})"
      f" + b 形状 ({lstm.hidden_size},)")
print(f"  4 个门 x ({lstm.hidden_size}x{lstm.input_size + lstm.hidden_size} + "
      f"{lstm.hidden_size}) = {lstm.param_count()}")

# 生成一个简单序列
x_demo = np.random.randn(5, 3)  # 5个时间步，3维输入
h_all, c_all, gates_all = lstm.forward_sequence(x_demo)
print(f"\n输入序列形状: {x_demo.shape}")
print(f"隐状态序列形状: {h_all.shape}")
print(f"细胞状态序列形状: {c_all.shape}")
print(f"时间步 0 的遗忘门值: {gates_all[0]['forget'].round(3)}")
print()


# ====================================================================
# 第二部分：门的可视化——理解每个门在做什么
# ====================================================================
print("=" * 60)
print("第二部分：门的可视化——理解每个门在做什么")
print("=" * 60)

# 构造一个有特征的输入序列：正弦波 + 突变信号
# 这样能更直观地看到门如何响应不同模式的输入
seq_len = 50
t_axis = np.arange(seq_len)

# 输入信号：一个正弦波，在 t=20 处有一个突然的脉冲
signal = np.sin(2 * np.pi * t_axis / 20)  # 基础正弦波
signal[20:23] = 3.0  # 在 t=20 处加入一个脉冲
X_signal = signal.reshape(-1, 1)  # (seq_len, 1)

# 用 LSTM 处理这个信号
lstm_viz = LSTMCell(input_size=1, hidden_size=8)
h_viz, c_viz, gates_viz = lstm_viz.forward_sequence(X_signal)

# 提取各门的均值（跨隐藏维度取平均，便于可视化）
forget_vals = np.array([g["forget"].mean() for g in gates_viz])
input_vals = np.array([g["input"].mean() for g in gates_viz])
output_vals = np.array([g["output"].mean() for g in gates_viz])
candidate_vals = np.array([g["candidate"].mean() for g in gates_viz])

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# 输入信号
axes[0].plot(t_axis, signal, "k-", linewidth=2)
axes[0].fill_between(t_axis, signal, alpha=0.2, color="gray")
axes[0].set_ylabel("输入信号", fontsize=11)
axes[0].set_title("LSTM 门控值随时间变化（观察门如何响应输入）", fontsize=14)
axes[0].axvspan(20, 22, alpha=0.3, color="red", label="脉冲区域")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# 遗忘门
axes[1].plot(t_axis, forget_vals, "b-", linewidth=2, label="遗忘门 f")
axes[1].fill_between(t_axis, forget_vals, alpha=0.2, color="blue")
axes[1].set_ylabel("遗忘门 f", fontsize=11)
axes[1].set_ylim(-0.05, 1.05)
axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[1].annotate("接近1 = 保留旧记忆\n接近0 = 遗忘旧记忆",
                 xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
axes[1].grid(True, alpha=0.3)

# 输入门
axes[2].plot(t_axis, input_vals, "g-", linewidth=2, label="输入门 i")
axes[2].fill_between(t_axis, input_vals, alpha=0.2, color="green")
axes[2].set_ylabel("输入门 i", fontsize=11)
axes[2].set_ylim(-0.05, 1.05)
axes[2].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[2].annotate("接近1 = 大量写入新信息\n接近0 = 拒绝写入",
                 xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
axes[2].grid(True, alpha=0.3)

# 输出门
axes[3].plot(t_axis, output_vals, "r-", linewidth=2, label="输出门 o")
axes[3].fill_between(t_axis, output_vals, alpha=0.2, color="red")
axes[3].set_ylabel("输出门 o", fontsize=11)
axes[3].set_ylim(-0.05, 1.05)
axes[3].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[3].annotate("接近1 = 充分输出\n接近0 = 抑制输出",
                 xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
axes[3].grid(True, alpha=0.3)

# 细胞状态
c_mean = c_viz.mean(axis=1)
axes[4].plot(t_axis, c_mean, "m-", linewidth=2, label="细胞状态均值")
axes[4].fill_between(t_axis, c_mean, alpha=0.2, color="purple")
axes[4].set_ylabel("细胞状态 C", fontsize=11)
axes[4].set_xlabel("时间步 t", fontsize=11)
axes[4].annotate("细胞状态 = 长期记忆\n由遗忘门和输入门共同控制",
                 xy=(0.98, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_lstm_gates_visualization.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 02_lstm_gates_visualization.png")
print()


# ====================================================================
# 第三部分：LSTM vs Vanilla RNN 处理长序列
# ====================================================================
print("=" * 60)
print("第三部分：LSTM vs Vanilla RNN 处理长序列")
print("=" * 60)

# 任务：长距离依赖检测
# 序列开头给一个标记（+1 或 -1），中间填充噪声，
# 模型需要在序列末尾"回忆"开头的标记
# RNN 因梯度消失，长序列下会失败；LSTM 能记住


class VanillaRNNCell:
    """
    最基本的 RNN 单元，用于与 LSTM 对比。

    公式：h_t = tanh(W_h · h_{t-1} + W_x · x_t + b)
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        scale_h = np.sqrt(2.0 / hidden_size)
        scale_x = np.sqrt(2.0 / input_size)
        self.W_h = np.random.randn(hidden_size, hidden_size) * scale_h
        self.W_x = np.random.randn(hidden_size, input_size) * scale_x
        self.b = np.zeros(hidden_size)

    def forward_sequence(self, X):
        """对整个序列前向传播，返回所有时间步的隐状态"""
        seq_len = X.shape[0]
        h_t = np.zeros(self.hidden_size)
        all_h = []
        for t in range(seq_len):
            h_t = np.tanh(self.W_h @ h_t + self.W_x @ X[t] + self.b)
            all_h.append(h_t.copy())
        return np.array(all_h)

    def param_count(self):
        """RNN 总参数量"""
        return (self.hidden_size * self.hidden_size +
                self.hidden_size * self.input_size +
                self.hidden_size)


# 长距离依赖实验
print("实验：在序列开头放置标记信号，测试模型能否在末尾保留该信息")
print()

hidden_size = 16
results = {"lengths": [], "rnn_retention": [], "lstm_retention": []}

for seq_length in [10, 30, 50, 100, 200]:
    # 构造序列：开头标记 (+3.0) + 中间噪声 + 观察末尾隐状态
    X_test = np.random.randn(seq_length, 1) * 0.1  # 小噪声
    X_test[0, 0] = 3.0  # 开头的强标记信号

    # Vanilla RNN
    rnn = VanillaRNNCell(input_size=1, hidden_size=hidden_size)
    h_rnn = rnn.forward_sequence(X_test)

    # LSTM
    lstm_test = LSTMCell(input_size=1, hidden_size=hidden_size)
    h_lstm, c_lstm, _ = lstm_test.forward_sequence(X_test)

    # 信息保留度：末尾隐状态的范数 / 第1步隐状态的范数
    # 越接近 1 说明信息保留越好
    rnn_retention = np.linalg.norm(h_rnn[-1]) / (np.linalg.norm(h_rnn[0]) + 1e-10)
    lstm_retention = np.linalg.norm(h_lstm[-1]) / (np.linalg.norm(h_lstm[0]) + 1e-10)

    results["lengths"].append(seq_length)
    results["rnn_retention"].append(rnn_retention)
    results["lstm_retention"].append(lstm_retention)

    print(f"序列长度 {seq_length:>4d}:  RNN 信息保留 = {rnn_retention:.4f},"
          f"  LSTM 信息保留 = {lstm_retention:.4f}")

print()
print("观察：随着序列变长，RNN 的信息保留急剧衰减，LSTM 相对稳定。")
print("这就是 LSTM '信息高速公路'的优势！")
print()

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：信息保留度随序列长度变化
ax = axes[0]
ax.plot(results["lengths"], results["rnn_retention"], "ro--",
        linewidth=2, markersize=8, label="Vanilla RNN")
ax.plot(results["lengths"], results["lstm_retention"], "bs-",
        linewidth=2, markersize=8, label="LSTM")
ax.set_xlabel("序列长度", fontsize=12)
ax.set_ylabel("信息保留度 (末尾/开头隐状态范数比)", fontsize=11)
ax.set_title("长距离依赖：LSTM vs RNN", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# 右图：隐状态范数随时间变化（以 seq_len=100 为例）
seq_length = 100
X_long = np.random.randn(seq_length, 1) * 0.1
X_long[0, 0] = 3.0

rnn_long = VanillaRNNCell(input_size=1, hidden_size=hidden_size)
h_rnn_long = rnn_long.forward_sequence(X_long)

lstm_long = LSTMCell(input_size=1, hidden_size=hidden_size)
h_lstm_long, _, _ = lstm_long.forward_sequence(X_long)

ax = axes[1]
ax.plot(np.linalg.norm(h_rnn_long, axis=1), "r-", alpha=0.8,
        linewidth=1.5, label="RNN |h_t|")
ax.plot(np.linalg.norm(h_lstm_long, axis=1), "b-", alpha=0.8,
        linewidth=1.5, label="LSTM |h_t|")
ax.axvline(x=0, color="green", linestyle="--", alpha=0.5, label="标记信号位置")
ax.set_xlabel("时间步", fontsize=12)
ax.set_ylabel("隐状态范数 |h_t|", fontsize=11)
ax.set_title("隐状态范数随时间变化（序列长度=100）", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_lstm_vs_rnn_long_dependency.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 02_lstm_vs_rnn_long_dependency.png")
print()


# ====================================================================
# 第四部分：GRU Cell 实现
# ====================================================================
print("=" * 60)
print("第四部分：GRU Cell 实现——LSTM 的简化版")
print("=" * 60)

print("""
GRU 对 LSTM 的简化：
  LSTM: 3个门(遗忘/输入/输出) + 细胞状态 → 复杂但强大
  GRU:  2个门(重置/更新)       + 无细胞状态 → 简洁且高效

  关键区别：
  - GRU 把遗忘门和输入门合并为"更新门" z
  - 更新门 z 同时控制遗忘和记忆：h_t = (1-z)·h_{t-1} + z·h̃
    → z 接近0：保留旧信息（相当于 LSTM 的遗忘门接近1）
    → z 接近1：接受新信息（相当于 LSTM 的输入门接近1）
  - 没有独立的细胞状态，直接在隐状态上操作
""")


class GRUCell:
    """
    GRU 单元的完整实现。

    核心公式：
        z_t = σ(W_z · [h_{t-1}, x_t] + b_z)     更新门
        r_t = σ(W_r · [h_{t-1}, x_t] + b_r)     重置门
        h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)  候选隐状态
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t   最终隐状态

    参数说明：
        input_size  : 输入向量的维度
        hidden_size : 隐藏状态的维度
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size

        scale = np.sqrt(2.0 / concat_size)

        # ── 更新门（Update Gate）──
        # z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        # 相当于 LSTM 的遗忘门和输入门的"合体"
        # z 接近 0 → 保持旧隐状态（不更新）
        # z 接近 1 → 用新候选值替换（大幅更新）
        self.W_z = np.random.randn(hidden_size, concat_size) * scale
        self.b_z = np.zeros(hidden_size)

        # ── 重置门（Reset Gate）──
        # r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        # 控制"看到多少过去的信息"来生成候选值
        # r 接近 0 → 忽略历史，像处理全新输入
        # r 接近 1 → 完全利用历史信息
        self.W_r = np.random.randn(hidden_size, concat_size) * scale
        self.b_r = np.zeros(hidden_size)

        # ── 候选隐状态 ──
        # h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)
        self.W_h = np.random.randn(hidden_size, concat_size) * scale
        self.b_h = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        """
        GRU 的前向传播（单时间步）。

        参数：
            x_t    : 当前输入，形状 (input_size,)
            h_prev : 上一时间步隐状态，形状 (hidden_size,)

        返回：
            h_t    : 当前隐状态
            gates  : 字典，包含各门的值
        """
        # 步骤 0：拼接 [h_{t-1}, x_t]
        concat = np.concatenate([h_prev, x_t])

        # 步骤 1：更新门 —— "要更新多少？"
        z_t = sigmoid(self.W_z @ concat + self.b_z)

        # 步骤 2：重置门 —— "参考多少历史？"
        r_t = sigmoid(self.W_r @ concat + self.b_r)

        # 步骤 3：候选隐状态 —— 重置门控制"看到"多少过去
        # 注意：r_t ⊙ h_prev 就是"经过筛选的历史信息"
        concat_reset = np.concatenate([r_t * h_prev, x_t])
        h_tilde = np.tanh(self.W_h @ concat_reset + self.b_h)

        # 步骤 4：混合旧隐状态与候选隐状态
        # (1-z)·保持旧信息 + z·接受新信息
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        gates = {"update": z_t, "reset": r_t, "candidate": h_tilde}
        return h_t, gates

    def forward_sequence(self, X):
        """对整个序列进行前向传播"""
        seq_len = X.shape[0]
        h_t = np.zeros(self.hidden_size)
        all_h, all_gates = [], []
        for t in range(seq_len):
            h_t, gates = self.forward(X[t], h_t)
            all_h.append(h_t.copy())
            all_gates.append(gates)
        return np.array(all_h), all_gates

    def param_count(self):
        """GRU 总参数量：3组权重（更新门、重置门、候选值）"""
        concat_size = self.input_size + self.hidden_size
        per_gate = self.hidden_size * concat_size + self.hidden_size
        return 3 * per_gate  # 注意：GRU 只有 3 组，LSTM 有 4 组


# 演示 GRU
gru = GRUCell(input_size=3, hidden_size=4)
print(f"GRU 参数: input_size={gru.input_size}, hidden_size={gru.hidden_size}")
print(f"总参数量: {gru.param_count()}")

# 用同样的序列测试
h_gru, gates_gru = gru.forward_sequence(x_demo)
print(f"\n输入序列形状: {x_demo.shape}")
print(f"GRU 隐状态序列形状: {h_gru.shape}")
print(f"时间步 0 的更新门值: {gates_gru[0]['update'].round(3)}")
print(f"时间步 0 的重置门值: {gates_gru[0]['reset'].round(3)}")
print()


# ====================================================================
# 第五部分：LSTM vs GRU 对比
# ====================================================================
print("=" * 60)
print("第五部分：LSTM vs GRU 全面对比")
print("=" * 60)

# 5.1 参数量对比
print("── 参数量对比 ──")
for inp, hid in [(1, 8), (8, 32), (32, 128), (128, 256)]:
    lstm_p = LSTMCell(inp, hid).param_count()
    gru_p = GRUCell(inp, hid).param_count()
    ratio = gru_p / lstm_p
    print(f"  input={inp:>3d}, hidden={hid:>3d} → "
          f"LSTM: {lstm_p:>7d} 参数, GRU: {gru_p:>7d} 参数, "
          f"GRU/LSTM = {ratio:.2f}")

print()
print("结论：GRU 参数量约为 LSTM 的 75%（3/4），因为少了一个门。")
print()

# 5.2 性能对比：在相同序列上的隐状态演化
print("── 隐状态演化对比 ──")
hidden_size_cmp = 16
input_size_cmp = 1

# 构造复杂信号
seq_len_cmp = 80
t_cmp = np.arange(seq_len_cmp)
signal_cmp = (np.sin(2 * np.pi * t_cmp / 15)
              + 0.5 * np.sin(2 * np.pi * t_cmp / 7)
              + np.random.randn(seq_len_cmp) * 0.1)
X_cmp = signal_cmp.reshape(-1, 1)

lstm_cmp = LSTMCell(input_size_cmp, hidden_size_cmp)
gru_cmp = GRUCell(input_size_cmp, hidden_size_cmp)

h_lstm_cmp, c_lstm_cmp, _ = lstm_cmp.forward_sequence(X_cmp)
h_gru_cmp, _ = gru_cmp.forward_sequence(X_cmp)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# 输入信号
axes[0].plot(t_cmp, signal_cmp, "k-", linewidth=1.5)
axes[0].fill_between(t_cmp, signal_cmp, alpha=0.15, color="gray")
axes[0].set_ylabel("输入信号", fontsize=11)
axes[0].set_title("LSTM vs GRU：隐状态演化对比", fontsize=14)
axes[0].grid(True, alpha=0.3)

# LSTM 隐状态热力图
im1 = axes[1].imshow(h_lstm_cmp.T, aspect="auto", cmap="RdBu_r",
                      interpolation="nearest")
axes[1].set_ylabel("LSTM 隐藏维度", fontsize=11)
axes[1].set_title("LSTM 隐状态 h_t（每行一个维度，颜色=值）", fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

# GRU 隐状态热力图
im2 = axes[2].imshow(h_gru_cmp.T, aspect="auto", cmap="RdBu_r",
                      interpolation="nearest")
axes[2].set_ylabel("GRU 隐藏维度", fontsize=11)
axes[2].set_xlabel("时间步 t", fontsize=11)
axes[2].set_title("GRU 隐状态 h_t（每行一个维度，颜色=值）", fontsize=11)
plt.colorbar(im2, ax=axes[2], fraction=0.02, pad=0.02)

plt.tight_layout()
plt.savefig("02_lstm_vs_gru_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 02_lstm_vs_gru_comparison.png")
print()

# 结构差异总结表
print("┌────────────────────┬──────────────────┬──────────────────┐")
print("│       特性         │      LSTM        │      GRU         │")
print("├────────────────────┼──────────────────┼──────────────────┤")
print("│  门的数量          │    3 个           │    2 个           │")
print("│  门的名称          │ 遗忘/输入/输出   │   更新/重置       │")
print("│  细胞状态          │    有             │    无             │")
print("│  参数比例          │    1.0x           │    0.75x          │")
print("│  记忆机制          │ 独立的 C_t       │ 直接在 h_t 上    │")
print("│  计算速度          │    较慢           │    较快            │")
print("│  适用场景          │ 超长序列/复杂依赖│ 中等序列/效率优先 │")
print("└────────────────────┴──────────────────┴──────────────────┘")
print()


# ====================================================================
# 第六部分：梯度流分析——为什么 LSTM 解决了梯度消失
# ====================================================================
print("=" * 60)
print("第六部分：梯度流分析——为什么 LSTM 解决了梯度消失")
print("=" * 60)

print("""
核心数学直觉：

  Vanilla RNN 的梯度传播：
    ∂h_t/∂h_{t-1} = diag(1 - h_t^2) · W_h
    经过 T 步后：梯度 ∝ W_h^T
    → 如果 W_h 的最大特征值 < 1 → 梯度指数消失
    → 如果 W_h 的最大特征值 > 1 → 梯度指数爆炸

  LSTM 的梯度传播（通过细胞状态）：
    ∂C_t/∂C_{t-1} = f_t    （只是遗忘门的值！）
    经过 T 步后：梯度 ∝ ∏ f_t
    → 只要 f_t 接近 1，梯度就几乎不衰减！
    → 这就是"信息高速公路"：梯度沿着细胞状态畅通无阻
""")


def compute_rnn_gradient_norms(input_size, hidden_size, seq_len, num_trials=20):
    """
    模拟 RNN 梯度范数随时间步的衰减。

    原理：RNN 的梯度通过 ∏_{k=t}^{T} (diag(1-h_k^2) · W_h) 传播。
    我们近似计算这个连乘的范数来观察梯度的衰减/爆炸。
    """
    all_norms = []
    for _ in range(num_trials):
        W_h = np.random.randn(hidden_size, hidden_size) * 0.5
        W_x = np.random.randn(hidden_size, input_size) * 0.5
        X = np.random.randn(seq_len, input_size) * 0.5

        # 前向传播，记录所有隐状态
        h_t = np.zeros(hidden_size)
        h_list = [h_t.copy()]
        for t in range(seq_len):
            h_t = np.tanh(W_h @ h_t + W_x @ X[t])
            h_list.append(h_t.copy())

        # 反向传播：计算 ∂h_T/∂h_t 的范数
        # 从最后一步往前累积雅可比矩阵
        grad_norms = [1.0]  # ∂h_T/∂h_T = I，范数为 1
        jacobian_prod = np.eye(hidden_size)
        for t in range(seq_len - 1, 0, -1):
            # 当前时间步的雅可比：diag(1 - h_{t+1}^2) · W_h
            h_val = h_list[t + 1]
            diag_deriv = np.diag(1 - h_val ** 2)  # tanh 的导数
            jacobian = diag_deriv @ W_h
            jacobian_prod = jacobian_prod @ jacobian  # 累积
            grad_norms.append(np.linalg.norm(jacobian_prod))

        grad_norms.reverse()
        all_norms.append(grad_norms)

    return np.mean(all_norms, axis=0)


def compute_lstm_gradient_norms(input_size, hidden_size, seq_len, num_trials=20):
    """
    模拟 LSTM 通过细胞状态的梯度范数。

    核心：∂C_T/∂C_t = ∏_{k=t+1}^{T} f_k
    其中 f_k 是遗忘门的值（在 0~1 之间）。
    """
    all_norms = []
    for _ in range(num_trials):
        lstm = LSTMCell(input_size, hidden_size)
        X = np.random.randn(seq_len, input_size) * 0.5

        # 前向传播，收集遗忘门值
        _, _, all_gates = lstm.forward_sequence(X)
        forget_values = [g["forget"] for g in all_gates]

        # 计算 ∂C_T/∂C_t ≈ ∏_{k=t+1}^{T} f_k 的范数
        grad_norms = [1.0]  # ∂C_T/∂C_T = I
        f_product = np.ones(hidden_size)
        for t in range(seq_len - 1, 0, -1):
            f_product = f_product * forget_values[t]
            grad_norms.append(np.linalg.norm(f_product) /
                              np.sqrt(hidden_size))  # 归一化
        grad_norms.reverse()
        all_norms.append(grad_norms)

    return np.mean(all_norms, axis=0)


# 计算梯度范数
seq_len_grad = 50
hidden_grad = 16

rnn_grads = compute_rnn_gradient_norms(1, hidden_grad, seq_len_grad)
lstm_grads = compute_lstm_gradient_norms(1, hidden_grad, seq_len_grad)

# 可视化梯度流
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：线性坐标
ax = axes[0]
time_back = np.arange(seq_len_grad + 1)  # 时间步
ax.plot(time_back, rnn_grads, "r-", linewidth=2, label="RNN 梯度范数")
ax.plot(time_back, lstm_grads, "b-", linewidth=2, label="LSTM 梯度范数")
ax.set_xlabel("从最后一步向前的时间步距离", fontsize=11)
ax.set_ylabel("梯度范数（线性坐标）", fontsize=11)
ax.set_title("梯度流对比（线性坐标）", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 右图：对数坐标（更清楚地展示指数衰减）
ax = axes[1]
ax.semilogy(time_back, np.array(rnn_grads) + 1e-20, "r-",
            linewidth=2, label="RNN 梯度范数")
ax.semilogy(time_back, np.array(lstm_grads) + 1e-20, "b-",
            linewidth=2, label="LSTM 梯度范数")
ax.set_xlabel("从最后一步向前的时间步距离", fontsize=11)
ax.set_ylabel("梯度范数（对数坐标）", fontsize=11)
ax.set_title("梯度流对比（对数坐标）", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

fig.suptitle("RNN vs LSTM 梯度衰减：LSTM 的细胞状态维持了梯度流",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("02_gradient_flow_rnn_vs_lstm.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 02_gradient_flow_rnn_vs_lstm.png")
print()

print("关键发现：")
print("  - RNN 梯度在 ~20 步后几乎衰减到 0（指数消失）")
print("  - LSTM 梯度衰减缓慢得多，信息可以跨越更长的时间距离")
print("  - 这就是为什么 LSTM 能处理长序列而 RNN 不能")
print()
print("类比 ResNet：")
print("  LSTM 的细胞状态 ←→ ResNet 的跳跃连接")
print("  两者都建立了'梯度高速公路'，让梯度畅通无阻地回传")
print()


# ====================================================================
# 第七部分：思考题
# ====================================================================
print("=" * 60)
print("第七部分：思考题")
print("=" * 60)

print("""
1. 【遗忘门偏置初始化】
   在 LSTMCell 实现中，遗忘门的偏置初始化为 1 而不是 0。
   这是为什么？如果初始化为 0，训练初期会出什么问题？
   提示：想想 sigmoid(0) 和 sigmoid(1) 分别是多少，
         以及这对细胞状态的保留意味着什么。

2. 【GRU 的更新门 vs LSTM 的遗忘门】
   在 GRU 中，h_t = (1-z)·h_{t-1} + z·h̃_t。
   注意这里 (1-z) 和 z 是"此消彼长"的关系。
   而在 LSTM 中，遗忘门 f 和输入门 i 是独立的。
   LSTM 这种"独立控制"有什么优势？能实现什么 GRU 做不到的？
   提示：考虑 f=1, i=1 和 f=0, i=0 这两种极端情况。

3. 【Peephole 连接】
   标准 LSTM 的门只看 [h_{t-1}, x_t]，即上一步的隐状态和当前输入。
   "Peephole LSTM" 让门还能直接看到细胞状态 C_{t-1}。
   这在什么场景下有用？请修改 LSTMCell 的代码实现 peephole 连接。
   提示：遗忘门变为 f = σ(W_f·[h_{t-1}, x_t, C_{t-1}] + b_f)。

4. 【双向 LSTM】
   标准 LSTM 只能看到过去（从左到右处理序列）。
   双向 LSTM 增加一个反向 LSTM（从右到左），然后拼接两个方向的隐状态。
   请用现有的 LSTMCell 实现一个 BidirectionalLSTM：
   对输入序列正向和反向各跑一遍，把两个方向的 h_t 拼接起来。

5. 【LSTM 与 Transformer 的关系】
   Transformer 的自注意力机制取代了 LSTM 的循环结构。
   但门控思想并没有消失——Transformer 中哪些组件体现了
   类似"门"的信息过滤思想？
   提示：思考注意力权重（softmax后的得分）和前馈网络中的
         GLU（Gated Linear Unit）激活函数。
""")


# ====================================================================
# 总结：本节核心要点
# ====================================================================
print("=" * 60)
print("总结：本节核心要点")
print("=" * 60)
print("""
  1. LSTM 用 4 个组件解决梯度消失：
     遗忘门(f) + 输入门(i) + 候选值(C̃) + 输出门(o)

  2. 细胞状态是 LSTM 的核心——"信息高速公路"
     C_t = f ⊙ C_{t-1} + i ⊙ C̃
     梯度通过 f_t 传播，不会像 RNN 那样指数衰减

  3. GRU 用 2 个门实现类似效果：更新门(z) + 重置门(r)
     参数量为 LSTM 的 75%，速度更快

  4. LSTM vs GRU 选择指南：
     - 超长序列/精度优先 → LSTM
     - 训练速度/资源有限 → GRU
     - 实际差异通常不大，先试 GRU，不够再用 LSTM

  5. 门控思想的深远影响：
     ResNet 跳跃连接、Transformer 注意力权重、GLU 激活
     都是"门"这一核心思想的不同化身

  下一节预告：第4章 · 第3节 · 序列到序列(Seq2Seq)与注意力机制
  → 从固定长度编码到动态注意力，迈向 Transformer 的关键一步
""")
