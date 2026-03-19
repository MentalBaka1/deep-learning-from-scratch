"""
====================================================================
第4章 · 第1节 · RNN 基础与 BPTT
====================================================================

【一句话总结】
RNN 通过"隐状态"传递历史信息，让网络能处理序列数据——
但它的"记忆"很短暂，这个缺陷直接推动了 LSTM 和 Transformer 的诞生。

【为什么深度学习需要这个？】
- 文本、语音、时间序列都是序列数据，普通网络处理不了
- RNN 引入了"时间维度"的概念，是理解所有序列模型的起点
- RNN 的梯度消失问题是 LSTM 和 Transformer 被发明的直接原因
- 理解 BPTT（时间反向传播）有助于理解为什么 Transformer 放弃了循环

【核心概念】

1. 序列数据的挑战
   - 输入长度可变（句子有长有短）
   - 顺序很重要（"猫追狗" ≠ "狗追猫"）
   - 长距离依赖（"我在中国出生...所以我说___"）

2. RNN 基本结构
   - h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
   - y_t = W_hy · h_t
   - 参数共享：所有时间步使用相同的 W_hh, W_xh, W_hy
   - 隐状态 h_t 就是网络的"记忆"

3. 时间反向传播（BPTT）
   - 将 RNN 展开成多层网络，每个时间步一层
   - 反向传播梯度沿时间轴回传
   - 问题：展开后的网络很深 → 梯度消失/爆炸

4. 梯度消失 vs 爆炸
   - ∂h_t/∂h_0 = Π W_hh · diag(tanh')
   - 如果 W_hh 的特征值 < 1：梯度指数级衰减（消失）
   - 如果 W_hh 的特征值 > 1：梯度指数级增长（爆炸）
   - 消失 → 学不到长距离依赖
   - 爆炸 → 梯度裁剪可以解决

5. 梯度裁剪
   - if ||gradient|| > threshold: gradient = gradient × threshold / ||gradient||

【前置知识】
第2章 - 神经网络基础、反向传播
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示（根据系统情况可能需要调整）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
np.random.seed(42)


# ====================================================================
# 第一部分：序列数据的直觉
# ====================================================================
def part1_sequence_intuition():
    """
    序列数据的直觉：为什么普通的全连接网络处理不了序列数据？

    三个核心挑战：
    1. 长度可变 —— 句子有长有短，固定大小的输入层搞不定
    2. 顺序重要 —— "我吃鱼" 和 "鱼吃我" 意思完全不同
    3. 长距离依赖 —— 需要记住很久以前的信息
    """
    print("=" * 60)
    print("第一部分：序列数据的直觉")
    print("=" * 60)

    # --- 挑战1：长度可变 ---
    print("\n挑战1：输入长度可变")
    print("-" * 40)
    sentences = ["猫", "猫追狗", "那只黑色的猫追着白色的狗跑"]
    print("  不同句子长度完全不同：")
    for s in sentences:
        print(f"    '{s}' → 长度 {len(s)}")
    print("  全连接网络的输入维度是固定的，没法直接处理！")
    print("  强行补齐到最大长度？浪费计算，而且最大长度不可预知。")

    # --- 挑战2：顺序重要 ---
    print("\n挑战2：顺序很重要")
    print("-" * 40)
    # 用简单编码演示：把每个"词"编码为一个数字
    vocab = {"猫": 1, "追": 2, "狗": 3, "吃": 4, "鱼": 5}
    seq_a = [vocab["猫"], vocab["追"], vocab["狗"]]  # 猫追狗
    seq_b = [vocab["狗"], vocab["追"], vocab["猫"]]  # 狗追猫

    # 如果直接把序列拼成向量丢给全连接网络
    vec_a = np.array(seq_a)
    vec_b = np.array(seq_b)
    print(f"  '猫追狗' 编码: {vec_a}  → 求和={vec_a.sum()}")
    print(f"  '狗追猫' 编码: {vec_b}  → 求和={vec_b.sum()}")
    print(f"  求和相同！但意思完全不同。")
    print(f"  如果用词袋模型(bag-of-words)，顺序信息就丢失了。")
    print(f"  RNN 的解法：逐个处理，用隐状态记住之前看过的内容。")

    # --- 挑战3：长距离依赖 ---
    print("\n挑战3：长距离依赖")
    print("-" * 40)
    print("  '我在中国出生，在北京长大，... (省略100字) ... 所以我说___'")
    print("  答案是'中文'，但关键信息'中国'出现在很久之前。")
    print("  网络需要跨越很长的距离传递信息。")
    print("  → 这正是 RNN 最大的弱点，也是 LSTM/Transformer 的设计动机。")

    # --- 可视化：顺序 vs 无序 ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左图：时间序列中顺序的重要性
    t = np.linspace(0, 4 * np.pi, 100)
    signal = np.sin(t) + 0.5 * np.sin(3 * t)
    ax = axes[0]
    ax.plot(t, signal, "b-", linewidth=2, label="原始信号（有序）")
    shuffled_signal = signal.copy()
    np.random.shuffle(shuffled_signal)
    ax.plot(t, shuffled_signal, "r-", alpha=0.5, linewidth=1,
            label="打乱顺序后（噪声）")
    ax.set_title("时间序列：顺序 = 信息", fontsize=12, fontweight="bold")
    ax.set_xlabel("时间")
    ax.set_ylabel("信号值")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：RNN 的核心思想——逐步处理 + 记忆
    ax = axes[1]
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("RNN 的核心思想：逐步处理 + 记忆传递", fontsize=12,
                 fontweight="bold")

    words = ["我", "在", "中国", "出生"]
    for i, word in enumerate(words):
        # 输入框
        ax.add_patch(plt.Rectangle((i * 1.4, 0), 0.8, 0.6, fill=True,
                                    facecolor="#AED6F1", edgecolor="black",
                                    linewidth=1.5))
        ax.text(i * 1.4 + 0.4, 0.3, f"$x_{i}$\n{word}", ha="center",
                va="center", fontsize=9, fontweight="bold")
        # RNN 单元
        ax.add_patch(plt.Rectangle((i * 1.4, 1.2), 0.8, 0.8, fill=True,
                                    facecolor="#FAD7A0", edgecolor="black",
                                    linewidth=1.5))
        ax.text(i * 1.4 + 0.4, 1.6, f"$h_{i}$\nRNN", ha="center",
                va="center", fontsize=9, fontweight="bold")
        # 输入到 RNN 的箭头
        ax.annotate("", xy=(i * 1.4 + 0.4, 1.2), xytext=(i * 1.4 + 0.4, 0.6),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        # 隐状态传递箭头
        if i < len(words) - 1:
            ax.annotate("", xy=((i + 1) * 1.4, 1.6),
                         xytext=(i * 1.4 + 0.8, 1.6),
                         arrowprops=dict(arrowstyle="->", color="#E74C3C",
                                         lw=2.5))

    # "记忆"标注
    ax.text(2.5, 2.5, "隐状态 = 记忆（逐步传递历史信息）",
            ha="center", fontsize=10, color="#E74C3C", fontweight="bold")

    plt.tight_layout()
    plt.savefig("04_01_part1_sequence_intuition.png", dpi=100,
                bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 04_01_part1_sequence_intuition.png]")


# ====================================================================
# 第二部分：RNN Cell 实现
# ====================================================================
def part2_rnn_cell():
    """
    RNN Cell 实现：单个时间步的前向传播。

    核心公式：
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y

    参数共享：所有时间步使用同一组 (W_xh, W_hh, W_hy, b_h, b_y)，
    这使得 RNN 可以处理任意长度的序列。
    """
    print("\n" + "=" * 60)
    print("第二部分：RNN Cell 实现")
    print("=" * 60)

    class RNNCell:
        """
        单步 RNN 单元。

        参数：
            input_size  : 输入向量维度
            hidden_size : 隐状态向量维度

        前向计算：
            h_new = tanh(W_xh @ x + W_hh @ h_prev + b_h)
        """

        def __init__(self, input_size, hidden_size):
            # Xavier 初始化权重
            scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
            scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
            self.W_xh = np.random.randn(hidden_size, input_size) * scale_xh
            self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
            self.b_h = np.zeros(hidden_size)
            self.hidden_size = hidden_size

        def forward(self, x, h_prev):
            """
            单个时间步的前向传播。

            参数：
                x      : 输入向量，形状 (input_size,)
                h_prev : 上一步隐状态，形状 (hidden_size,)

            返回：
                h_new  : 新隐状态，形状 (hidden_size,)
            """
            # 核心公式：h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            z = self.W_xh @ x + self.W_hh @ h_prev + self.b_h
            h_new = np.tanh(z)
            return h_new

        def init_hidden(self):
            """初始化隐状态为全零向量"""
            return np.zeros(self.hidden_size)

    # --- 演示 RNN Cell ---
    input_size = 3
    hidden_size = 4
    cell = RNNCell(input_size, hidden_size)

    print(f"\nRNN Cell 参数形状：")
    print(f"  W_xh: {cell.W_xh.shape}  (hidden_size x input_size)")
    print(f"  W_hh: {cell.W_hh.shape}  (hidden_size x hidden_size)")
    print(f"  b_h:  {cell.b_h.shape}   (hidden_size,)")

    # 单步前向传播
    x = np.array([1.0, 0.5, -0.3])
    h_prev = cell.init_hidden()
    h_new = cell.forward(x, h_prev)

    print(f"\n单步前向传播：")
    print(f"  输入 x     = {x}")
    print(f"  上一隐状态 = {h_prev}")
    print(f"  新隐状态   = {np.round(h_new, 4)}")
    print(f"  （tanh 将所有值压缩到 [-1, 1]）")

    # 连续两步，观察隐状态如何积累信息
    x2 = np.array([0.2, -0.8, 0.6])
    h_new2 = cell.forward(x2, h_new)
    print(f"\n再走一步：")
    print(f"  输入 x2    = {x2}")
    print(f"  上一隐状态 = {np.round(h_new, 4)} (包含了 x1 的信息)")
    print(f"  新隐状态   = {np.round(h_new2, 4)} (包含了 x1 和 x2 的信息)")

    return RNNCell


# ====================================================================
# 第三部分：展开 RNN —— 处理完整序列
# ====================================================================
def part3_unrolled_rnn(RNNCell):
    """
    展开 RNN：将 RNNCell 按时间步展开，处理一整个序列。

    可视化隐状态如何随时间演变，展示 RNN 的"记忆"机制。
    """
    print("\n" + "=" * 60)
    print("第三部分：展开 RNN —— 处理完整序列")
    print("=" * 60)

    # --- 构造一个简单的序列（正弦波采样点）---
    seq_len = 20
    input_size = 1
    hidden_size = 8
    t = np.linspace(0, 2 * np.pi, seq_len)
    # 输入序列：正弦波的每个采样点作为一个时间步
    sequence = np.sin(t).reshape(seq_len, input_size)

    cell = RNNCell(input_size, hidden_size)
    h = cell.init_hidden()

    # 逐步处理序列，记录每步隐状态
    hidden_states = [h.copy()]
    for step in range(seq_len):
        h = cell.forward(sequence[step], h)
        hidden_states.append(h.copy())

    hidden_states = np.array(hidden_states)  # (seq_len+1, hidden_size)

    print(f"\n序列长度: {seq_len}")
    print(f"隐状态维度: {hidden_size}")
    print(f"隐状态矩阵形状: {hidden_states.shape} (时间步+1, 隐状态维度)")
    print(f"\n每一步隐状态的范数（反映信息积累）：")
    for step in range(0, seq_len + 1, 5):
        norm = np.linalg.norm(hidden_states[step])
        print(f"  t={step:2d}: ||h|| = {norm:.4f}")

    # --- 可视化 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：输入序列
    ax = axes[0]
    ax.plot(range(seq_len), sequence[:, 0], "bo-", linewidth=2, markersize=5,
            label="输入 $x_t$（正弦波）")
    ax.set_title("输入序列", fontsize=12, fontweight="bold")
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("$x_t$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 下图：隐状态热力图
    ax = axes[1]
    im = ax.imshow(hidden_states[1:].T, aspect="auto", cmap="RdBu_r",
                    interpolation="nearest")
    ax.set_title("隐状态随时间的变化（每行一个隐藏单元）", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("隐藏单元编号")
    ax.set_yticks(range(hidden_size))
    plt.colorbar(im, ax=ax, label="激活值")

    plt.tight_layout()
    plt.savefig("04_01_part3_unrolled_rnn.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 04_01_part3_unrolled_rnn.png]")

    print("\n观察：")
    print("  - 不同隐藏单元对输入的响应模式不同")
    print("  - 有的单元跟踪趋势，有的单元关注局部变化")
    print("  - 隐状态是 RNN 对'已看到内容'的压缩表示")


# ====================================================================
# 第四部分：BPTT 实现 —— 时间反向传播
# ====================================================================
def part4_bptt():
    """
    BPTT（Backpropagation Through Time）：从零实现时间反向传播。

    核心思想：
    1. 将 RNN 在时间轴上展开，得到一个"很深"的网络
    2. 用标准反向传播计算梯度
    3. 但梯度需要沿时间轴回传，导致连乘效应 → 梯度消失/爆炸

    实现的 RNN：
        h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
        y_t = W_hy @ h_t + b_y
        loss = 0.5 * sum((y_t - target_t)^2)
    """
    print("\n" + "=" * 60)
    print("第四部分：BPTT 实现 —— 时间反向传播")
    print("=" * 60)

    class SimpleRNN:
        """
        带完整 BPTT 的简单 RNN。

        参数：
            input_size  : 输入维度
            hidden_size : 隐状态维度
            output_size : 输出维度
        """

        def __init__(self, input_size, hidden_size, output_size):
            self.hidden_size = hidden_size
            # 初始化权重（使用较小的值以避免初始梯度爆炸）
            scale = 0.1
            self.W_xh = np.random.randn(hidden_size, input_size) * scale
            self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
            self.b_h = np.zeros(hidden_size)
            self.W_hy = np.random.randn(output_size, hidden_size) * scale
            self.b_y = np.zeros(output_size)

        def forward(self, inputs, h0=None):
            """
            前向传播：处理整个序列。

            参数：
                inputs : 输入序列，形状 (seq_len, input_size)
                h0     : 初始隐状态，默认全零

            返回：
                outputs : 输出序列 (seq_len, output_size)
                hiddens : 隐状态序列 (seq_len+1, hidden_size)，含 h0
            """
            seq_len = len(inputs)
            if h0 is None:
                h0 = np.zeros(self.hidden_size)

            hiddens = [h0]
            outputs = []

            for t in range(seq_len):
                # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
                z_h = self.W_xh @ inputs[t] + self.W_hh @ hiddens[t] + self.b_h
                h_t = np.tanh(z_h)
                hiddens.append(h_t)

                # y_t = W_hy @ h_t + b_y
                y_t = self.W_hy @ h_t + self.b_y
                outputs.append(y_t)

            return np.array(outputs), np.array(hiddens)

        def bptt(self, inputs, targets, h0=None):
            """
            BPTT：时间反向传播，计算所有参数的梯度。

            参数：
                inputs  : 输入序列 (seq_len, input_size)
                targets : 目标序列 (seq_len, output_size)
                h0      : 初始隐状态

            返回：
                loss : 标量损失值
                grads : 字典，包含所有参数的梯度
            """
            seq_len = len(inputs)

            # ===== 前向传播（保存中间值以供反向传播） =====
            outputs, hiddens = self.forward(inputs, h0)

            # 计算 MSE 损失
            loss = 0.5 * np.sum((outputs - targets) ** 2) / seq_len

            # ===== 反向传播（BPTT） =====
            # 初始化梯度
            dW_xh = np.zeros_like(self.W_xh)
            dW_hh = np.zeros_like(self.W_hh)
            db_h = np.zeros_like(self.b_h)
            dW_hy = np.zeros_like(self.W_hy)
            db_y = np.zeros_like(self.b_y)

            # 从最后一个时间步开始的隐状态梯度（初始为零）
            dh_next = np.zeros(self.hidden_size)

            for t in reversed(range(seq_len)):
                # --- 输出层梯度 ---
                # dL/dy_t = (y_t - target_t) / seq_len
                dy = (outputs[t] - targets[t]) / seq_len

                # dL/dW_hy += dy @ h_t^T
                dW_hy += np.outer(dy, hiddens[t + 1])
                db_y += dy

                # --- 隐状态梯度（来自两条路径） ---
                # 路径1：来自当前输出 y_t = W_hy @ h_t
                dh = self.W_hy.T @ dy
                # 路径2：来自下一个时间步的隐状态
                dh += dh_next

                # --- 穿过 tanh 的梯度 ---
                # h_t = tanh(z_h)，tanh 的导数 = 1 - tanh^2
                dtanh = (1 - hiddens[t + 1] ** 2) * dh

                # --- 权重梯度 ---
                # z_h = W_xh @ x_t + W_hh @ h_{t-1} + b_h
                dW_xh += np.outer(dtanh, inputs[t])
                dW_hh += np.outer(dtanh, hiddens[t])
                db_h += dtanh

                # --- 传递给前一个时间步的梯度 ---
                dh_next = self.W_hh.T @ dtanh

            grads = {
                "W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                "W_hy": dW_hy, "b_y": db_y
            }
            return loss, grads

    # --- 测试 BPTT 正确性：数值梯度验证 ---
    print("\n数值梯度验证 BPTT 的正确性：")
    input_size, hidden_size, output_size = 2, 3, 1
    rnn = SimpleRNN(input_size, hidden_size, output_size)

    # 生成小序列
    seq_len = 4
    inputs = np.random.randn(seq_len, input_size)
    targets = np.random.randn(seq_len, output_size)

    # 解析梯度
    loss, grads = rnn.bptt(inputs, targets)
    print(f"  损失: {loss:.6f}")

    # 数值梯度（中心差分法）
    eps = 1e-5
    print(f"\n  {'参数':>8s}  {'解析梯度':>12s}  {'数值梯度':>12s}  {'相对误差':>12s}")
    print(f"  {'-' * 50}")

    for param_name in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
        param = getattr(rnn, param_name)
        grad_ana = grads[param_name]
        grad_num = np.zeros_like(param)

        # 对参数的每个元素做数值差分
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            # f(x + eps)
            param[idx] = old_val + eps
            loss_plus, _ = rnn.bptt(inputs, targets)
            # f(x - eps)
            param[idx] = old_val - eps
            loss_minus, _ = rnn.bptt(inputs, targets)
            # 还原
            param[idx] = old_val
            grad_num[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()

        # 相对误差
        diff = np.abs(grad_ana - grad_num)
        denom = np.maximum(np.abs(grad_ana) + np.abs(grad_num), 1e-15)
        rel_error = np.max(diff / denom)

        print(f"  {param_name:>8s}  {np.max(np.abs(grad_ana)):>12.6f}  "
              f"{np.max(np.abs(grad_num)):>12.6f}  {rel_error:>12.2e}  "
              f"{'PASS' if rel_error < 1e-4 else 'FAIL'}")

    print("\n  结论：解析梯度与数值梯度吻合 → BPTT 实现正确！")

    return SimpleRNN


# ====================================================================
# 第五部分：梯度消失演示
# ====================================================================
def part5_vanishing_gradients(SimpleRNN):
    """
    梯度消失演示：展示 RNN 的梯度如何随时间步距离指数级衰减。

    关键数学：
        ∂h_t/∂h_k = Π_{i=k+1}^{t} W_hh^T · diag(1 - h_i^2)
    如果 W_hh 的谱范数 < 1，这个连乘积会指数级趋近于零。
    """
    print("\n" + "=" * 60)
    print("第五部分：梯度消失演示")
    print("=" * 60)

    # --- 实验：测量梯度随时间距离的衰减 ---
    hidden_size = 16
    seq_len = 50

    # 用不同的 W_hh 谱范数来对比
    spectral_norms = [0.5, 0.9, 1.0, 1.5]
    results = {}

    for sn in spectral_norms:
        # 构造 W_hh 使其谱范数约为 sn
        # 方法：先随机生成，再缩放
        W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        # 缩放到目标谱范数
        current_sn = np.linalg.norm(W_hh, ord=2)
        W_hh = W_hh * (sn / current_sn)

        # 模拟 BPTT 中梯度沿时间的传播
        # ∂h_t/∂h_k 的范数近似为 |sn|^(t-k) （忽略 tanh 导数的影响）
        # 但更精确地，我们模拟实际的梯度传播

        # 前向传播生成隐状态
        h = np.zeros(hidden_size)
        x_input = np.random.randn(seq_len, 1) * 0.1
        W_xh = np.random.randn(hidden_size, 1) * 0.1
        hiddens = [h]
        for t in range(seq_len):
            z = W_xh @ x_input[t] + W_hh @ h + np.zeros(hidden_size)
            h = np.tanh(z)
            hiddens.append(h)

        # 反向传播：从最后一个时间步出发，追踪梯度范数
        grad_norms = []
        dh = np.ones(hidden_size)  # 假设从最后一步回传的梯度
        dh = dh / np.linalg.norm(dh)  # 归一化，方便比较

        for t in reversed(range(seq_len)):
            grad_norms.append(np.linalg.norm(dh))
            # 穿过 tanh：乘以 (1 - h_t^2)
            dtanh = (1 - hiddens[t + 1] ** 2)
            dh = W_hh.T @ (dtanh * dh)

        grad_norms.reverse()  # 变成从远到近的顺序
        results[sn] = grad_norms

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 左图：不同谱范数下的梯度衰减
    ax = axes[0]
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    for (sn, gnorms), color in zip(results.items(), colors):
        # 距离 = 从最后一步到当前步
        distances = list(range(seq_len))
        ax.semilogy(distances, gnorms, "-", linewidth=2, color=color,
                     label=f"$\\rho(W_{{hh}})$ = {sn}")
    ax.set_title("梯度范数 vs 时间步距离", fontsize=12, fontweight="bold")
    ax.set_xlabel("与最后时间步的距离")
    ax.set_ylabel("梯度范数（对数坐标）")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右图：理论衰减曲线
    ax = axes[1]
    distances = np.arange(seq_len)
    for sn, color in zip(spectral_norms, colors):
        # 理论近似：梯度 ~ sn^distance * (max tanh')^distance
        # tanh 的最大导数是 1（在 0 处），所以上界是 sn^d
        theoretical = sn ** distances
        ax.semilogy(distances, theoretical, "--", linewidth=2, color=color,
                     label=f"$\\rho^d$, $\\rho$={sn}")
    ax.axhline(y=1e-7, color="gray", linestyle=":", alpha=0.7,
               label="梯度消失阈值")
    ax.set_title("理论预测：$||\\partial h_t / \\partial h_k|| \\approx \\rho^{(t-k)}$",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("时间步距离 (t - k)")
    ax.set_ylabel("梯度范数上界")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_01_part5_vanishing_gradients.png", dpi=100,
                bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 04_01_part5_vanishing_gradients.png]")

    print("\n关键观察：")
    print(f"  - 谱范数 < 1 (如 0.5)：梯度指数级衰减 → 梯度消失")
    print(f"  - 谱范数 = 1 (如 1.0)：梯度近似保持 → 理想情况")
    print(f"  - 谱范数 > 1 (如 1.5)：梯度指数级增长 → 梯度爆炸")
    print(f"  - 距离 30 步以上的梯度几乎为 0 → RNN 学不到长距离依赖")
    print(f"  - 这就是为什么 LSTM 引入了'门控'机制，Transformer 放弃了循环")


# ====================================================================
# 第六部分：梯度裁剪
# ====================================================================
def part6_gradient_clipping(SimpleRNN):
    """
    梯度裁剪（Gradient Clipping）：防止梯度爆炸的简单有效方法。

    方法：if ||g|| > threshold: g = g * threshold / ||g||

    直觉：不改变梯度方向，只缩小梯度大小。
    就像给汽车装了限速器——方向盘你说了算，但速度有上限。
    """
    print("\n" + "=" * 60)
    print("第六部分：梯度裁剪")
    print("=" * 60)

    def clip_gradients(grads, max_norm):
        """
        梯度裁剪：如果梯度范数超过阈值，等比例缩小。

        参数：
            grads    : 梯度字典 {参数名: 梯度数组}
            max_norm : 最大允许范数

        返回：
            clipped_grads : 裁剪后的梯度字典
            total_norm    : 裁剪前的总范数
        """
        # 计算所有梯度的总范数
        total_norm = 0.0
        for g in grads.values():
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)

        # 如果超过阈值，等比例缩小
        clip_coef = max_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            clipped_grads = {k: v * clip_coef for k, v in grads.items()}
        else:
            clipped_grads = {k: v.copy() for k, v in grads.items()}

        return clipped_grads, total_norm

    # --- 演示：有无梯度裁剪的训练对比 ---
    print("\n演示：用 RNN 拟合简单序列，对比有无梯度裁剪")

    # 生成训练数据：正弦波预测
    seq_len = 15
    t = np.linspace(0, 3 * np.pi, seq_len + 1)
    data = np.sin(t)
    inputs = data[:-1].reshape(seq_len, 1)
    targets = data[1:].reshape(seq_len, 1)

    # 训练参数
    lr = 0.01
    n_epochs = 150
    max_norm = 5.0

    # --- 无裁剪训练 ---
    np.random.seed(123)
    rnn_noclip = SimpleRNN(1, 8, 1)
    losses_noclip = []
    gnorms_noclip = []

    for epoch in range(n_epochs):
        loss, grads = rnn_noclip.bptt(inputs, targets)
        # 计算梯度范数
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        losses_noclip.append(loss)
        gnorms_noclip.append(total_norm)
        # 直接用梯度更新（可能爆炸）
        for pname in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
            param = getattr(rnn_noclip, pname)
            param -= lr * grads[pname]

    # --- 有裁剪训练 ---
    np.random.seed(123)
    rnn_clip = SimpleRNN(1, 8, 1)
    losses_clip = []
    gnorms_clip = []

    for epoch in range(n_epochs):
        loss, grads = rnn_clip.bptt(inputs, targets)
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        gnorms_clip.append(total_norm)
        # 裁剪梯度
        clipped_grads, _ = clip_gradients(grads, max_norm)
        losses_clip.append(loss)
        # 用裁剪后的梯度更新
        for pname in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
            param = getattr(rnn_clip, pname)
            param -= lr * clipped_grads[pname]

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 左图：损失曲线对比
    ax = axes[0]
    ax.plot(losses_noclip, "r-", linewidth=1.5, alpha=0.8, label="无裁剪")
    ax.plot(losses_clip, "b-", linewidth=1.5, alpha=0.8, label="有裁剪")
    ax.set_title("损失曲线对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("MSE 损失")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 限制 y 轴防止爆炸值影响可视化
    valid_max = max(max(losses_clip), np.nanmedian(losses_noclip) * 5)
    ax.set_ylim(0, min(valid_max, 5.0))

    # 中图：梯度范数对比
    ax = axes[1]
    ax.semilogy(gnorms_noclip, "r-", linewidth=1.5, alpha=0.8, label="无裁剪")
    ax.semilogy(gnorms_clip, "b-", linewidth=1.5, alpha=0.8, label="有裁剪")
    ax.axhline(y=max_norm, color="green", linestyle="--", linewidth=2,
               label=f"裁剪阈值 = {max_norm}")
    ax.set_title("梯度范数对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("梯度范数（对数坐标）")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：裁剪的几何直觉
    ax = axes[2]
    # 生成一些 2D 梯度向量
    np.random.seed(7)
    raw_grads = np.random.randn(30, 2) * 3
    threshold = 2.0
    clipped = []
    for g in raw_grads:
        norm = np.linalg.norm(g)
        if norm > threshold:
            clipped.append(g * threshold / norm)
        else:
            clipped.append(g.copy())
    clipped = np.array(clipped)

    # 画原始梯度
    for g in raw_grads:
        ax.arrow(0, 0, g[0], g[1], head_width=0.08, head_length=0.05,
                 fc="red", ec="red", alpha=0.3)
    # 画裁剪后的梯度
    for g in clipped:
        ax.arrow(0, 0, g[0], g[1], head_width=0.08, head_length=0.05,
                 fc="blue", ec="blue", alpha=0.5)
    # 画裁剪半径
    circle = plt.Circle((0, 0), threshold, fill=False, color="green",
                          linestyle="--", linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title("裁剪的几何直觉", fontsize=12, fontweight="bold")
    ax.set_xlabel("梯度分量 1")
    ax.set_ylabel("梯度分量 2")
    # 手动添加图例
    ax.plot([], [], "r-", linewidth=2, alpha=0.5, label="原始梯度")
    ax.plot([], [], "b-", linewidth=2, alpha=0.7, label="裁剪后梯度")
    ax.plot([], [], "g--", linewidth=2, label=f"裁剪半径 = {threshold}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_01_part6_gradient_clipping.png", dpi=100,
                bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 04_01_part6_gradient_clipping.png]")

    print("\n关键观察：")
    print("  - 无裁剪时，梯度范数可能突然暴增，导致参数更新过大")
    print("  - 梯度裁剪保留了梯度方向，只限制了步长大小")
    print("  - 裁剪阈值是超参数，常用值为 1.0 ~ 10.0")
    print("  - 裁剪能治'梯度爆炸'，但治不了'梯度消失'")
    print("  - 梯度消失需要结构性改进：LSTM 门控 / 残差连接 / Transformer")


# ====================================================================
# 第七部分：字符级文本生成
# ====================================================================
def part7_char_generation():
    """
    字符级文本生成：用 RNN 学习一段文本的字符模式，然后生成新文本。

    流程：
    1. 将文本拆成字符，建立字符-索引映射
    2. 用 one-hot 编码输入，训练 RNN 预测下一个字符
    3. 采样生成新文本

    这是最简单但最直观的语言模型！
    """
    print("\n" + "=" * 60)
    print("第七部分：字符级文本生成")
    print("=" * 60)

    # --- 准备训练文本 ---
    text = "hello world hello deep learning hello neural network "
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)

    print(f"\n训练文本: '{text}'")
    print(f"词表大小: {vocab_size}")
    print(f"字符表: {chars}")

    # --- 构造训练数据 ---
    # 输入：text[:-1] 的 one-hot 编码
    # 目标：text[1:] 的索引
    def one_hot(idx, size):
        """将索引转换为 one-hot 向量"""
        v = np.zeros(size)
        v[idx] = 1.0
        return v

    inputs_idx = [char_to_idx[ch] for ch in text[:-1]]
    targets_idx = [char_to_idx[ch] for ch in text[1:]]
    inputs_oh = np.array([one_hot(i, vocab_size) for i in inputs_idx])

    # --- RNN 字符模型（含 softmax 输出） ---
    class CharRNN:
        """
        字符级 RNN 语言模型。

        结构：one-hot 输入 → 隐藏层(tanh) → softmax 输出
        损失：交叉熵损失
        """

        def __init__(self, vocab_size, hidden_size):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            # 权重初始化
            scale = 0.1
            self.W_xh = np.random.randn(hidden_size, vocab_size) * scale
            self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
            self.b_h = np.zeros(hidden_size)
            self.W_hy = np.random.randn(vocab_size, hidden_size) * scale
            self.b_y = np.zeros(vocab_size)

        def forward_and_loss(self, inputs_oh, targets_idx):
            """
            前向传播 + 计算交叉熵损失。

            参数：
                inputs_oh   : one-hot 输入序列 (seq_len, vocab_size)
                targets_idx : 目标字符索引序列 (seq_len,)

            返回：
                loss, cache (用于反向传播)
            """
            seq_len = len(inputs_oh)
            h = np.zeros(self.hidden_size)

            hs = {-1: h.copy()}  # 保存隐状态
            probs = {}  # 保存 softmax 概率
            loss = 0.0

            for t in range(seq_len):
                # 隐藏层
                z_h = self.W_xh @ inputs_oh[t] + self.W_hh @ hs[t - 1] + self.b_h
                hs[t] = np.tanh(z_h)
                # 输出层
                z_y = self.W_hy @ hs[t] + self.b_y
                # Softmax（数值稳定版本）
                z_y -= np.max(z_y)
                exp_z = np.exp(z_y)
                probs[t] = exp_z / np.sum(exp_z)
                # 交叉熵损失
                loss -= np.log(probs[t][targets_idx[t]] + 1e-12)

            loss /= seq_len
            cache = (hs, probs, inputs_oh, targets_idx, seq_len)
            return loss, cache

        def backward(self, cache):
            """
            BPTT 反向传播。

            返回：所有参数的梯度字典
            """
            hs, probs, inputs_oh, targets_idx, seq_len = cache

            dW_xh = np.zeros_like(self.W_xh)
            dW_hh = np.zeros_like(self.W_hh)
            db_h = np.zeros_like(self.b_h)
            dW_hy = np.zeros_like(self.W_hy)
            db_y = np.zeros_like(self.b_y)
            dh_next = np.zeros(self.hidden_size)

            for t in reversed(range(seq_len)):
                # softmax + 交叉熵的梯度 = probs - one_hot(target)
                dy = probs[t].copy()
                dy[targets_idx[t]] -= 1.0
                dy /= seq_len

                dW_hy += np.outer(dy, hs[t])
                db_y += dy

                dh = self.W_hy.T @ dy + dh_next
                dtanh = (1 - hs[t] ** 2) * dh

                dW_xh += np.outer(dtanh, inputs_oh[t])
                dW_hh += np.outer(dtanh, hs[t - 1])
                db_h += dtanh
                dh_next = self.W_hh.T @ dtanh

            grads = {
                "W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                "W_hy": dW_hy, "b_y": db_y
            }
            return grads

        def generate(self, seed_char, length, temperature=1.0):
            """
            从种子字符开始，逐步生成文本。

            参数：
                seed_char   : 起始字符
                length      : 生成长度
                temperature : 温度参数（越高越随机，越低越确定）
            """
            h = np.zeros(self.hidden_size)
            idx = char_to_idx[seed_char]
            result = [seed_char]

            for _ in range(length):
                x = one_hot(idx, self.vocab_size)
                h = np.tanh(self.W_xh @ x + self.W_hh @ h + self.b_h)
                z_y = self.W_hy @ h + self.b_y
                # 温度缩放
                z_y = z_y / temperature
                z_y -= np.max(z_y)
                exp_z = np.exp(z_y)
                probs = exp_z / np.sum(exp_z)
                # 按概率采样下一个字符
                idx = np.random.choice(self.vocab_size, p=probs)
                result.append(idx_to_char[idx])

            return "".join(result)

    # --- 训练 ---
    print("\n开始训练字符级 RNN...")
    np.random.seed(42)
    model = CharRNN(vocab_size, hidden_size=32)
    lr = 0.05
    n_epochs = 300
    losses = []

    for epoch in range(n_epochs):
        loss, cache = model.forward_and_loss(inputs_oh, targets_idx)
        grads = model.backward(cache)

        # 梯度裁剪
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        max_norm = 5.0
        if total_norm > max_norm:
            clip_coef = max_norm / total_norm
            grads = {k: v * clip_coef for k, v in grads.items()}

        # 参数更新
        for pname in ["W_xh", "W_hh", "b_h", "W_hy", "b_y"]:
            setattr(model, pname,
                    getattr(model, pname) - lr * grads[pname])

        losses.append(loss)

        if epoch % 60 == 0 or epoch == n_epochs - 1:
            sample = model.generate("h", 40, temperature=0.8)
            print(f"  epoch {epoch:3d} | loss={loss:.4f} | 生成: '{sample}'")

    # --- 不同温度的生成效果 ---
    print("\n不同温度下的生成效果：")
    for temp in [0.3, 0.8, 1.5]:
        sample = model.generate("h", 50, temperature=temp)
        print(f"  温度 {temp}: '{sample}'")

    print("\n温度解读：")
    print("  低温 (0.3) → 更确定，倾向重复训练数据")
    print("  中温 (0.8) → 平衡创造性和连贯性")
    print("  高温 (1.5) → 更随机，可能出现新组合")

    # --- 可视化训练过程 ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(losses, "b-", linewidth=1.5, alpha=0.7)
    # 滑动平均
    window = 20
    if len(losses) > window:
        smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(losses)), smooth, "r-", linewidth=2.5,
                label=f"滑动平均 (窗口={window})")
    ax.set_title("字符级 RNN 训练损失", fontsize=13, fontweight="bold")
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("交叉熵损失")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("04_01_part7_char_generation.png", dpi=100,
                bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 04_01_part7_char_generation.png]")


# ====================================================================
# 第八部分：思考题
# ====================================================================
def part8_exercises():
    """
    思考题：检验你对 RNN 和 BPTT 的理解。
    """
    print("\n" + "=" * 60)
    print("第八部分：思考题")
    print("=" * 60)

    questions = [
        {
            "q": "RNN 的参数共享有什么好处？如果每个时间步用不同的权重会怎样？",
            "hint": "提示：想想参数数量和序列长度的关系。",
            "answer": (
                "参数共享让 RNN 的参数数量与序列长度无关，\n"
                "    这使得同一个模型可以处理任意长度的序列。\n"
                "    如果每步用不同权重：1) 参数量与序列长度成正比，巨大浪费\n"
                "    2) 训练时见过长度10的序列，预测时来了长度20的就没权重用了\n"
                "    3) 不同位置无法共享学到的模式（如'hello'在句首句中都应被识别）"
            )
        },
        {
            "q": "为什么 RNN 用 tanh 而不是 sigmoid 做隐状态的激活函数？",
            "hint": "提示：比较两者的输出范围和梯度特性。",
            "answer": (
                "1) tanh 输出范围是 [-1, 1]，以零为中心；\n"
                "   sigmoid 输出范围是 [0, 1]，总是正数。\n"
                "   非零中心会导致梯度更新只朝一个方向走（zig-zag 效应）。\n"
                "2) tanh 在 0 处的梯度为 1（vs sigmoid 的 0.25），\n"
                "   连乘时衰减更慢，梯度消失稍轻。\n"
                "3) 但 tanh 仍然有梯度消失问题，只是比 sigmoid 好一些。"
            )
        },
        {
            "q": "BPTT 截断（Truncated BPTT）是什么？为什么实践中要用它？",
            "hint": "提示：如果序列长度是 10000，完整 BPTT 会有什么问题？",
            "answer": (
                "完整 BPTT 需要存储所有时间步的中间值，内存消耗与序列长度成正比。\n"
                "    序列长度 10000 → 存 10000 个隐状态 → 内存爆炸。\n"
                "    截断 BPTT 只回传固定步数 k（如 35 步），超过 k 步的梯度直接丢弃。\n"
                "    代价：失去了超过 k 步的长距离依赖。\n"
                "    好处：内存和计算量恒定，训练稳定。\n"
                "    这也间接说明了 RNN 本就学不好太长的依赖。"
            )
        },
        {
            "q": "字符级生成中的'温度'参数是怎么工作的？\n"
                 "   temperature → 0 和 temperature → ∞ 分别会怎样？",
            "hint": "提示：softmax(z/T) 中 T 的作用。",
            "answer": (
                "温度 T 控制 softmax 分布的'尖锐度'：\n"
                "    softmax(z_i / T) = exp(z_i/T) / sum(exp(z_j/T))\n"
                "    T → 0：分布趋向 one-hot（只选概率最高的），完全确定性\n"
                "    T = 1：标准 softmax，正常的概率分布\n"
                "    T → ∞：分布趋向均匀分布，完全随机\n"
                "    实践中 T = 0.7~1.0 较常用，平衡多样性和质量。"
            )
        },
        {
            "q": "RNN 的梯度消失和全连接深层网络的梯度消失有什么本质区别？\n"
                 "   为什么同样是梯度消失，解决方案却不同？",
            "hint": "提示：深层网络每层参数不同，RNN 每步参数相同。",
            "answer": (
                "全连接深层网络：每层权重不同，可以逐层调整初始化/归一化来缓解。\n"
                "    RNN 的梯度消失：同一个 W_hh 反复相乘 → 特征值决定一切。\n"
                "    如果 W_hh 的特征值 < 1，无论怎么初始化，连乘必定衰减。\n"
                "    所以 RNN 需要结构性改变：\n"
                "    - LSTM：引入'细胞状态'，通过门控让梯度走'高速公路'\n"
                "    - GRU：LSTM 的简化版，更新门和重置门\n"
                "    - Transformer：完全放弃循环，用注意力直连所有位置\n"
                "    ResNet 的残差连接思想与 LSTM 的细胞状态有异曲同工之妙。"
            )
        },
    ]

    for i, item in enumerate(questions, 1):
        print(f"\n思考题 {i}：{item['q']}")
        print(f"  {item['hint']}")
        print(f"\n  参考答案：{item['answer']}")


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   第4章 · 第1节 · RNN 基础与 BPTT                       |")
    print("|   序列建模的起点——理解循环神经网络的核心机制            |")
    print("+" + "=" * 58 + "+")

    part1_sequence_intuition()                   # 序列数据的直觉
    RNNCell = part2_rnn_cell()                   # RNN Cell 实现
    part3_unrolled_rnn(RNNCell)                  # 展开 RNN
    SimpleRNN = part4_bptt()                     # BPTT 实现
    part5_vanishing_gradients(SimpleRNN)          # 梯度消失演示
    part6_gradient_clipping(SimpleRNN)            # 梯度裁剪
    part7_char_generation()                       # 字符级文本生成
    part8_exercises()                             # 思考题

    print("\n" + "=" * 60)
    print("本节总结")
    print("=" * 60)
    print("""
    1. 序列数据需要特殊处理：长度可变、顺序重要、长距离依赖
    2. RNN 通过隐状态 h_t 在时间步之间传递信息，参数共享处理任意长度
    3. BPTT 将 RNN 展开为深层网络，用链式法则计算梯度
    4. 梯度消失/爆炸源于 W_hh 的反复相乘（特征值 < 1 消失，> 1 爆炸）
    5. 梯度裁剪可治爆炸，但消失需要结构性改进（LSTM/Transformer）
    6. 字符级 RNN 是最简单的语言模型，温度参数控制生成的随机性

    下一节预告：第4章 · 第2节 · LSTM 与 GRU
    → LSTM 如何用门控机制解决梯度消失？GRU 的简化思路是什么？
    """)
