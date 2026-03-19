"""
====================================================================
第2章 · 第2节 · 激活函数全览
====================================================================

【一句话总结】
激活函数给神经网络注入"非线性"——没有它，再多层也只是线性变换，
解决不了任何复杂问题。

【为什么深度学习需要这个？】
- 没有激活函数：多层线性变换 = 一层线性变换（矩阵乘法的结合律）
- 有了激活函数：多层网络可以逼近任意复杂函数（万能近似定理）
- 不同激活函数有不同特性，选错了会导致训练失败（梯度消失）
- Transformer 用 GELU，LLaMA 用 SwiGLU——了解为什么

【核心概念】

1. 为什么需要非线性？
   - 证明：W2(W1·x) = (W2·W1)·x = W'·x，多层线性=一层线性
   - 加入非线性：W2·σ(W1·x) ≠ W'·x，表达能力质变

2. Sigmoid：σ(x) = 1/(1+e^{-x})
   - 输出范围 (0,1)，可解释为概率
   - 缺点：梯度消失（两端梯度接近0）、非零中心
   - 现在只用在二分类输出层

3. Tanh：tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
   - 输出范围 (-1,1)，零中心（比sigmoid好）
   - 仍有梯度消失问题
   - 在 LSTM 门控中仍然使用

4. ReLU：max(0, x)
   - 计算简单，梯度不会消失（正区间）
   - 缺点：Dead ReLU（负输入梯度永远为0，神经元"死亡"）
   - 深度学习的默认选择

5. Leaky ReLU：max(αx, x)，α通常=0.01
   - 解决 Dead ReLU：负区间也有小梯度

6. GELU：x·Φ(x)，Φ是标准正态CDF
   - 平滑版ReLU，概率性地"门控"输入
   - Transformer、BERT、GPT 的默认激活函数
   - 为什么比ReLU好：平滑+允许小负值通过

7. Swish / SiLU：x·σ(x)
   - GELU的近亲，Google发现的
   - 自门控：输入自己决定通过多少

8. SwiGLU：用在 LLaMA、Qwen 等现代大模型的 FFN 中
   - Swish(xW1) ⊙ (xW2)，双线性门控
   - 目前大模型的主流选择

【前置知识】
第2章第1节 - 感知机，第0章第2节 - 导数
"""

import numpy as np
import matplotlib.pyplot as plt


def _erf(x):
    """
    误差函数 erf(x) 的高精度近似（Abramowitz & Stegun 公式 7.1.26）。
    最大误差 < 1.5e-7，足够用于 GELU 计算，避免依赖 scipy。
    """
    sign = np.sign(x)
    x = np.abs(x)
    # 常数来自 Abramowitz & Stegun
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * np.exp(-x**2)
    return sign * y


# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 第1部分：为什么需要非线性？
# ════════════════════════════════════════════════════════════════════
#
# 核心证明：
#   层1输出: h = W1 · x
#   层2输出: y = W2 · h = W2 · (W1 · x) = (W2 · W1) · x = W' · x
#   无论堆多少层，结果仍然是一个线性变换 W' · x
#   加入非线性 σ 后：y = W2 · σ(W1 · x)，无法合并成单一矩阵
#

print("=" * 60)
print("第1部分：为什么需要非线性？")
print("=" * 60)

# --- 实验：多层线性 = 一层线性 ---
# 随机初始化三层线性变换的权重
W1 = np.random.randn(4, 3)   # 第1层：3维 → 4维
W2 = np.random.randn(5, 4)   # 第2层：4维 → 5维
W3 = np.random.randn(2, 5)   # 第3层：5维 → 2维

# 输入向量
x = np.random.randn(3)

# 逐层计算（无激活函数）
h1 = W1 @ x        # 第1层输出
h2 = W2 @ h1       # 第2层输出
y_multi = W3 @ h2   # 第3层输出（三层线性）

# 合并成一层：W_combined = W3 · W2 · W1
W_combined = W3 @ W2 @ W1  # 形状 (2, 3)
y_single = W_combined @ x   # 一层线性

print(f"三层线性网络输出:  {y_multi}")
print(f"合并成一层的输出:  {y_single}")
print(f"两者完全相同？     {np.allclose(y_multi, y_single)}")
print(f"\n结论：没有激活函数，3层线性 = 1层线性，白堆了两层！")

# --- 加入非线性后就不一样了 ---
h1_act = np.maximum(0, W1 @ x)        # 加 ReLU
h2_act = np.maximum(0, W2 @ h1_act)   # 加 ReLU
y_nonlinear = W3 @ h2_act

print(f"\n加入 ReLU 后的输出: {y_nonlinear}")
print(f"与一层线性相同？    {np.allclose(y_nonlinear, y_single)}")
print(f"结论：加入非线性后，多层网络 ≠ 一层线性，表达能力质变！\n")


# ════════════════════════════════════════════════════════════════════
# 第2部分：Sigmoid 详解
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第2部分：Sigmoid 详解")
print("=" * 60)


def sigmoid(x):
    """Sigmoid 激活函数：σ(x) = 1 / (1 + e^{-x})"""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid 的导数：σ'(x) = σ(x) · (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


x_range = np.linspace(-8, 8, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：Sigmoid 函数 ---
ax = axes[0]
ax.plot(x_range, sigmoid(x_range), "b-", linewidth=2.5, label="σ(x)")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="y=0.5")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
# 标注饱和区域
ax.axvspan(-8, -4, alpha=0.1, color="red")
ax.axvspan(4, 8, alpha=0.1, color="red")
ax.text(-6, 0.8, "饱和区\n梯度≈0", fontsize=10, ha="center", color="red")
ax.text(6, 0.2, "饱和区\n梯度≈0", fontsize=10, ha="center", color="red")
ax.set_title("Sigmoid 函数", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("σ(x)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

# --- 右图：Sigmoid 导数 ---
ax = axes[1]
ax.plot(x_range, sigmoid_derivative(x_range), "r-", linewidth=2.5, label="σ'(x)")
ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="最大值=0.25")
ax.set_title("Sigmoid 导数（梯度）", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("σ'(x)")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
# 标注关键信息
ax.annotate("最大梯度仅 0.25\n多层连乘→指数衰减!",
            xy=(0, 0.25), xytext=(3, 0.2),
            fontsize=10, color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred"))

plt.suptitle("Sigmoid：早期神经网络的默认选择，现在只用于输出层", fontsize=13)
plt.tight_layout()
plt.show()

# 演示梯度消失
print("\nSigmoid 梯度消失演示：")
print(f"  σ'(0)  = {sigmoid_derivative(0):.4f}  ← 最大梯度，仅 0.25")
print(f"  σ'(3)  = {sigmoid_derivative(3):.4f}  ← 已经很小了")
print(f"  σ'(5)  = {sigmoid_derivative(5):.6f}")
print(f"  σ'(10) = {sigmoid_derivative(10):.10f}  ← 几乎为零")
print(f"\n  假设10层网络，每层梯度最大 0.25：")
print(f"  反向传播到第1层：0.25^10 = {0.25**10:.10f}  ← 梯度消失！")


# ════════════════════════════════════════════════════════════════════
# 第3部分：Tanh 详解
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("第3部分：Tanh 详解")
print("=" * 60)


def tanh(x):
    """Tanh 激活函数：tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})"""
    return np.tanh(x)


def tanh_derivative(x):
    """Tanh 的导数：tanh'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：Tanh vs Sigmoid ---
ax = axes[0]
ax.plot(x_range, tanh(x_range), "b-", linewidth=2.5, label="tanh(x)")
ax.plot(x_range, sigmoid(x_range), "r--", linewidth=1.5, alpha=0.6, label="σ(x)")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("Tanh vs Sigmoid", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(3, -0.5, "tanh 是零中心的\n(-1, 1)", fontsize=10, color="blue",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# --- 右图：导数对比 ---
ax = axes[1]
ax.plot(x_range, tanh_derivative(x_range), "b-", linewidth=2.5, label="tanh'(x)")
ax.plot(x_range, sigmoid_derivative(x_range), "r--", linewidth=1.5, alpha=0.6,
        label="σ'(x)")
ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
ax.set_title("导数对比", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("导数值")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.annotate("tanh 最大梯度=1\n比 sigmoid 的 0.25 好很多",
            xy=(0, 1.0), xytext=(2.5, 0.7),
            fontsize=10, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

plt.suptitle("Tanh：零中心 + 更大梯度，但仍有饱和问题", fontsize=13)
plt.tight_layout()
plt.show()

print("Tanh vs Sigmoid 优势：")
print("  1. 零中心输出 (-1,1) → 下一层的输入均值接近0，训练更稳定")
print("  2. 最大梯度 = 1（sigmoid 仅 0.25）→ 梯度消失没那么严重")
print("  缺点：两端仍然饱和，深层网络依旧有梯度消失问题")
print("  现状：主要用在 LSTM/GRU 门控中\n")


# ════════════════════════════════════════════════════════════════════
# 第4部分：ReLU 家族
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第4部分：ReLU 家族")
print("=" * 60)


def relu(x):
    """ReLU 激活函数：max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU 的导数：x>0 时为1，x<0 时为0"""
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU：max(αx, x)，默认 α=0.01"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU 的导数"""
    return np.where(x > 0, 1.0, alpha)


def prelu(x, alpha=0.25):
    """PReLU（参数化ReLU）：α 是可学习参数，这里用固定值演示"""
    return np.where(x > 0, x, alpha * x)


# --- ReLU 家族对比图 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左图：三种 ReLU 变体
ax = axes[0]
ax.plot(x_range, relu(x_range), "b-", linewidth=2.5, label="ReLU")
ax.plot(x_range, leaky_relu(x_range, 0.1), "r--", linewidth=2, label="Leaky ReLU (α=0.1)")
ax.plot(x_range, prelu(x_range, 0.25), "g:", linewidth=2, label="PReLU (α=0.25)")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("ReLU 家族对比", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-2, 8)

# 中图：ReLU 导数
ax = axes[1]
ax.plot(x_range, relu_derivative(x_range), "b-", linewidth=2.5, label="ReLU 导数")
ax.plot(x_range, leaky_relu_derivative(x_range, 0.1), "r--", linewidth=2,
        label="Leaky ReLU 导数")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("导数对比", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f'(x)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.2, 1.5)
# 标注 Dead ReLU 区域
ax.fill_between(x_range[x_range < 0], 0, 0, alpha=0.1, color="red")
ax.annotate("Dead ReLU 区域\n梯度=0, 永远不更新!",
            xy=(-4, 0), xytext=(-5, 0.8),
            fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

# 右图：Dead ReLU 问题可视化
ax = axes[2]
np.random.seed(123)
# 模拟一层网络的权重和输入
n_neurons = 200
inputs = np.random.randn(1000)
weights = np.random.randn(n_neurons) * 2  # 较大的权重
biases = np.random.randn(n_neurons) * 2   # 较大的偏置

# 统计每个神经元在所有输入上 ReLU 输出为零的比例
dead_ratios = []
for i in range(n_neurons):
    pre_activation = weights[i] * inputs + biases[i]
    dead_ratio = np.mean(pre_activation <= 0)
    dead_ratios.append(dead_ratio)

dead_ratios = np.array(dead_ratios)
fully_dead = np.sum(dead_ratios > 0.99)
mostly_dead = np.sum(dead_ratios > 0.8)

ax.hist(dead_ratios, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax.axvline(x=0.99, color="red", linestyle="--", linewidth=2, label=f"完全死亡 (>{fully_dead}个)")
ax.set_title(f"Dead ReLU 统计\n{n_neurons}个神经元中{fully_dead}个完全死亡", fontsize=13)
ax.set_xlabel("输出为零的比例")
ax.set_ylabel("神经元个数")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.suptitle("ReLU：简单高效，但小心 Dead ReLU 问题", fontsize=13)
plt.tight_layout()
plt.show()

print("ReLU 核心要点：")
print("  优势：计算极简（一次比较）、正区间梯度恒为1（不消失）")
print("  缺陷：Dead ReLU——若权重初始化不好或学习率太大，")
print("        神经元可能永远输出0，参数不再更新（"死亡"）")
print(f"  演示中 {n_neurons} 个神经元有 {fully_dead} 个完全死亡，{mostly_dead} 个大部分死亡")
print("  对策：Leaky ReLU（负区间留小梯度）、He 初始化、BatchNorm\n")


# ════════════════════════════════════════════════════════════════════
# 第5部分：GELU 详解
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第5部分：GELU 详解")
print("=" * 60)


def gelu(x):
    """
    GELU 激活函数：x · Φ(x)，其中 Φ 是标准正态累积分布函数。
    精确公式：GELU(x) = x · 0.5 · (1 + erf(x / sqrt(2)))
    直觉：以概率 Φ(x) 保留输入 x，概率 1-Φ(x) 丢弃（输出0）
    """
    return x * 0.5 * (1 + _erf(x / np.sqrt(2)))


def gelu_approx(x):
    """GELU 的 tanh 近似（PyTorch 中常用）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x, dx=1e-5):
    """GELU 导数（数值近似）"""
    return (gelu(x + dx) - gelu(x - dx)) / (2 * dx)


fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- 左图：GELU vs ReLU ---
ax = axes[0]
ax.plot(x_range, gelu(x_range), "b-", linewidth=2.5, label="GELU（精确）")
ax.plot(x_range, gelu_approx(x_range), "c--", linewidth=1.5, alpha=0.7,
        label="GELU（tanh近似）")
ax.plot(x_range, relu(x_range), "r:", linewidth=2, alpha=0.5, label="ReLU")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("GELU vs ReLU", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 8)

# 标注 GELU 的关键特点
ax.annotate("GELU 允许小负值通过\n（ReLU 直接截断为0）",
            xy=(-1.5, gelu(-1.5)), xytext=(-6, 2),
            fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

# --- 中图：GELU 导数 vs ReLU 导数 ---
ax = axes[1]
ax.plot(x_range, gelu_derivative(x_range), "b-", linewidth=2.5, label="GELU 导数")
ax.plot(x_range, relu_derivative(x_range), "r:", linewidth=2, alpha=0.5,
        label="ReLU 导数")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("导数对比", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f'(x)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.5)
ax.annotate("GELU 导数平滑连续\n（ReLU 在0处不可导）",
            xy=(0, 0.5), xytext=(2, 1.2),
            fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

# --- 右图：概率门控解释 ---
ax = axes[2]
# Φ(x) 就是"保留概率"
phi_x = 0.5 * (1 + _erf(x_range / np.sqrt(2)))
ax.plot(x_range, phi_x, "g-", linewidth=2.5, label="Φ(x)：保留概率")
ax.fill_between(x_range, 0, phi_x, alpha=0.15, color="green")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("GELU 的概率性门控", fontsize=14)
ax.set_xlabel("x（输入值）")
ax.set_ylabel("Φ(x)（通过概率）")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.text(-4, 0.7, "大负值 → 高概率被丢弃", fontsize=10, color="red")
ax.text(2, 0.3, "大正值 → 高概率被保留", fontsize=10, color="green")
ax.text(-1, 0.45, "x=0 → 50%概率", fontsize=9, color="gray")

plt.suptitle("GELU：Transformer 家族（BERT、GPT）的默认激活函数", fontsize=13)
plt.tight_layout()
plt.show()

print("GELU 核心要点：")
print("  公式：GELU(x) = x · Φ(x)，Φ 是标准正态CDF")
print("  直觉：输入 x 越大（越"重要"），越可能被保留")
print("        像一个"概率性开关"——不是非黑即白，而是渐变式")
print("  相比 ReLU 的优势：")
print("    1. 平滑可导（ReLU 在 x=0 处不可导）")
print("    2. 允许小负值通过（ReLU 直接截断）")
print("    3. 实验证明在 Transformer 中效果更好")
print("  使用场景：BERT、GPT、ViT 等 Transformer 模型\n")


# ════════════════════════════════════════════════════════════════════
# 第6部分：Swish 和 SwiGLU
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第6部分：Swish 和 SwiGLU")
print("=" * 60)


def swish(x, beta=1.0):
    """
    Swish（也叫 SiLU）：x · σ(βx)
    Google 通过自动搜索发现的激活函数。
    β=1 时就是 SiLU，和 GELU 非常相似。
    """
    return x * sigmoid(beta * x)


def swish_derivative(x, beta=1.0):
    """Swish 的导数：σ(βx) + βx·σ(βx)·(1-σ(βx))"""
    s = sigmoid(beta * x)
    return s + beta * x * s * (1 - s)


fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# --- 左图：Swish 不同 β 值 ---
ax = axes[0]
betas = [0.5, 1.0, 2.0, 5.0]
colors_beta = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
for beta, color in zip(betas, colors_beta):
    ax.plot(x_range, swish(x_range, beta), color=color, linewidth=2,
            label=f"Swish (β={beta})")
ax.plot(x_range, relu(x_range), "k:", linewidth=1.5, alpha=0.4, label="ReLU")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("Swish 随 β 的变化", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 8)
ax.text(3, -1, "β→∞ 时趋近 ReLU", fontsize=9, color="gray",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# --- 中图：自门控机制 ---
ax = axes[1]
ax.plot(x_range, x_range, "gray", linewidth=1, linestyle=":", alpha=0.5, label="x（原始输入）")
ax.plot(x_range, sigmoid(x_range), "r--", linewidth=1.5, label="σ(x)（门控值）")
ax.plot(x_range, swish(x_range), "b-", linewidth=2.5, label="x·σ(x)（Swish）")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("自门控机制分解", fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("值")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.5, 6)
ax.annotate("σ(x) 作为"门"\n决定 x 通过多少",
            xy=(2, sigmoid(2)), xytext=(4, 0.3),
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

# --- 右图：SwiGLU 结构图解（用文字和公式说明） ---
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("SwiGLU 结构（LLaMA/Qwen FFN）", fontsize=14)

# 用文字画流程图
box_style = dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="steelblue",
                 linewidth=2)
arrow_style = dict(arrowstyle="->", color="steelblue", linewidth=2)

ax.text(5, 9.0, "输入 x", fontsize=13, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"))

ax.annotate("", xy=(3, 7.5), xytext=(4.5, 8.5), arrowprops=arrow_style)
ax.annotate("", xy=(7, 7.5), xytext=(5.5, 8.5), arrowprops=arrow_style)

ax.text(3, 7, "x · W1", fontsize=12, ha="center", bbox=box_style)
ax.text(7, 7, "x · W2", fontsize=12, ha="center", bbox=box_style)

ax.annotate("", xy=(3, 5.5), xytext=(3, 6.5), arrowprops=arrow_style)
ax.text(3, 5, "Swish(...)", fontsize=12, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffe0e0", edgecolor="red",
                  linewidth=2))

ax.annotate("", xy=(5, 3.8), xytext=(3.5, 4.5), arrowprops=arrow_style)
ax.annotate("", xy=(5, 3.8), xytext=(6.5, 6.5), arrowprops=arrow_style)

ax.text(5, 3.2, "逐元素相乘 (Hadamard)", fontsize=11, ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e0ffe0", edgecolor="green",
                  linewidth=2))

ax.annotate("", xy=(5, 1.8), xytext=(5, 2.7), arrowprops=arrow_style)
ax.text(5, 1.2, "输出", fontsize=13, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="orange"))

# 公式
ax.text(5, 0.3, "SwiGLU(x) = Swish(x·W1) ⊙ (x·W2)", fontsize=11,
        ha="center", fontweight="bold", color="darkblue")

plt.tight_layout()
plt.show()

print("Swish / SiLU 要点：")
print("  公式：Swish(x) = x · σ(x)")
print("  自门控：输入 x 自己控制"开关"开多大（σ(x) ∈ (0,1)）")
print("  和 GELU 非常接近，实践中效果相当")
print("\nSwiGLU 要点：")
print("  公式：SwiGLU(x, W1, W2) = Swish(x·W1) ⊙ (x·W2)")
print("  双路径：一路做非线性变换（Swish），一路做线性投影（门控）")
print("  逐元素相乘相当于"信息筛选"：线性路径决定通过什么，非线性路径决定通过多少")
print("  使用场景：LLaMA、Qwen、Mistral 等现代大模型的 FFN 层\n")


# ════════════════════════════════════════════════════════════════════
# 第7部分：所有激活函数对比图
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("第7部分：所有激活函数对比图")
print("=" * 60)

x_vis = np.linspace(-5, 5, 500)

# 所有激活函数和导数
activations = {
    "Sigmoid":    (sigmoid(x_vis),            sigmoid_derivative(x_vis)),
    "Tanh":       (tanh(x_vis),               tanh_derivative(x_vis)),
    "ReLU":       (relu(x_vis),               relu_derivative(x_vis)),
    "Leaky ReLU": (leaky_relu(x_vis, 0.1),    leaky_relu_derivative(x_vis, 0.1)),
    "GELU":       (gelu(x_vis),               gelu_derivative(x_vis)),
    "Swish":      (swish(x_vis),              swish_derivative(x_vis)),
}

colors_all = ["#e74c3c", "#e67e22", "#2ecc71", "#27ae60", "#3498db", "#9b59b6"]

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- 上图：所有激活函数 ---
ax = axes[0]
for (name, (f_val, _)), color in zip(activations.items(), colors_all):
    ax.plot(x_vis, f_val, color=color, linewidth=2.5, label=name)
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("所有激活函数对比", fontsize=15)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f(x)", fontsize=12)
ax.legend(fontsize=11, loc="upper left", ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim(-2, 5)

# --- 下图：所有导数 ---
ax = axes[1]
for (name, (_, d_val)), color in zip(activations.items(), colors_all):
    ax.plot(x_vis, d_val, color=color, linewidth=2.5, label=name + " 导数")
ax.axhline(y=0, color="k", linewidth=0.5)
ax.axhline(y=1, color="gray", linestyle=":", alpha=0.3)
ax.axvline(x=0, color="k", linewidth=0.5)
ax.set_title("所有激活函数的导数（梯度）对比", fontsize=15)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("f'(x)", fontsize=12)
ax.legend(fontsize=9, loc="upper left", ncol=3)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 1.5)

plt.tight_layout()
plt.show()

# --- 对比总结表 ---
print("\n激活函数对比速查表：")
print("-" * 80)
print(f"{'函数':<12} {'输出范围':<14} {'零中心':^8} {'梯度消失':^8} {'计算量':^8} {'典型用途'}")
print("-" * 80)
print(f"{'Sigmoid':<12} {'(0, 1)':<14} {'否':^8} {'严重':^8} {'中':^8} {'二分类输出层'}")
print(f"{'Tanh':<12} {'(-1, 1)':<14} {'是':^8} {'中等':^8} {'中':^8} {'LSTM/GRU 门控'}")
print(f"{'ReLU':<12} {'[0, +∞)':<14} {'否':^8} {'无(正区)':^8} {'低':^8} {'CNN/MLP 默认'}")
print(f"{'Leaky ReLU':<12} {'(-∞, +∞)':<14} {'否':^8} {'无':^8} {'低':^8} {'替代 ReLU'}")
print(f"{'GELU':<12} {'≈(-0.17, +∞)':<14} {'否':^8} {'无':^8} {'中':^8} {'Transformer'}")
print(f"{'Swish/SiLU':<12} {'≈(-0.28, +∞)':<14} {'否':^8} {'无':^8} {'中':^8} {'EfficientNet'}")
print(f"{'SwiGLU':<12} {'(-∞, +∞)':<14} {'—':^8} {'无':^8} {'高':^8} {'LLaMA/Qwen FFN'}")
print("-" * 80)


# ════════════════════════════════════════════════════════════════════
# 第8部分：表达能力实验
# ════════════════════════════════════════════════════════════════════
#
# 实验设计：用简单网络拟合非线性函数 y = sin(x)
#   - 无激活函数（纯线性网络）→ 只能拟合直线
#   - 有激活函数（ReLU / GELU）→ 可以拟合曲线
# 训练方式：手写梯度下降（与第1章第1节一致）
#

print("\n" + "=" * 60)
print("第8部分：表达能力实验 —— 拟合 sin(x)")
print("=" * 60)


def make_sin_data(n=200):
    """生成 sin(x) 数据"""
    x = np.linspace(-2 * np.pi, 2 * np.pi, n)
    y = np.sin(x)
    return x.reshape(-1, 1), y.reshape(-1, 1)


class SimpleNetwork:
    """
    简单两层网络，手写前向和反向传播。
    结构：输入(1) → 隐藏层(hidden_dim) → 输出(1)

    参数:
        hidden_dim   : 隐藏层神经元数
        activation   : 激活函数名，'none'/'relu'/'gelu'
    """
    def __init__(self, hidden_dim=32, activation="relu"):
        self.activation = activation
        # He 初始化（适用于 ReLU 类激活函数）
        self.W1 = np.random.randn(1, hidden_dim) * np.sqrt(2.0 / 1)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))

    def _activate(self, z):
        """前向：激活函数"""
        if self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "gelu":
            return z * 0.5 * (1 + _erf(z / np.sqrt(2)))
        else:  # 无激活函数
            return z

    def _activate_grad(self, z):
        """反向：激活函数导数"""
        if self.activation == "relu":
            return (z > 0).astype(float)
        elif self.activation == "gelu":
            # GELU 导数的解析近似
            phi = 0.5 * (1 + _erf(z / np.sqrt(2)))
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            return phi + z * pdf
        else:
            return np.ones_like(z)

    def forward(self, x):
        """前向传播，保存中间值用于反向传播"""
        self.x = x                              # (n, 1)
        self.z1 = x @ self.W1 + self.b1         # (n, hidden)
        self.a1 = self._activate(self.z1)       # (n, hidden)
        self.z2 = self.a1 @ self.W2 + self.b2   # (n, 1)
        return self.z2

    def backward(self, y_pred, y_true, lr=0.001):
        """反向传播 + 参数更新"""
        n = y_true.shape[0]
        # 输出层梯度
        dL_dz2 = 2 * (y_pred - y_true) / n          # (n, 1)
        dL_dW2 = self.a1.T @ dL_dz2                 # (hidden, 1)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # 隐藏层梯度
        dL_da1 = dL_dz2 @ self.W2.T                 # (n, hidden)
        dL_dz1 = dL_da1 * self._activate_grad(self.z1)  # (n, hidden)
        dL_dW1 = self.x.T @ dL_dz1                  # (1, hidden)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # 参数更新
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1

    def train(self, x, y, epochs=2000, lr=0.01):
        """训练循环"""
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            self.backward(y_pred, y, lr=lr)
            if (epoch + 1) % 500 == 0:
                print(f"    Epoch {epoch+1:>5d}/{epochs} | MSE = {loss:.6f}")
        return losses


# --- 训练三种网络 ---
X_sin, Y_sin = make_sin_data(200)
configs = [
    ("无激活函数（纯线性）", "none",  0.001),
    ("ReLU 激活",          "relu",  0.005),
    ("GELU 激活",          "gelu",  0.005),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
all_losses = {}

for idx, (title, act, lr) in enumerate(configs):
    print(f"\n--- 训练网络: {title} ---")
    np.random.seed(42)  # 每次重置种子保证公平比较
    net = SimpleNetwork(hidden_dim=64, activation=act)
    losses = net.train(X_sin, Y_sin, epochs=3000, lr=lr)
    all_losses[title] = losses

    # 预测并绘图
    Y_pred = net.forward(X_sin)
    ax = axes[idx]
    ax.plot(X_sin, Y_sin, "b-", linewidth=1.5, alpha=0.6, label="sin(x) 真实值")
    ax.plot(X_sin, Y_pred, "r-", linewidth=2, label="网络预测")
    ax.set_title(f"{title}\n最终 MSE = {losses[-1]:.6f}", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

plt.suptitle("表达能力实验：有无激活函数拟合 sin(x) 的差异", fontsize=14)
plt.tight_layout()
plt.show()

# --- 损失曲线对比 ---
plt.figure(figsize=(10, 5))
for name, losses in all_losses.items():
    plt.plot(losses, linewidth=2, label=name)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE 损失", fontsize=12)
plt.title("三种网络的训练损失对比", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale("log")
plt.tight_layout()
plt.show()

print("\n实验结论：")
print("  1. 无激活函数 → 只能拟合直线，无法逼近 sin(x)")
print("  2. ReLU 激活   → 用分段线性逼近曲线，效果不错但有棱角")
print("  3. GELU 激活   → 平滑逼近，拟合效果最好")
print("  核心洞察：激活函数 = 非线性 = 表达能力的来源！")


# ════════════════════════════════════════════════════════════════════
# 第9部分：思考题
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【梯度消失的量化分析】
   假设一个 20 层的深度网络，每层都用 Sigmoid 激活函数。
   Sigmoid 导数最大值为 0.25，那么梯度从最后一层传到第一层
   最多衰减为多少？如果换成 ReLU 呢？
   请计算 0.25^19 和 1^19，体会两者的差距。
   延伸：这就是为什么深度网络发展了 30 年才成功——
   直到 ReLU 的出现才真正解决了梯度消失问题。

2. 【GELU vs ReLU 的边界行为】
   当 x = -0.5 时，ReLU(x) = 0（完全丢弃），
   但 GELU(-0.5) ≈ -0.154（保留一部分负信息）。
   在 NLP 中，为什么保留小负值可能是有益的？
   提示：想想词嵌入空间中，"轻微不相关"和"完全无关"的区别。

3. 【SwiGLU 的参数量】
   传统 FFN：y = W2 · ReLU(W1·x + b1) + b2
   假设输入维度 d=4096，FFN 中间维度 d_ff=11008。
   SwiGLU 需要三个矩阵（W1, W2, W_gate），参数量比传统 FFN 多多少？
   LLaMA 为什么选择增加参数量也要用 SwiGLU？
   提示：效果提升 > 参数量增加的代价。

4. 【激活函数与初始化的关系】
   ReLU 网络通常使用 He 初始化（方差 = 2/n），
   而 Sigmoid/Tanh 网络使用 Xavier 初始化（方差 = 1/n）。
   为什么不同激活函数需要不同的初始化方案？
   提示：考虑激活函数在 x=0 附近的斜率。

5. 【动手实验】
   修改第8部分的 SimpleNetwork，将隐藏层从 1 层增加到 3 层，
   分别测试 Sigmoid、ReLU、GELU 三种激活函数拟合 sin(x) 的效果。
   观察：哪种激活函数在深层网络中最容易训练？哪种最容易失败？
   这个实验会让你直观感受到"梯度消失"和"激活函数选择"的重要性。
""")


# ════════════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了激活函数的全部核心知识：

  1. 为什么需要非线性：多层线性=一层线性，激活函数是表达能力的来源
  2. Sigmoid：历史重要但已过时，梯度消失+非零中心
  3. Tanh：零中心改善了 Sigmoid，但仍有饱和问题
  4. ReLU：简单有效，深度学习的默认选择，小心 Dead ReLU
  5. GELU：Transformer 的默认激活，平滑+概率性门控
  6. Swish/SwiGLU：现代大模型的主流选择，自门控机制

激活函数的演进历史 = 深度学习的发展史：
  Sigmoid (1980s) → Tanh (1990s) → ReLU (2010s) → GELU/Swish (2017+) → SwiGLU (2020+)

下一节预告: 第2章 · 第3节 · 损失函数与优化器
""")
