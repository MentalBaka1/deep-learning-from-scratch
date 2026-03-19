"""
====================================================================
第0章 · 第3节 · 概率论与信息论
====================================================================

【一句话总结】
概率论让我们处理不确定性，信息论让我们量化"意外程度"——
交叉熵损失函数就是信息论的直接应用。

【为什么深度学习需要这个？】
- 神经网络输出概率分布（softmax），概率论是基础
- 分类任务的损失函数（交叉熵）来自信息论
- KL散度衡量两个分布的差异，是 VAE、RLHF 的核心工具
- 权重初始化（Xavier/He）基于方差分析
- Dropout 本质是伯努利采样

【核心概念】
1. 概率分布
   - 离散分布：伯努利、分类分布（Categorical）
   - 连续分布：正态分布（初始化）、均匀分布
   - Softmax：将任意向量转为概率分布

2. 期望与方差
   - 期望 = 加权平均，方差 = 波动程度
   - 在深度学习中：BatchNorm 用均值和方差归一化

3. 信息量与熵
   - 信息量：I(x) = -log P(x)，越不可能的事件信息量越大
   - 熵：H(P) = -Σ P(x) log P(x)，分布的"不确定程度"
   - 均匀分布熵最大（最不确定），确定事件熵为0

4. 交叉熵（Cross-Entropy）
   - H(P,Q) = -Σ P(x) log Q(x)
   - 直觉：用分布Q的编码方案来编码分布P的数据，平均需要多少位
   - 在深度学习中：P是真实标签，Q是模型预测，交叉熵越小越好

5. KL散度
   - KL(P||Q) = H(P,Q) - H(P) = Σ P(x) log(P(x)/Q(x))
   - 衡量Q偏离P有多远（不对称！）
   - 在深度学习中：VAE的损失函数、RLHF中约束策略不要偏离太远

【前置知识】
第0章第1节、第2节
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（根据系统环境可能需要调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ====================================================================
# 第一部分：概率分布可视化
# ====================================================================
# 概率分布是深度学习的语言。神经网络的输出经过 softmax 之后就是
# 一个概率分布，表示模型对每个类别的"信心"。
# ====================================================================

def plot_distributions():
    """可视化三种核心分布：伯努利、正态、均匀。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 伯努利分布 ---
    # Dropout 本质就是对每个神经元做伯努利采样
    # P(X=1) = p, P(X=0) = 1-p
    ps = [0.3, 0.5, 0.7]
    x_vals = [0, 1]
    width = 0.2
    for i, p in enumerate(ps):
        probs = [1 - p, p]
        offset = (i - 1) * width
        axes[0].bar([x + offset for x in x_vals], probs, width=width,
                    label=f'p={p}', alpha=0.8)
    axes[0].set_title('伯努利分布 (Dropout 基础)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('P(X=x)')
    axes[0].set_xticks([0, 1])
    axes[0].legend()

    # --- 正态（高斯）分布 ---
    # 权重初始化的核心：Xavier 和 He 初始化都基于正态分布
    # N(μ, σ²)，μ 决定中心位置，σ 决定扩展程度
    x_normal = np.linspace(-5, 5, 300)
    params = [(0, 1, 'μ=0, σ=1 (标准正态)'),
              (0, 0.5, 'μ=0, σ=0.5 (Xavier风格)'),
              (0, 2, 'μ=0, σ=2 (过大初始化)')]
    for mu, sigma, label in params:
        # 手写正态分布概率密度函数，不调用 scipy
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
              np.exp(-0.5 * ((x_normal - mu) / sigma) ** 2)
        axes[1].plot(x_normal, pdf, label=label, linewidth=2)
    axes[1].set_title('正态分布 (权重初始化)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('概率密度 f(x)')
    axes[1].legend(fontsize=8)

    # --- 均匀分布 ---
    # 均匀分布在信息论中有特殊地位：它的熵最大（最不确定）
    intervals = [(-1, 1), (-2, 2), (-3, 3)]
    for a, b in intervals:
        x_unif = np.linspace(a - 0.5, b + 0.5, 300)
        pdf = np.where((x_unif >= a) & (x_unif <= b), 1.0 / (b - a), 0)
        axes[2].plot(x_unif, pdf, label=f'U({a},{b})', linewidth=2)
    axes[2].set_title('均匀分布 (最大熵)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('概率密度 f(x)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('prob_distributions.png', dpi=100, bbox_inches='tight')
    plt.show()


def softmax_demo():
    """
    Softmax：将任意实数向量转为概率分布。
    公式：softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
    这是神经网络分类输出的标准操作。
    """
    def softmax(z):
        """数值稳定版本的 softmax（减去最大值防止溢出）。"""
        z_stable = z - np.max(z)           # 数值稳定性技巧！
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    # 模拟神经网络最后一层的输出（logits）
    logits = np.array([2.0, 1.0, 0.1, -1.0, 3.5])
    probs = softmax(logits)

    print("=" * 60)
    print("Softmax 演示：从 logits 到概率")
    print("=" * 60)
    print(f"  原始 logits:     {logits}")
    print(f"  softmax 概率:    {np.round(probs, 4)}")
    print(f"  概率之和:        {np.sum(probs):.6f} (应该 = 1.0)")
    print(f"  最大概率对应索引: {np.argmax(probs)} (值={probs[np.argmax(probs)]:.4f})")
    print()

    # 可视化：logits vs 概率
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    classes = [f'类别{i}' for i in range(len(logits))]

    axes[0].bar(classes, logits, color='steelblue', alpha=0.8)
    axes[0].set_title('原始 logits（可以是任意实数）')
    axes[0].set_ylabel('logit 值')
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    axes[1].bar(classes, probs, color='coral', alpha=0.8)
    axes[1].set_title('经过 Softmax 后（合法概率分布）')
    axes[1].set_ylabel('概率')
    # 在柱子上方标注概率值
    for i, p in enumerate(probs):
        axes[1].text(i, p + 0.01, f'{p:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('softmax_demo.png', dpi=100, bbox_inches='tight')
    plt.show()


# ====================================================================
# 第二部分：期望与方差
# ====================================================================
# 期望 E[X] = Σ x·P(x) 是"加权平均"
# 方差 Var(X) = E[(X - E[X])²] 衡量"波动程度"
#
# 深度学习关联：
# - Xavier 初始化：Var(W) = 1/n_in（保持前向传播方差不变）
# - He 初始化：Var(W) = 2/n_in（适配 ReLU 激活）
# - BatchNorm：对每个 batch 做 (x - μ) / σ 归一化
# ====================================================================

def expectation_and_variance():
    """计算不同分布的期望和方差，并关联到权重初始化。"""
    print("=" * 60)
    print("期望与方差")
    print("=" * 60)

    # ---- 离散分布的期望和方差 ----
    # 一个简单的骰子（6面）
    outcomes = np.array([1, 2, 3, 4, 5, 6])
    probs = np.ones(6) / 6  # 均匀分布

    # 手动计算期望：E[X] = Σ x_i · P(x_i)
    expectation = np.sum(outcomes * probs)
    # 手动计算方差：Var(X) = E[(X - E[X])²] = Σ (x_i - μ)² · P(x_i)
    variance = np.sum((outcomes - expectation) ** 2 * probs)

    print(f"\n  公平骰子:")
    print(f"    期望 E[X]  = {expectation:.4f}  （直觉：平均值 3.5）")
    print(f"    方差 Var(X) = {variance:.4f}")
    print(f"    标准差 σ    = {np.sqrt(variance):.4f}")

    # ---- 加载骰子（不均匀分布） ----
    biased_probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])  # 偏向6
    exp_biased = np.sum(outcomes * biased_probs)
    var_biased = np.sum((outcomes - exp_biased) ** 2 * biased_probs)
    print(f"\n  加载骰子（偏向6）:")
    print(f"    期望 E[X]  = {exp_biased:.4f}  （比3.5大，因为偏向6）")
    print(f"    方差 Var(X) = {var_biased:.4f}")

    # ---- 权重初始化中的方差 ----
    print(f"\n  ---- 权重初始化与方差 ----")
    n_in = 256   # 输入维度
    n_out = 128  # 输出维度

    # Xavier 初始化：适用于 tanh/sigmoid
    # 方差 = 2 / (n_in + n_out)
    xavier_var = 2.0 / (n_in + n_out)
    xavier_std = np.sqrt(xavier_var)
    w_xavier = np.random.normal(0, xavier_std, size=(n_in, n_out))

    # He 初始化：适用于 ReLU
    # 方差 = 2 / n_in（因为 ReLU 会砍掉一半激活值）
    he_var = 2.0 / n_in
    he_std = np.sqrt(he_var)
    w_he = np.random.normal(0, he_std, size=(n_in, n_out))

    print(f"    Xavier 初始化: σ = {xavier_std:.6f}  "
          f"(理论方差 = {xavier_var:.6f}, 实际方差 = {np.var(w_xavier):.6f})")
    print(f"    He 初始化:     σ = {he_std:.6f}  "
          f"(理论方差 = {he_var:.6f}, 实际方差 = {np.var(w_he):.6f})")

    # 演示：为什么初始化方差很重要
    # 模拟多层前向传播，观察激活值的方差如何变化
    print(f"\n  ---- 模拟10层前向传播 ----")
    print(f"  {'层':>4s}  {'太大初始化':>12s}  {'Xavier':>12s}  {'太小初始化':>12s}")
    x = np.random.randn(32, 256)  # batch=32, dim=256
    x_big, x_xavier, x_small = x.copy(), x.copy(), x.copy()

    for layer in range(10):
        # 三种不同的初始化方差
        w_big = np.random.normal(0, 1.0, (256, 256))      # σ=1，太大
        w_good = np.random.normal(0, xavier_std, (256, 256))  # Xavier
        w_small = np.random.normal(0, 0.01, (256, 256))    # σ=0.01，太小

        x_big = np.tanh(x_big @ w_big)
        x_xavier = np.tanh(x_xavier @ w_good)
        x_small = np.tanh(x_small @ w_small)

        print(f"  {layer + 1:>4d}  "
              f"{np.var(x_big):>12.6f}  "
              f"{np.var(x_xavier):>12.6f}  "
              f"{np.var(x_small):>12.6f}")

    print("  → 太大：方差趋近1（饱和）  Xavier：方差稳定  太小：方差趋近0（梯度消失）")


# ====================================================================
# 第三部分：信息量与熵
# ====================================================================
# 信息量：I(x) = -log₂ P(x)
# 直觉：越不可能的事件，发生时提供的"信息"越多。
#   - 太阳东升（P≈1）：没有信息量
#   - 六合彩中奖（P≈0）：信息量很大！
#
# 熵：H(P) = -Σ P(x) log₂ P(x) = E[I(x)]
# 直觉：一个分布的"平均意外程度"。
#   - 均匀分布：熵最大（最不确定，最"意外"）
#   - 确定事件：熵 = 0（毫无悬念）
# ====================================================================

def information_and_entropy():
    """可视化信息量，计算不同分布的熵。"""
    print("\n" + "=" * 60)
    print("信息量与熵")
    print("=" * 60)

    # ---- 信息量可视化 ----
    # I(x) = -log₂ P(x)
    p_vals = np.linspace(0.001, 1.0, 500)
    info = -np.log2(p_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(p_vals, info, 'b-', linewidth=2)
    axes[0].set_title('信息量 I(x) = -log₂ P(x)')
    axes[0].set_xlabel('概率 P(x)')
    axes[0].set_ylabel('信息量 (bits)')
    # 标注一些关键点
    examples = [(1.0, '必然事件\n(0 bits)'),
                (0.5, '抛硬币\n(1 bit)'),
                (0.125, '掷骰子得1\n(3 bits)')]
    for p, label in examples:
        i = -np.log2(p)
        axes[0].annotate(label, xy=(p, i), fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='red'),
                         xytext=(p + 0.15, i + 0.5))
        axes[0].plot(p, i, 'ro', markersize=6)
    axes[0].grid(True, alpha=0.3)

    # ---- 不同分布的熵 ----
    # 二元分布的熵：H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
    p_binary = np.linspace(0.001, 0.999, 500)
    # 处理边界避免 log(0)
    h_binary = -p_binary * np.log2(p_binary) - \
               (1 - p_binary) * np.log2(1 - p_binary)

    axes[1].plot(p_binary, h_binary, 'g-', linewidth=2)
    axes[1].set_title('二元分布的熵 H(p)')
    axes[1].set_xlabel('P(X=1) = p')
    axes[1].set_ylabel('熵 H (bits)')
    axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.6,
                    label='p=0.5 时熵最大')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('information_entropy.png', dpi=100, bbox_inches='tight')
    plt.show()

    # ---- 数值对比 ----
    def entropy(probs):
        """计算离散分布的熵（以 nats 为单位，用自然对数）。"""
        # 过滤掉 0 概率（0·log(0) 定义为 0）
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    # 不同分布的熵比较
    distributions = {
        '均匀分布 [0.25, 0.25, 0.25, 0.25]': np.array([0.25, 0.25, 0.25, 0.25]),
        '略有偏向 [0.4, 0.3, 0.2, 0.1]':     np.array([0.4, 0.3, 0.2, 0.1]),
        '非常偏向 [0.9, 0.05, 0.03, 0.02]':   np.array([0.9, 0.05, 0.03, 0.02]),
        '完全确定 [1.0, 0.0, 0.0, 0.0]':      np.array([1.0, 0.0, 0.0, 0.0]),
    }

    print(f"\n  {'分布描述':<40s} {'熵 (nats)':>10s}")
    print("  " + "-" * 52)
    for name, dist in distributions.items():
        h = entropy(dist)
        print(f"  {name:<40s} {h:>10.4f}")
    print("  → 均匀分布熵最大（log(4)≈1.3863），确定事件熵为0")


# ====================================================================
# 第四部分：交叉熵详解
# ====================================================================
# H(P, Q) = -Σ P(x) · log Q(x)
#
# 深度学习中最重要的公式之一！
# - P 是真实标签的分布（one-hot）
# - Q 是模型预测的概率分布（softmax输出）
# - 交叉熵越小，说明 Q 越接近 P，模型预测越准
#
# 为什么用交叉熵而不是均方误差(MSE)？
# - 交叉熵的梯度在预测错误时更大 → 学得更快
# - MSE 在 sigmoid/softmax 输出时梯度容易饱和
# ====================================================================

def cross_entropy_explained():
    """一步步拆解交叉熵计算。"""
    print("\n" + "=" * 60)
    print("交叉熵：分类损失函数的核心")
    print("=" * 60)

    def softmax(z):
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    def cross_entropy(p_true, q_pred):
        """
        计算交叉熵 H(P, Q) = -Σ P(x) · log Q(x)。
        p_true: 真实分布（通常是 one-hot）
        q_pred: 预测概率分布
        """
        # 加小常数防止 log(0)
        q_pred = np.clip(q_pred, 1e-12, 1.0)
        return -np.sum(p_true * np.log(q_pred))

    # 模拟一个3分类问题
    # 真实标签：类别1（one-hot 编码）
    p_true = np.array([0, 1, 0])

    # 三个不同"水平"的模型预测
    predictions = {
        '好模型（很自信且正确）': np.array([0.05, 0.90, 0.05]),
        '一般模型（有点对）':      np.array([0.2, 0.5, 0.3]),
        '差模型（预测错误）':      np.array([0.7, 0.1, 0.2]),
    }

    print(f"\n  真实标签 P = {p_true}  （类别1）\n")
    print(f"  {'模型描述':<28s} {'预测 Q':<24s} {'交叉熵 H(P,Q)':>14s}")
    print("  " + "-" * 68)

    for desc, q_pred in predictions.items():
        ce = cross_entropy(p_true, q_pred)
        print(f"  {desc:<28s} {str(np.round(q_pred, 2)):<24s} {ce:>14.4f}")

    print("\n  → 交叉熵越小 = 预测越好。好模型的损失远小于差模型。")

    # 逐步拆解计算过程
    print("\n  ---- 逐步拆解（好模型） ----")
    q = np.array([0.05, 0.90, 0.05])
    print(f"    P = {p_true}")
    print(f"    Q = {q}")
    print(f"    -P[0]·log(Q[0]) = -{p_true[0]}·log({q[0]}) = {-p_true[0] * np.log(q[0]):.4f}")
    print(f"    -P[1]·log(Q[1]) = -{p_true[1]}·log({q[1]}) = {-p_true[1] * np.log(q[1]):.4f}  ← 唯一有贡献的项！")
    print(f"    -P[2]·log(Q[2]) = -{p_true[2]}·log({q[2]}) = {-p_true[2] * np.log(q[2]):.4f}")
    print(f"    总和 = {cross_entropy(p_true, q):.4f}")
    print(f"\n    关键洞察：对于 one-hot 标签，交叉熵 = -log(正确类别的预测概率)")
    print(f"    即 H(P,Q) = -log(Q[正确类别]) = -log({q[1]}) = {-np.log(q[1]):.4f}")

    # 可视化：交叉熵 vs 预测概率
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    q_correct = np.linspace(0.01, 0.99, 200)
    ce_vals = -np.log(q_correct)  # one-hot 情况下简化为 -log(q)

    ax.plot(q_correct, ce_vals, 'r-', linewidth=2)
    ax.set_title('交叉熵 vs 正确类别的预测概率')
    ax.set_xlabel('模型对正确类别的预测概率 Q(正确)')
    ax.set_ylabel('交叉熵损失 = -log Q(正确)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.annotate('预测越自信且正确\n损失越小', xy=(0.9, -np.log(0.9)),
                xytext=(0.6, 1.5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('预测错误\n损失暴涨！', xy=(0.05, -np.log(0.05)),
                xytext=(0.25, 3.5), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cross_entropy.png', dpi=100, bbox_inches='tight')
    plt.show()


# ====================================================================
# 第五部分：KL散度
# ====================================================================
# KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
#            = H(P, Q) - H(P)
#            = 交叉熵 - 真实分布的熵
#
# 性质：
# - KL(P||Q) >= 0，当且仅当 P = Q 时等于 0
# - 不对称！KL(P||Q) ≠ KL(Q||P)
#
# 深度学习应用：
# - VAE 的损失 = 重构损失 + KL(后验 || 先验)
# - RLHF 中用 KL 惩罚防止策略偏离参考模型太远
# - 知识蒸馏中衡量学生模型和教师模型的差异
# ====================================================================

def kl_divergence_demo():
    """计算KL散度，展示其不对称性。"""
    print("\n" + "=" * 60)
    print("KL散度：衡量分布差异")
    print("=" * 60)

    def kl_divergence(p, q):
        """
        计算 KL(P || Q) = Σ P(x) · log(P(x) / Q(x))。
        解读：用 Q 近似 P 时损失了多少信息。
        """
        # 过滤掉 P(x)=0 的项（0·log(0/q) 定义为 0）
        mask = p > 0
        p_valid = p[mask]
        q_valid = np.clip(q[mask], 1e-12, 1.0)
        return np.sum(p_valid * np.log(p_valid / q_valid))

    # 定义两个分布
    p = np.array([0.4, 0.3, 0.2, 0.1])   # "真实" 分布
    q = np.array([0.25, 0.25, 0.25, 0.25])  # 均匀分布（模型猜测）

    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    print(f"\n  P = {p}  （真实分布）")
    print(f"  Q = {q}  （模型猜测 - 均匀分布）")
    print(f"\n  KL(P || Q) = {kl_pq:.4f} nats")
    print(f"  KL(Q || P) = {kl_qp:.4f} nats")
    print(f"  差值       = {abs(kl_pq - kl_qp):.4f}")
    print(f"  → KL散度不对称！KL(P||Q) ≠ KL(Q||P)")

    # 验证：KL = 交叉熵 - 熵
    def entropy(p):
        p_valid = p[p > 0]
        return -np.sum(p_valid * np.log(p_valid))

    def cross_entropy(p, q):
        q_safe = np.clip(q, 1e-12, 1.0)
        return -np.sum(p * np.log(q_safe))

    h_p = entropy(p)
    h_pq = cross_entropy(p, q)
    print(f"\n  验证：KL(P||Q) = H(P,Q) - H(P)")
    print(f"         {kl_pq:.4f}  = {h_pq:.4f} - {h_p:.4f} = {h_pq - h_p:.4f}  ✓")

    # 可视化：两个分布和它们的KL散度
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    categories = ['A', 'B', 'C', 'D']

    # 左图：两个分布对比
    x_pos = np.arange(len(categories))
    width = 0.35
    axes[0].bar(x_pos - width / 2, p, width, label='P (真实)', color='steelblue')
    axes[0].bar(x_pos + width / 2, q, width, label='Q (模型)', color='coral')
    axes[0].set_title(f'P vs Q  |  KL(P||Q)={kl_pq:.4f}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].set_ylabel('概率')

    # 中图：KL随分布差异变化
    # 固定 P，让 Q 从 P 逐渐偏移到均匀分布
    alphas = np.linspace(0, 1, 100)
    q_uniform = np.ones(4) / 4
    kl_values = []
    for alpha in alphas:
        q_interp = (1 - alpha) * p + alpha * q_uniform  # 从 P 线性插值到均匀
        kl_values.append(kl_divergence(p, q_interp))

    axes[1].plot(alphas, kl_values, 'r-', linewidth=2)
    axes[1].set_title('KL(P || Q) 随差异增大的变化')
    axes[1].set_xlabel('α (0=Q等于P, 1=Q为均匀)')
    axes[1].set_ylabel('KL散度 (nats)')
    axes[1].grid(True, alpha=0.3)

    # 右图：不对称性可视化
    # 取不同的 Q，分别计算 KL(P||Q) 和 KL(Q||P)
    kl_forward = []
    kl_reverse = []
    for alpha in alphas:
        q_interp = (1 - alpha) * p + alpha * q_uniform
        kl_forward.append(kl_divergence(p, q_interp))
        kl_reverse.append(kl_divergence(q_interp, p))

    axes[2].plot(alphas, kl_forward, 'b-', linewidth=2, label='KL(P||Q)')
    axes[2].plot(alphas, kl_reverse, 'r--', linewidth=2, label='KL(Q||P)')
    axes[2].set_title('KL散度的不对称性')
    axes[2].set_xlabel('α (Q 偏离 P 的程度)')
    axes[2].set_ylabel('KL散度 (nats)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('kl_divergence.png', dpi=100, bbox_inches='tight')
    plt.show()


# ====================================================================
# 第六部分：Softmax 深入
# ====================================================================
# softmax(z_i) = exp(z_i / T) / Σ_j exp(z_j / T)
#
# 温度参数 T 的效果：
# - T → 0：输出趋近 one-hot（最大值独占概率）
# - T = 1：标准 softmax
# - T → ∞：输出趋近均匀分布
#
# 应用：
# - GPT 生成文本时用温度控制创造性
# - 知识蒸馏中用高温软化教师模型的输出
#
# 数值稳定性：
# - 直接计算 exp(z) 可能溢出（z=1000 → exp(1000) = ∞）
# - 技巧：减去 max(z)，因为 softmax(z) = softmax(z - c)
# ====================================================================

def softmax_deep_dive():
    """温度参数和数值稳定性。"""
    print("\n" + "=" * 60)
    print("Softmax 深入：温度与数值稳定性")
    print("=" * 60)

    def softmax_with_temperature(z, temperature=1.0):
        """带温度参数的 softmax。"""
        z_scaled = z / temperature
        z_stable = z_scaled - np.max(z_scaled)  # 数值稳定性
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    logits = np.array([2.0, 1.0, 0.5, -0.5, -1.0])

    # 不同温度的效果
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n  logits = {logits}\n")
    print(f"  {'温度 T':>8s}  |  {'softmax 输出':<40s}  |  {'最大概率':>8s}  |  {'熵':>8s}")
    print("  " + "-" * 76)

    all_probs = []
    entropies = []
    for T in temperatures:
        probs = softmax_with_temperature(logits, T)
        all_probs.append(probs)
        # 计算熵
        h = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(h)
        print(f"  {T:>8.1f}  |  {str(np.round(probs, 4)):<40s}  |  {np.max(probs):>8.4f}  |  {h:>8.4f}")

    print(f"\n  → T↓ → 更尖锐（更确定） → 低熵")
    print(f"  → T↑ → 更平滑（更随机） → 高熵")

    # 可视化温度效果
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    categories = [f'类别{i}' for i in range(len(logits))]
    selected_temps = [0.1, 0.5, 1.0, 2.0, 10.0]
    x_pos = np.arange(len(logits))
    width = 0.15

    for i, T in enumerate(selected_temps):
        probs = softmax_with_temperature(logits, T)
        axes[0].bar(x_pos + i * width, probs, width, label=f'T={T}', alpha=0.8)

    axes[0].set_title('温度对 Softmax 输出的影响')
    axes[0].set_xlabel('类别')
    axes[0].set_ylabel('概率')
    axes[0].set_xticks(x_pos + width * 2)
    axes[0].set_xticklabels(categories)
    axes[0].legend(fontsize=8)

    # 熵 vs 温度
    t_range = np.linspace(0.05, 15.0, 200)
    h_range = []
    for T in t_range:
        probs = softmax_with_temperature(logits, T)
        h = -np.sum(probs * np.log(probs + 1e-12))
        h_range.append(h)

    max_entropy = np.log(len(logits))  # 均匀分布的熵
    axes[1].plot(t_range, h_range, 'b-', linewidth=2, label='softmax 熵')
    axes[1].axhline(y=max_entropy, color='red', linestyle='--',
                    label=f'最大熵 (均匀) = {max_entropy:.2f}')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('温度 vs 熵')
    axes[1].set_xlabel('温度 T')
    axes[1].set_ylabel('熵 (nats)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('softmax_temperature.png', dpi=100, bbox_inches='tight')
    plt.show()

    # ---- 数值稳定性演示 ----
    print(f"\n  ---- 数值稳定性 ----")
    z_dangerous = np.array([1000, 1001, 999])

    # 不稳定版本
    print(f"\n  logits = {z_dangerous}")
    try:
        exp_raw = np.exp(z_dangerous)
        print(f"  直接 exp(): {exp_raw}  → 溢出！(inf)")
    except Exception as e:
        print(f"  直接 exp(): 溢出错误!")

    # 稳定版本
    z_shifted = z_dangerous - np.max(z_dangerous)  # 减去最大值
    exp_stable = np.exp(z_shifted)
    result = exp_stable / np.sum(exp_stable)
    print(f"  减去 max 后: logits = {z_shifted}")
    print(f"  exp() 结果:  {np.round(exp_stable, 4)}")
    print(f"  softmax:     {np.round(result, 4)}  → 安全！")
    print(f"\n  原理：softmax(z) = softmax(z - c) 对任意常数 c 成立")
    print(f"  证明：exp(z_i - c) / Σ exp(z_j - c)")
    print(f"       = exp(z_i)·exp(-c) / Σ exp(z_j)·exp(-c)")
    print(f"       = exp(z_i) / Σ exp(z_j)  ← exp(-c) 约掉了！")


# ====================================================================
# 第七部分：思考题
# ====================================================================

def print_exercises():
    """输出思考题。"""
    print("\n" + "=" * 60)
    print("思考题")
    print("=" * 60)
    print("""
  1. 为什么分类任务用交叉熵损失而不用均方误差(MSE)？
     提示：考虑梯度的大小。当模型预测完全错误时（比如真实标签=1，
     预测概率=0.01），交叉熵和MSE各自的梯度是多少？

  2. KL散度 KL(P||Q) 为什么是不对称的？请给出一个直觉解释，
     并说明在实际应用中，"前向KL"和"反向KL"分别倾向于什么行为。
     提示：前向KL → Q倾向覆盖P的所有模式（mean-seeking）
           反向KL → Q倾向锁定P的一个模式（mode-seeking）

  3. Softmax 温度 T 在 GPT 文本生成中的作用：
     - T=0.1 时生成的文本有什么特点？
     - T=2.0 时呢？
     - 为什么 T→0 等价于 argmax（贪心解码）？

  4. 证明：对于一个有 n 个类别的离散分布，均匀分布 [1/n, ..., 1/n]
     的熵最大，等于 log(n)。
     提示：使用拉格朗日乘数法，或者利用 KL(P||U) >= 0。

  5. 在 VAE（变分自编码器）中，损失函数包含一个 KL 散度项
     KL(q(z|x) || p(z))，其中 p(z) 通常是标准正态分布 N(0,1)。
     这个 KL 项的直觉意义是什么？如果去掉这个项会发生什么？
     提示：它约束了潜在空间的结构，防止编码器随意编码。
""")


# ====================================================================
# 主函数：运行所有演示
# ====================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  第0章 · 第3节 · 概率论与信息论")
    print("  深度学习教程 - 纯 NumPy 实现")
    print("=" * 60)

    # 第一部分：概率分布可视化
    print("\n>>> 第一部分：概率分布可视化")
    plot_distributions()
    softmax_demo()

    # 第二部分：期望与方差
    print("\n>>> 第二部分：期望与方差")
    expectation_and_variance()

    # 第三部分：信息量与熵
    print("\n>>> 第三部分：信息量与熵")
    information_and_entropy()

    # 第四部分：交叉熵详解
    print("\n>>> 第四部分：交叉熵详解")
    cross_entropy_explained()

    # 第五部分：KL散度
    print("\n>>> 第五部分：KL散度")
    kl_divergence_demo()

    # 第六部分：Softmax深入
    print("\n>>> 第六部分：Softmax 深入")
    softmax_deep_dive()

    # 思考题
    print_exercises()

    print("\n" + "=" * 60)
    print("  本节完成！")
    print("  下一节：第0章 · 第4节 · 链式法则与计算图")
    print("=" * 60)
