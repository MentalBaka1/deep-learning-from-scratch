"""
====================================================================
第2章 · 第4节 · 优化器：SGD → Momentum → Adam
====================================================================

【一句话总结】
优化器决定了"知道梯度后如何更新参数"——Adam 是目前最常用的优化器，
但理解它的演进过程比直接用它更重要。

【为什么深度学习需要这个？】
- 梯度下降的变种有几十种，选择影响训练速度和最终效果
- Adam 是 Transformer/GPT 训练的默认优化器
- 学习率调度（warmup + cosine decay）是大模型训练的标配
- 不理解优化器，就无法调参

【核心概念】

1. 随机梯度下降（SGD）
   - 原始版本：每次用全部数据计算梯度（慢）
   - Mini-batch SGD：每次用一小批数据（快、有噪声但能逃离局部最优）
   - w = w - lr × gradient
   - 问题：在峡谷地形中震荡严重

2. 动量（Momentum）
   - v = β·v + gradient（累积历史梯度）
   - w = w - lr × v
   - 直觉：球滚下山会积累速度
   - 好处：峡谷中减少震荡，加速收敛

3. RMSProp
   - s = β·s + (1-β)·gradient²（累积梯度平方的均值）
   - w = w - lr × gradient / √(s + ε)
   - 直觉：对不同参数自适应调整学习率
   - 梯度大的参数学习率自动变小

4. Adam（Adaptive Moment Estimation）
   - 结合 Momentum（一阶矩）和 RMSProp（二阶矩）
   - m = β1·m + (1-β1)·gradient （动量）
   - v = β2·v + (1-β2)·gradient² （自适应学习率）
   - 偏差修正：m̂ = m/(1-β1^t), v̂ = v/(1-β2^t)
   - w = w - lr × m̂ / (√v̂ + ε)
   - 默认超参：β1=0.9, β2=0.999, ε=1e-8

5. 学习率调度
   - Warmup：从0线性增长到目标学习率（避免初始梯度爆炸）
   - Cosine Decay：余弦曲线下降到接近0
   - Warmup + Cosine Decay 是训练 GPT/LLaMA 的标准配置

【前置知识】
第2章第3节 - MLP与反向传播
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 第1部分：优化器基类与 SGD 实现
# ════════════════════════════════════════════════════════════════════
#
# 所有优化器的共同接口：
#   1. 初始化时传入参数列表和超参数
#   2. step() 方法：根据当前梯度更新参数
#
# SGD 是最简单的优化器，直接沿负梯度方向走一步。
# 但在"峡谷"地形中（一个方向陡、另一个方向缓），SGD 会剧烈震荡。
#

print("=" * 60)
print("第1部分：优化器实现")
print("=" * 60)


class SGD:
    """
    随机梯度下降（Stochastic Gradient Descent）

    更新公式：
        w = w - lr × gradient

    参数:
        lr : 学习率（步长大小）
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """
        执行一步参数更新。

        参数:
            params : 参数列表，每个元素是一个 numpy 数组
            grads  : 梯度列表，与 params 一一对应
        """
        for p, g in zip(params, grads):
            p -= self.lr * g


print("  SGD: w = w - lr * grad")
print("  最简单，但在峡谷地形中震荡严重\n")


# ════════════════════════════════════════════════════════════════════
# 第2部分：Momentum（动量）实现
# ════════════════════════════════════════════════════════════════════
#
# 直觉：想象一个球滚下山坡——它会积累速度。
# 如果连续多步梯度方向一致，速度越来越快（加速收敛）。
# 如果梯度方向频繁变化（峡谷中左右震荡），动量会抵消震荡。
#
# 更新公式：
#   v = β·v + gradient        ← 速度 = 惯性 + 当前推力
#   w = w - lr × v            ← 沿速度方向移动
#
# β（动量系数）通常取 0.9，意味着"记住 90% 的历史速度"。
#

class Momentum:
    """
    带动量的 SGD

    更新公式：
        v = β·v + gradient
        w = w - lr × v

    参数:
        lr   : 学习率
        beta : 动量系数，控制历史速度的衰减（通常 0.9）
    """
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None  # 速度缓存，首次调用时初始化

    def step(self, params, grads):
        """执行一步带动量的参数更新"""
        # 首次调用：为每个参数创建零初始化的速度缓存
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            # 更新速度：惯性 + 当前梯度
            self.velocities[i] = self.beta * self.velocities[i] + g
            # 沿速度方向更新参数
            p -= self.lr * self.velocities[i]


print("  Momentum: v = β·v + grad,  w = w - lr·v")
print("  球滚下山积累速度，减少峡谷震荡\n")


# ════════════════════════════════════════════════════════════════════
# 第3部分：RMSProp 实现
# ════════════════════════════════════════════════════════════════════
#
# 核心思想：自适应学习率——对每个参数单独调整步长。
#
# 如果某个参数的梯度一直很大（说明这个方向很陡），就减小它的学习率。
# 如果某个参数的梯度一直很小（说明这个方向很平），就增大它的学习率。
#
# 实现方式：跟踪梯度平方的指数移动平均（二阶矩），用它来归一化梯度。
#
# 更新公式：
#   s = β·s + (1-β)·gradient²     ← 累积梯度平方的均值
#   w = w - lr × gradient / √(s + ε)  ← 自适应缩放
#

class RMSProp:
    """
    RMSProp（Root Mean Square Propagation）

    更新公式：
        s = β·s + (1-β)·gradient²
        w = w - lr × gradient / √(s + ε)

    参数:
        lr   : 学习率
        beta : 二阶矩的衰减系数（通常 0.999）
        eps  : 防止除以零的小常数
    """
    def __init__(self, lr=0.01, beta=0.999, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sq_grads = None  # 梯度平方的累积缓存

    def step(self, params, grads):
        """执行一步 RMSProp 参数更新"""
        if self.sq_grads is None:
            self.sq_grads = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            # 累积梯度平方的指数移动平均
            self.sq_grads[i] = self.beta * self.sq_grads[i] + (1 - self.beta) * g ** 2
            # 自适应缩放：梯度大的参数步长小，梯度小的参数步长大
            p -= self.lr * g / (np.sqrt(self.sq_grads[i]) + self.eps)


print("  RMSProp: s = β·s + (1-β)·grad²,  w = w - lr·grad/√(s+ε)")
print("  自适应学习率：陡峭方向步子小，平坦方向步子大\n")


# ════════════════════════════════════════════════════════════════════
# 第4部分：Adam 实现
# ════════════════════════════════════════════════════════════════════
#
# Adam = Momentum + RMSProp + 偏差修正
#
# 它同时跟踪：
#   - 一阶矩 m（梯度的均值，即 Momentum）
#   - 二阶矩 v（梯度平方的均值，即 RMSProp）
#
# 偏差修正的意义：
#   m 和 v 初始化为 0，前几步的估计会偏小。
#   除以 (1 - β^t) 可以修正这个偏差，让早期训练更稳定。
#
# 为什么 Adam 这么受欢迎？
#   - 自适应学习率（不同参数不同步长）
#   - 动量加速（利用历史梯度信息）
#   - 对超参数不敏感（默认值通常就能用）
#   - GPT、BERT、LLaMA 都用 Adam（或其变体 AdamW）训练
#

class Adam:
    """
    Adam（Adaptive Moment Estimation）

    更新公式：
        m = β1·m + (1-β1)·gradient       ← 一阶矩（动量）
        v = β2·v + (1-β2)·gradient²      ← 二阶矩（自适应学习率）
        m_hat = m / (1 - β1^t)           ← 偏差修正
        v_hat = v / (1 - β2^t)           ← 偏差修正
        w = w - lr × m_hat / (√v_hat + ε)

    参数:
        lr    : 学习率（通常 1e-3 或 3e-4）
        beta1 : 一阶矩衰减系数（通常 0.9）
        beta2 : 二阶矩衰减系数（通常 0.999）
        eps   : 防止除以零的小常数
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 一阶矩缓存
        self.v = None  # 二阶矩缓存
        self.t = 0     # 时间步（用于偏差修正）

    def step(self, params, grads):
        """执行一步 Adam 参数更新"""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            # 更新一阶矩（动量）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # 更新二阶矩（自适应学习率）
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # 偏差修正：消除零初始化造成的偏差
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


print("  Adam: m = β1·m + (1-β1)·grad,  v = β2·v + (1-β2)·grad²")
print("        偏差修正后: w = w - lr·m̂/(√v̂+ε)")
print("  结合动量和自适应学习率，是目前最常用的优化器")


# ════════════════════════════════════════════════════════════════════
# 第5部分：优化器对比实验 —— 在 Beale 函数上可视化收敛路径
# ════════════════════════════════════════════════════════════════════
#
# 为了直观比较各优化器的行为，我们用经典的测试函数：
#
# Beale 函数: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
#   - 全局最小值在 (3, 0.5)
#   - 地形复杂：有狭窄的峡谷，是优化器的经典压力测试
#
# 我们从同一起点出发，观察不同优化器如何走向最小值。
#

print("\n" + "=" * 60)
print("第5部分：优化器对比实验（Beale 函数）")
print("=" * 60)


def beale(x, y):
    """Beale 函数：经典优化测试函数，最小值在 (3, 0.5)"""
    return ((1.5 - x + x * y) ** 2 +
            (2.25 - x + x * y ** 2) ** 2 +
            (2.625 - x + x * y ** 3) ** 2)


def beale_grad(x, y):
    """Beale 函数的梯度（对 x 和 y 的偏导数）"""
    # df/dx
    dx = (2 * (1.5 - x + x * y) * (-1 + y) +
          2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2) +
          2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3))
    # df/dy
    dy = (2 * (1.5 - x + x * y) * x +
          2 * (2.25 - x + x * y ** 2) * (2 * x * y) +
          2 * (2.625 - x + x * y ** 3) * (3 * x * y ** 2))
    return np.array([dx, dy])


def run_optimizer_on_surface(optimizer, x0, y0, n_steps=500):
    """
    在 2D 函数上运行优化器，记录轨迹。

    参数:
        optimizer : 优化器实例
        x0, y0   : 起始坐标
        n_steps   : 迭代步数

    返回:
        trajectory : 形状 (n_steps+1, 2) 的坐标轨迹
        losses     : 每步的函数值
    """
    # 用 numpy 数组包装参数，使优化器能原地修改
    pos = [np.array([x0]), np.array([y0])]
    trajectory = [[x0, y0]]
    losses = [beale(x0, y0)]

    for _ in range(n_steps):
        x_val, y_val = pos[0][0], pos[1][0]
        grad = beale_grad(x_val, y_val)
        grads = [np.array([grad[0]]), np.array([grad[1]])]

        # 梯度裁剪：防止梯度爆炸导致飞出可视范围
        for g in grads:
            np.clip(g, -10.0, 10.0, out=g)

        optimizer.step(pos, grads)

        # 限制参数范围，防止飞出可视区域
        pos[0][0] = np.clip(pos[0][0], -4.5, 4.5)
        pos[1][0] = np.clip(pos[1][0], -4.5, 4.5)

        trajectory.append([pos[0][0], pos[1][0]])
        losses.append(beale(pos[0][0], pos[1][0]))

    return np.array(trajectory), losses


# --- 运行四种优化器 ---
x0, y0 = 0.0, 0.0  # 统一起点
n_steps = 300

optimizers = {
    "SGD (lr=0.0002)": SGD(lr=0.0002),
    "Momentum (lr=0.0001, β=0.9)": Momentum(lr=0.0001, beta=0.9),
    "RMSProp (lr=0.005)": RMSProp(lr=0.005, beta=0.9),
    "Adam (lr=0.05)": Adam(lr=0.05, beta1=0.9, beta2=0.999),
}

results = {}
for name, opt in optimizers.items():
    traj, losses = run_optimizer_on_surface(opt, x0, y0, n_steps)
    results[name] = (traj, losses)
    final = traj[-1]
    print(f"  {name:35s} → 终点 ({final[0]:.4f}, {final[1]:.4f}), "
          f"最终损失 = {losses[-1]:.6f}")

print(f"  {'真实最小值':35s} → (3.0000, 0.5000), 损失 = 0.000000")

# --- 可视化：等高线图 + 优化轨迹 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：等高线 + 轨迹
ax = axes[0]
xx = np.linspace(-4.5, 4.5, 400)
yy = np.linspace(-4.5, 4.5, 400)
XX, YY = np.meshgrid(xx, yy)
ZZ = beale(XX, YY)

# 用对数尺度显示等高线，更清楚地看到地形细节
levels = np.logspace(-1, 5, 30)
ax.contour(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.6, linewidths=0.5)
ax.contourf(XX, YY, ZZ, levels=levels, cmap="viridis", alpha=0.2)

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
for (name, (traj, _)), color in zip(results.items(), colors):
    # 画路径
    ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.5,
            alpha=0.8, label=name)
    # 标注起点和终点
    ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=8)
    ax.plot(traj[-1, 0], traj[-1, 1], "*", color=color, markersize=14)

# 标注全局最小值
ax.plot(3.0, 0.5, "r+", markersize=20, markeredgewidth=3, label="最小值 (3, 0.5)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("各优化器在 Beale 函数上的收敛路径", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.grid(True, alpha=0.2)

# 右图：损失曲线对比
ax = axes[1]
for (name, (_, losses)), color in zip(results.items(), colors):
    ax.plot(losses, color=color, linewidth=1.5, label=name)

ax.set_xlabel("迭代步数")
ax.set_ylabel("函数值（对数尺度）")
ax.set_title("各优化器的损失收敛曲线", fontsize=13)
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("04_optimizer_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_optimizer_comparison.png")

print("\n关键观察：")
print("  - SGD：步子小且直，在峡谷中进展缓慢")
print("  - Momentum：利用惯性加速，但可能冲过头再回来")
print("  - RMSProp：自适应步长，不同方向步长不同")
print("  - Adam：结合两者优点，路径最平滑，收敛最快")


# ════════════════════════════════════════════════════════════════════
# 第6部分：学习率调度 —— Warmup + Cosine Decay
# ════════════════════════════════════════════════════════════════════
#
# 固定学习率有局限：
#   - 开始太大 → 训练不稳定
#   - 开始太小 → 浪费时间
#   - 结尾太大 → 无法精细收敛
#
# 现代大模型的标配策略：Warmup + Cosine Decay
#   1. Warmup 阶段：学习率从 0 线性增长到目标值（前 5-10% 的步数）
#      - 为什么？初始权重随机，梯度不靠谱，小学习率先稳住
#   2. Cosine Decay 阶段：学习率按余弦曲线平滑下降到接近 0
#      - 为什么？后期需要小步长精细调整，余弦曲线下降比线性更平滑
#
# GPT-3 论文中的配置：warmup 375M tokens，cosine decay 到 10% 的峰值 lr
#

print("\n" + "=" * 60)
print("第6部分：学习率调度")
print("=" * 60)


def warmup_cosine_schedule(total_steps, warmup_steps, peak_lr, min_lr=0.0):
    """
    生成 Warmup + Cosine Decay 学习率调度。

    参数:
        total_steps  : 总训练步数
        warmup_steps : warmup 阶段的步数
        peak_lr      : 峰值学习率
        min_lr       : 最小学习率（cosine decay 的下限）

    返回:
        lrs : 每一步的学习率列表
    """
    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            # Warmup 阶段：线性增长 0 → peak_lr
            lr = peak_lr * step / warmup_steps
        else:
            # Cosine Decay 阶段：peak_lr → min_lr
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        lrs.append(lr)
    return lrs


# --- 可视化不同调度策略 ---
total_steps = 1000
peak_lr = 1e-3

# 策略1：固定学习率（基线）
constant_lrs = [peak_lr] * total_steps

# 策略2：纯 Cosine Decay（没有 warmup）
cosine_only_lrs = warmup_cosine_schedule(total_steps, 0, peak_lr, min_lr=1e-5)

# 策略3：Warmup + Cosine Decay（GPT 标配）
warmup_cosine_lrs = warmup_cosine_schedule(total_steps, 100, peak_lr, min_lr=1e-5)

# 策略4：更长的 Warmup
long_warmup_lrs = warmup_cosine_schedule(total_steps, 300, peak_lr, min_lr=1e-5)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(constant_lrs, label="固定学习率", linewidth=2, linestyle="--", alpha=0.6)
ax.plot(cosine_only_lrs, label="Cosine Decay（无 Warmup）", linewidth=2)
ax.plot(warmup_cosine_lrs, label="Warmup(100步) + Cosine Decay（标配）",
        linewidth=2.5, color="#e74c3c")
ax.plot(long_warmup_lrs, label="Warmup(300步) + Cosine Decay", linewidth=2)

ax.set_xlabel("训练步数")
ax.set_ylabel("学习率")
ax.set_title("学习率调度策略对比\n（GPT/LLaMA 使用 Warmup + Cosine Decay）", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 标注关键阶段
ax.axvline(x=100, color="gray", linestyle=":", alpha=0.5)
ax.annotate("Warmup 结束", xy=(100, peak_lr), fontsize=9,
            xytext=(150, peak_lr * 1.1),
            arrowprops=dict(arrowstyle="->", color="gray"))

plt.tight_layout()
plt.savefig("04_lr_schedule.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_lr_schedule.png")

print("\n学习率调度要点：")
print("  - Warmup：防止初始梯度不稳定导致的训练崩溃")
print("  - Cosine Decay：后期平滑降低，精细调整参数")
print("  - GPT-3/LLaMA 都使用这个策略")


# ════════════════════════════════════════════════════════════════════
# 第7部分：实际效果对比 —— 用不同优化器训练同一个神经网络
# ════════════════════════════════════════════════════════════════════
#
# 理论分析不如实战。我们构造一个简单的二分类任务，
# 用同一个 MLP 结构分别搭配不同优化器训练，对比损失曲线。
#

print("\n" + "=" * 60)
print("第7部分：实际效果对比（MLP 训练）")
print("=" * 60)

# --- 生成月牙形二分类数据 ---
from numpy import pi


def make_moons(n_samples=300, noise=0.15):
    """生成月牙形二分类数据集"""
    n_each = n_samples // 2
    # 上半月牙
    theta1 = np.linspace(0, pi, n_each)
    x1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    # 下半月牙（平移）
    theta2 = np.linspace(0, pi, n_each)
    x2 = np.column_stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5])

    X = np.vstack([x1, x2]) + np.random.randn(n_samples, 2) * noise
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    return X, y


X_data, y_data = make_moons(n_samples=300, noise=0.15)
print(f"  数据集: {X_data.shape[0]} 个样本, {X_data.shape[1]} 个特征, 二分类")


# --- 简易 MLP 实现（手写前向 + 反向传播）---

def sigmoid(z):
    """Sigmoid 激活函数，将任意实数映射到 (0, 1)"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    """ReLU 激活函数"""
    return np.maximum(0, z)


def relu_grad(z):
    """ReLU 的导数"""
    return (z > 0).astype(float)


def binary_cross_entropy(y_pred, y_true):
    """二元交叉熵损失"""
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def train_mlp(X, y, optimizer_class, opt_kwargs, n_epochs=200,
              hidden_dim=16, lr_schedule=None):
    """
    用指定优化器训练一个两层 MLP。

    网络结构: 输入(2) → 隐藏层(hidden_dim, ReLU) → 输出(1, Sigmoid)

    参数:
        X               : 输入数据, 形状 (n, 2)
        y               : 标签, 形状 (n,)
        optimizer_class : 优化器类（SGD / Momentum / RMSProp / Adam）
        opt_kwargs      : 优化器的超参数字典
        n_epochs        : 训练轮数
        hidden_dim      : 隐藏层维度
        lr_schedule     : 可选的学习率调度列表

    返回:
        losses : 每轮的训练损失
    """
    n = X.shape[0]
    np.random.seed(0)  # 固定初始化，确保公平比较

    # 初始化权重（He 初始化）
    W1 = np.random.randn(2, hidden_dim) * np.sqrt(2.0 / 2)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(1)

    params = [W1, b1, W2, b2]
    optimizer = optimizer_class(**opt_kwargs)
    losses = []

    for epoch in range(n_epochs):
        # 如果有学习率调度，动态调整学习率
        if lr_schedule is not None and epoch < len(lr_schedule):
            optimizer.lr = lr_schedule[epoch]

        # ---- 前向传播 ----
        z1 = X @ W1 + b1              # (n, hidden_dim)
        a1 = relu(z1)                  # (n, hidden_dim)
        z2 = a1 @ W2 + b2             # (n, 1)
        a2 = sigmoid(z2)              # (n, 1) — 预测概率
        y_pred = a2.flatten()          # (n,)

        # ---- 计算损失 ----
        loss = binary_cross_entropy(y_pred, y)
        losses.append(loss)

        # ---- 反向传播 ----
        # 输出层梯度
        dz2 = (y_pred - y).reshape(-1, 1) / n   # (n, 1)
        dW2 = a1.T @ dz2                         # (hidden_dim, 1)
        db2 = np.sum(dz2, axis=0)                # (1,)

        # 隐藏层梯度
        da1 = dz2 @ W2.T                         # (n, hidden_dim)
        dz1 = da1 * relu_grad(z1)                # (n, hidden_dim)
        dW1 = X.T @ dz1                          # (2, hidden_dim)
        db1 = np.sum(dz1, axis=0)                # (hidden_dim,)

        grads = [dW1, db1, dW2, db2]

        # ---- 优化器更新 ----
        optimizer.step(params, grads)

        # 同步回来（因为优化器原地修改）
        W1, b1, W2, b2 = params

    return losses


# --- 用四种优化器分别训练 ---
n_epochs = 300
configs = {
    "SGD": (SGD, {"lr": 1.0}),
    "Momentum": (Momentum, {"lr": 0.5, "beta": 0.9}),
    "RMSProp": (RMSProp, {"lr": 0.01, "beta": 0.9}),
    "Adam": (Adam, {"lr": 0.01, "beta1": 0.9, "beta2": 0.999}),
}

all_losses = {}
for name, (opt_cls, opt_kw) in configs.items():
    losses = train_mlp(X_data, y_data, opt_cls, opt_kw, n_epochs=n_epochs)
    all_losses[name] = losses
    print(f"  {name:12s} → 最终损失 = {losses[-1]:.4f}")

# --- 可视化训练损失曲线 ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors_map = {"SGD": "#e74c3c", "Momentum": "#3498db",
              "RMSProp": "#2ecc71", "Adam": "#f39c12"}

# 左图：完整损失曲线
ax = axes[0]
for name, losses in all_losses.items():
    ax.plot(losses, label=name, color=colors_map[name], linewidth=2)
ax.set_xlabel("训练轮数（Epoch）")
ax.set_ylabel("二元交叉熵损失")
ax.set_title("各优化器训练 MLP 的损失曲线", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 右图：前 50 轮放大（看早期收敛速度差异）
ax = axes[1]
for name, losses in all_losses.items():
    ax.plot(losses[:50], label=name, color=colors_map[name], linewidth=2)
ax.set_xlabel("训练轮数（Epoch）")
ax.set_ylabel("二元交叉熵损失")
ax.set_title("前 50 轮放大 —— 早期收敛速度对比", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("04_mlp_optimizer_comparison.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_mlp_optimizer_comparison.png")

# --- 额外实验：Adam + 学习率调度 ---
print("\n--- 额外实验：Adam + Warmup + Cosine Decay ---")
schedule = warmup_cosine_schedule(n_epochs, warmup_steps=30, peak_lr=0.01, min_lr=1e-5)
losses_adam_scheduled = train_mlp(
    X_data, y_data, Adam, {"lr": 0.01}, n_epochs=n_epochs, lr_schedule=schedule
)
print(f"  Adam + 调度 → 最终损失 = {losses_adam_scheduled[-1]:.4f}")
print(f"  Adam 固定lr → 最终损失 = {all_losses['Adam'][-1]:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(all_losses["Adam"], label="Adam（固定 lr=0.01）",
        color="#f39c12", linewidth=2)
ax.plot(losses_adam_scheduled, label="Adam + Warmup + Cosine Decay",
        color="#9b59b6", linewidth=2.5)
ax.set_xlabel("训练轮数（Epoch）")
ax.set_ylabel("二元交叉熵损失")
ax.set_title("学习率调度的效果", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_lr_schedule_effect.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 04_lr_schedule_effect.png")


# ════════════════════════════════════════════════════════════════════
# 第8部分：完整总结与思考题
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了优化器的完整演进路线：

  1. SGD:      最朴素，w = w - lr·grad
               简单但震荡严重，收敛慢

  2. Momentum: 加入惯性，v = β·v + grad, w = w - lr·v
               利用历史梯度加速，减少震荡

  3. RMSProp:  自适应学习率，按梯度大小调整步长
               陡峭方向步子小，平坦方向步子大

  4. Adam:     Momentum + RMSProp + 偏差修正
               最稳健、最常用，GPT/LLaMA 的默认选择

  5. 学习率调度: Warmup + Cosine Decay
               从小到大再到小，大模型训练的标配

实际使用建议：
  - 默认用 Adam，lr=1e-3 或 3e-4
  - 大模型训练用 AdamW（Adam + 权重衰减）+ 学习率调度
  - 只有在特殊场景（如需要更好泛化）才考虑 SGD+Momentum
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【偏差修正的作用】
   Adam 中的偏差修正 m̂ = m/(1-β1^t) 在训练初期影响很大，
   后期几乎没有影响（因为 β1^t → 0）。
   请计算：当 β1=0.9 时，t=1 和 t=100 时的修正系数分别是多少？
   如果去掉偏差修正，训练初期会发生什么？
   提示：t=1 时 1/(1-0.9^1) = 10，意味着修正放大了 10 倍。

2. 【Momentum 的物理直觉】
   动量系数 β=0.9 意味着"记住 90% 的历史速度"。
   如果把 β 改为 0.99 或 0.5，优化轨迹会有什么变化？
   β 太大和太小分别有什么问题？
   提示：β 太大 → 惯性太强，冲过最小值；β 太小 → 回退到 SGD。

3. 【Adam 的超参数敏感性】
   Adam 的论文号称"对超参数不敏感"，但实际中 lr 的选择仍然很关键。
   请尝试用 lr=0.1 和 lr=0.0001 分别训练 MLP，观察损失曲线。
   Adam 对 β1 和 β2 的选择是否同样敏感？

4. 【Warmup 的必要性】
   在训练 Transformer 时，如果不使用 warmup 直接用大学习率，
   训练往往会在前几步就崩溃（loss 变成 NaN）。
   为什么 Transformer 比普通 MLP 更需要 warmup？
   提示：想想 Transformer 中 LayerNorm 和注意力机制在初始随机权重
   下的行为。

5. 【SGD vs Adam 的泛化之争】
   有研究表明 SGD+Momentum 训练的模型有时泛化能力优于 Adam。
   原因可能是什么？在什么场景下你会选择 SGD 而不是 Adam？
   提示：Adam 的自适应学习率可能导致模型走到"尖锐"的最小值，
   而 SGD 的噪声有助于找到"平坦"的最小值。
""")

print("下一节预告: 第2章 · 第5节 · 正则化与过拟合")
