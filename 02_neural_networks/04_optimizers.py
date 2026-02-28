"""
==============================================================
第2章 第4节：优化器 —— 如何更聪明地下山？
==============================================================

【为什么需要它？】
最基础的梯度下降（SGD）有很多问题：
  1. 在平坦区域（plateau）走得很慢
  2. 在曲率差异大的方向（椭圆碗）会震荡
  3. 学习率需要手动调整，很难找到合适值
  4. 所有参数用同一个学习率（不合理！）

更聪明的优化器通过记忆历史梯度信息来改进这些问题。

【生活类比】
SGD = 蒙眼睛在山上摸坡度下山，每步都从零开始
Momentum = 像个滚下山的球，有惯性，遇到小坑不会停
RMSprop = 像个有自适应步长的探险者，陡的方向步子小，平的方向步子大
Adam = 结合了 Momentum（惯性）和 RMSprop（自适应步长），最实用

【存在理由】
解决问题：SGD 收敛慢、学习率难调、对不同参数"一视同仁"
核心思想：利用梯度的历史信息，动态调整每个参数的有效学习率
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(42)

# ============================================================
# Part 1: 测试用的损失函数
# ============================================================
print("=" * 50)
print("Part 1: 测试损失曲面")
print("=" * 50)

"""
我们用两个经典测试函数：

1. 椭圆碗（Beale 简化版）：f(w1, w2) = w1² + 10*w2²
   - 两个方向曲率差10倍
   - SGD 会在 w2 方向震荡（步子太大）

2. Rosenbrock 函数（"香蕉函数"）：f(x,y) = (1-x)² + 100*(y-x²)²
   - 全局最小值在 (1,1)
   - 有一个弯曲的山谷，算法难以直接找到最低点
   - 经典优化测试基准
"""

def ellipse_bowl(w):
    """椭圆碗：曲率差异大，SGD 容易震荡"""
    return w[0]**2 + 10 * w[1]**2

def ellipse_bowl_grad(w):
    return np.array([2*w[0], 20*w[1]])

def rosenbrock(w):
    """Rosenbrock 香蕉函数：弯曲山谷，经典测试"""
    x, y = w[0], w[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(w):
    x, y = w[0], w[1]
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

print("椭圆碗 f(0,0) =", ellipse_bowl(np.array([0,0])))
print("椭圆碗 f(1,1) =", ellipse_bowl(np.array([1,1])))
print("Rosenbrock 最小值在 (1,1)，f(1,1) =", rosenbrock(np.array([1.0,1.0])))

# ============================================================
# Part 2: 实现四种优化器
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 实现四种优化器")
print("=" * 50)

class SGD:
    """
    随机梯度下降（最基础）
    w = w - lr * grad

    特点：简单，但需要精心调学习率，在曲率差异大时震荡
    """
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, w, grad):
        return w - self.lr * grad

class SGDMomentum:
    """
    带动量的 SGD
    v = β*v - lr*grad   （速度：历史方向 + 当前梯度）
    w = w + v

    类比：滚下山的球
      β=0.9 表示速度保留 90%，10% 被"摩擦"消耗
      在同一方向连续更新 → 速度累积（加速！）
      方向振荡 → 速度互相抵消（稳定！）
    """
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None  # 速度（一开始为0）

    def step(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)
        self.v = self.beta * self.v - self.lr * grad  # 速度更新
        return w + self.v

class RMSprop:
    """
    RMSprop：自适应学习率
    s = decay*s + (1-decay)*grad²    （梯度平方的指数移动平均）
    w = w - lr * grad / (sqrt(s) + ε)

    类比：陡的坡步子小，平的坡步子大
      s 记录"历史上这个方向的梯度有多大"
      如果某方向梯度一直很大（陡），学习率自动缩小
      如果某方向梯度一直很小（平），学习率自动放大
    """
    def __init__(self, lr=0.01, decay=0.9, eps=1e-8):
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.s = None

    def step(self, w, grad):
        if self.s is None:
            self.s = np.zeros_like(w)
        self.s = self.decay * self.s + (1 - self.decay) * grad**2
        return w - self.lr * grad / (np.sqrt(self.s) + self.eps)

class Adam:
    """
    Adam（Adaptive Moment Estimation）：结合了 Momentum + RMSprop
    m = β1*m + (1-β1)*grad           （一阶矩：梯度的指数移动平均，类似Momentum）
    v = β2*v + (1-β2)*grad²          （二阶矩：梯度平方的移动平均，类似RMSprop）
    m_hat = m / (1-β1^t)             （偏差修正：初始时m,v偏小）
    v_hat = v / (1-β2^t)
    w = w - lr * m_hat / (sqrt(v_hat) + ε)

    默认参数：lr=0.001, β1=0.9, β2=0.999, ε=1e-8
    通常是最稳健的默认优化器。
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # 一阶矩
        self.v = None  # 二阶矩
        self.t = 0     # 步数

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2

        # 偏差修正（初始几步m,v估计偏小）
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ============================================================
# Part 3: 在椭圆碗上对比四种优化器
# ============================================================
print("Part 3: 在椭圆碗上优化，对比轨迹")
print("=" * 50)

def run_optimizer(optimizer, grad_fn, start, n_steps=200):
    """运行优化器，记录轨迹"""
    w = start.copy().astype(float)
    trajectory = [w.copy()]
    losses = [ellipse_bowl(w)]

    for _ in range(n_steps):
        grad = grad_fn(w)
        w = optimizer.step(w, grad)
        trajectory.append(w.copy())
        losses.append(ellipse_bowl(w))

    return np.array(trajectory), losses

start = np.array([-4.0, 3.0])
configs = [
    (SGD(lr=0.05),              'SGD (lr=0.05)',        'red'),
    (SGDMomentum(lr=0.05, beta=0.9), 'Momentum (lr=0.05, β=0.9)', 'green'),
    (RMSprop(lr=0.1),           'RMSprop (lr=0.1)',     'blue'),
    (Adam(lr=0.3),              'Adam (lr=0.3)',        'orange'),
]

# 绘制等高线
w1 = np.linspace(-5, 5, 200)
w2 = np.linspace(-4, 4, 200)
W1, W2 = np.meshgrid(w1, w2)
Z = W1**2 + 10*W2**2

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.contour(W1, W2, Z, levels=30, cmap='Blues', alpha=0.5)
ax.contourf(W1, W2, Z, levels=30, cmap='Blues', alpha=0.2)

print(f"{'优化器':<25} {'步数':>6} {'最终Loss':>12}")
print("-" * 45)
for opt, name, color in configs:
    traj, losses = run_optimizer(opt, ellipse_bowl_grad, start, n_steps=200)
    # 找到收敛步数（loss < 0.01）
    converge_step = next((i for i, l in enumerate(losses) if l < 0.01), 200)
    print(f"{name:<25} {converge_step:>6} {losses[-1]:>12.6f}")

    ax.plot(traj[:, 0], traj[:, 1], '-o', color=color, markersize=2,
           linewidth=2, label=name, alpha=0.8)

ax.plot(*start, 'k^', markersize=12, label='起点', zorder=10)
ax.plot(0, 0, 'k*', markersize=15, label='最优点(0,0)', zorder=10)
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 4)
ax.set_title('椭圆碗：四种优化器轨迹对比\n（注意SGD的震荡 vs Adam的直奔目标）')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 损失曲线对比
ax = axes[1]
for opt, name, color in [(SGD(lr=0.05), 'SGD', 'red'),
                          (SGDMomentum(lr=0.05), 'Momentum', 'green'),
                          (RMSprop(lr=0.1), 'RMSprop', 'blue'),
                          (Adam(lr=0.3), 'Adam', 'orange')]:
    _, losses = run_optimizer(opt, ellipse_bowl_grad, start, n_steps=200)
    ax.semilogy(losses, '-', color=color, linewidth=2, label=name)

ax.set_xlabel('步数')
ax.set_ylabel('Loss（对数坐标）')
ax.set_title('收敛速度对比')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_neural_networks/optimizers.png', dpi=100, bbox_inches='tight')
print("\n图片已保存：02_neural_networks/optimizers.png")
plt.show()

# ============================================================
# Part 4: 学习率调度
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 学习率调度 —— 先大后小")
print("=" * 50)

"""
训练策略：学习率不是固定的，而是随时间衰减

直觉：
  - 开始时：用大学习率快速找到大致方向
  - 后来时：用小学习率精细调整，避免在最优点附近震荡

常见调度策略：
  1. 固定学习率（Constant）：简单，不一定最优
  2. 阶梯衰减（Step Decay）：每X轮减半
  3. 余弦退火（Cosine Annealing）：平滑下降
  4. 预热 + 衰减（Warmup + Decay）：Transformer 常用
"""

n_steps = 200
step_range = np.arange(n_steps)

lr_constant = np.ones(n_steps) * 0.1
lr_step = 0.1 * (0.5 ** (step_range // 50))
lr_cosine = 0.1 * (1 + np.cos(np.pi * step_range / n_steps)) / 2
# Transformer 预热调度
warmup = 20
lr_warmup = np.where(step_range < warmup,
                     0.1 * (step_range + 1) / warmup,
                     0.1 * np.sqrt(warmup / (step_range + 1)))

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(lr_constant, 'k-', linewidth=2, label='固定 lr=0.1')
ax.plot(lr_step, 'r-', linewidth=2, label='阶梯衰减（每50步减半）')
ax.plot(lr_cosine, 'b-', linewidth=2, label='余弦退火')
ax.plot(lr_warmup, 'g-', linewidth=2, label='预热+衰减（Transformer用）')
ax.set_xlabel('训练步数')
ax.set_ylabel('学习率')
ax.set_title('学习率调度策略对比')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_neural_networks/lr_schedule.png', dpi=100, bbox_inches='tight')
print("图片已保存：02_neural_networks/lr_schedule.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【偏差修正的必要性】
   Adam 的偏差修正：m_hat = m / (1 - β1^t)
   在 t=1（第一步）时，如果 β1=0.9：
   - m = (1-0.9) * grad = 0.1 * grad
   - m_hat = m / (1-0.9^1) = m / 0.1 = grad
   为什么没有偏差修正时，前几步的更新会太小？

2. 【动量的累积效应】
   用 Momentum 优化器，在一个 1D 函数 f(x) = x² 上：
   从 x=5 开始，lr=0.1，β=0.9，手动计算前 5 步的 x 和 velocity。
   和 SGD 相比，Momentum 在哪一步开始明显更快？

3. 【Rosenbrock 挑战】
   在 Rosenbrock 函数（香蕉函数）上，对比四种优化器：
   - 从 (-1.5, 1.5) 出发
   - 最多运行 5000 步
   - 谁能找到最优点 (1, 1)（f值 < 1e-4）？
   这个函数为什么对优化器是个经典挑战？
""")
