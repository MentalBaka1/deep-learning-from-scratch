"""
==============================================================
第0章 第2节：导数与梯度 —— 从斜率到方向
==============================================================

【为什么需要它？】
神经网络训练的本质是：不断调整参数，让"错误（loss）"越来越小。
怎么知道往哪个方向调？用导数！
导数告诉我们：把参数往这个方向动一点点，loss会怎么变？

【生活类比】
你蒙着眼睛站在山上，想找到山谷（最低点）。
每次用脚踩一踩，感觉脚下的坡度（=导数）。
往下坡方向走（=负梯度方向）。
步子不能太大也不能太小（=学习率）。
这就是梯度下降！
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# Part 1: 导数的本质 —— 瞬时斜率
# ============================================================
print("=" * 50)
print("Part 1: 导数 —— 函数在某点的斜率")
print("=" * 50)

"""
导数的定义：
  f'(x) = lim_{h→0} [f(x+h) - f(x)] / h

直觉：把 x 往右移一点点 h，看 f(x) 变化多少。
变化量 / 移动量 = 斜率 = 导数

导数 > 0：函数在这里是上升的
导数 < 0：函数在这里是下降的
导数 = 0：函数在这里是平的（可能是极值点！）
"""

def f(x):
    """一个测试函数：f(x) = x³ - 3x + 1"""
    return x**3 - 3*x + 1

def f_derivative_numerical(x, h=1e-5):
    """数值导数：用极小的 h 近似导数"""
    return (f(x + h) - f(x - h)) / (2 * h)  # 中心差分更准确

def f_derivative_analytical(x):
    """解析导数：手动推导 f'(x) = 3x² - 3"""
    return 3 * x**2 - 3

# 在 x=2 处计算导数
x_test = 2.0
numerical = f_derivative_numerical(x_test)
analytical = f_derivative_analytical(x_test)
print(f"在 x={x_test} 处：")
print(f"  数值导数 = {numerical:.8f}")
print(f"  解析导数 = {analytical:.8f}")
print(f"  误差 = {abs(numerical - analytical):.2e}  （非常小！）")

# 常见函数的导数
print("\n常见函数及其导数：")
print("  f(x) = x²    → f'(x) = 2x")
print("  f(x) = x³    → f'(x) = 3x²")
print("  f(x) = e^x   → f'(x) = e^x  （指数函数导数还是它自己！）")
print("  f(x) = ln(x) → f'(x) = 1/x")
print("  f(x) = sin(x)→ f'(x) = cos(x)")

# 验证 e^x 的导数
def exp_deriv_check():
    x = 1.5
    numerical = (np.exp(x + 1e-5) - np.exp(x - 1e-5)) / (2e-5)
    analytical = np.exp(x)
    print(f"\n验证 e^x 在 x={x}：数值={numerical:.6f}，解析={analytical:.6f}")
exp_deriv_check()

# ============================================================
# Part 2: 梯度 —— 多变量函数的导数
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 梯度 —— 多变量的偏导数")
print("=" * 50)

"""
当函数有多个变量时（如神经网络有百万个参数），
我们分别对每个变量求导，得到"偏导数"。
所有偏导数组成的向量叫做"梯度"。

梯度的方向 = 函数值上升最快的方向
负梯度方向 = 函数值下降最快的方向  ← 这就是为什么梯度下降要减去梯度！

例子：f(w1, w2) = w1² + w2²（一个碗形曲面）
  ∂f/∂w1 = 2*w1
  ∂f/∂w2 = 2*w2
  梯度 ∇f = [2*w1, 2*w2]

在 (w1=3, w2=4) 处，梯度 = [6, 8]，指向(3,4)的外侧（远离原点）
所以沿负梯度方向走，就是往原点走（下降！）
"""

def loss_surface(w1, w2):
    """二维 loss 曲面（椭圆碗形）"""
    return w1**2 + 2 * w2**2  # 两个方向曲率不同

def gradient(w1, w2):
    """解析梯度：[∂f/∂w1, ∂f/∂w2] = [2*w1, 4*w2]"""
    return np.array([2 * w1, 4 * w2])

def numerical_gradient(f, params, h=1e-5):
    """数值梯度：对每个参数分别做微小扰动"""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        grad[i] = (f(*params_plus) - f(*params_minus)) / (2 * h)
    return grad

w = np.array([3.0, 4.0])
analytical_grad = gradient(w[0], w[1])
numerical_grad = numerical_gradient(loss_surface, w)

print(f"在 (w1={w[0]}, w2={w[1]}) 处：")
print(f"  解析梯度 = {analytical_grad}")
print(f"  数值梯度 = {numerical_grad}")
print(f"  一致！梯度指向上升最快的方向")
print(f"  负梯度 = {-analytical_grad}  （下降最快的方向）")

# ============================================================
# Part 3: 梯度下降 —— 沿负梯度方向走
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 梯度下降 —— 一步一步找最低点")
print("=" * 50)

"""
梯度下降算法：
  1. 初始化参数（随机位置）
  2. 计算梯度（当前位置的坡度）
  3. 沿负梯度方向更新参数
     w = w - lr * ∇f(w)
  4. 重复直到收敛（梯度很小 或 loss不再下降）

lr（learning rate，学习率）= 步长
  太大：可能跨过最低点，来回震荡
  太小：太慢，需要很多步
"""

def gradient_descent(start_w, lr=0.1, n_steps=50):
    """在二维 loss 曲面上做梯度下降"""
    w = np.array(start_w, dtype=float)
    history = [w.copy()]
    losses = [loss_surface(w[0], w[1])]

    for _ in range(n_steps):
        grad = gradient(w[0], w[1])
        w = w - lr * grad  # 核心更新公式！
        history.append(w.copy())
        losses.append(loss_surface(w[0], w[1]))

    return np.array(history), losses

# 从 (3, 4) 开始
history, losses = gradient_descent([3.0, 4.0], lr=0.1, n_steps=30)
print(f"起点：{history[0]}，loss = {losses[0]:.2f}")
print(f"终点：{history[-1].round(4)}，loss = {losses[-1]:.6f}")
print(f"经过 {len(history)-1} 步，收敛到最低点 (0, 0)！")

# ============================================================
# Part 4: 学习率的影响 —— 可视化
# ============================================================

def plot_gradient_descent():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('梯度下降：学习率的影响', fontsize=14)

    # 等高线背景
    w1_range = np.linspace(-4, 4, 100)
    w2_range = np.linspace(-4, 4, 100)
    W1, W2 = np.meshgrid(w1_range, w2_range)
    Z = loss_surface(W1, W2)

    learning_rates = [0.01, 0.1, 0.4]
    titles = ['lr=0.01（太小，太慢）', 'lr=0.1（刚好）', 'lr=0.4（太大，震荡）']

    for ax, lr, title in zip(axes, learning_rates, titles):
        ax.contour(W1, W2, Z, levels=20, cmap='Blues', alpha=0.5)
        ax.contourf(W1, W2, Z, levels=20, cmap='Blues', alpha=0.2)

        hist, _ = gradient_descent([3.0, 2.0], lr=lr, n_steps=50)

        # 画轨迹
        ax.plot(hist[:, 0], hist[:, 1], 'ro-', markersize=4,
               linewidth=1.5, label='参数轨迹', alpha=0.8)
        ax.plot(hist[0, 0], hist[0, 1], 'g^', markersize=12, label='起点')
        ax.plot(0, 0, 'r*', markersize=15, label='最优点(0,0)')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_title(title)
        ax.set_xlabel('w₁')
        ax.set_ylabel('w₂')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('00_math_python/gradient_descent_lr.png', dpi=100, bbox_inches='tight')
    print("\n图片已保存：00_math_python/gradient_descent_lr.png")
    plt.show()

plot_gradient_descent()

# ============================================================
# Part 5: 常见激活函数的导数（为后面做准备）
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 激活函数及其导数")
print("=" * 50)

"""
神经网络中，反向传播需要计算激活函数的导数。
这里提前见一下，后面会深入讲。
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU'(x) = 1 if x > 0 else 0"""
    return (x > 0).astype(float)

x = np.linspace(-4, 4, 200)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('激活函数及其导数', fontsize=14)

for ax, func, deriv, name in zip(
    axes.flat,
    [sigmoid, relu, np.tanh, lambda x: np.where(x > 0, x, 0.01*x)],
    [sigmoid_derivative, relu_derivative,
     lambda x: 1 - np.tanh(x)**2,
     lambda x: np.where(x > 0, 1, 0.01)],
    ['Sigmoid', 'ReLU', 'Tanh', 'Leaky ReLU']
):
    ax.plot(x, func(x), 'b-', linewidth=2, label='函数本身')
    ax.plot(x, deriv(x), 'r--', linewidth=2, label='导数')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.3, 1.2)

plt.tight_layout()
plt.savefig('00_math_python/activation_derivatives.png', dpi=100, bbox_inches='tight')
print("图片已保存：00_math_python/activation_derivatives.png")
plt.show()

print("\n重要观察：")
print("  Sigmoid 导数最大值只有 0.25 → 梯度会衰减！（梯度消失问题）")
print("  ReLU 导数是 0 或 1 → 梯度不会衰减（但负区域永远0！）")
print("  Tanh 导数最大值是 1，但两端趋近0 → 也有轻度梯度消失")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【数值导数】
   对 f(x) = sin(x) 在 x=π/4 处，
   分别用 h=0.1, 0.01, 0.001, 0.0001 计算数值导数。
   和解析导数 cos(π/4) 的误差如何随 h 变化？

2. 【梯度下降实验】
   修改 gradient_descent 函数，在 f(w) = (w-3)² 上（1D）做梯度下降。
   - 解析梯度：f'(w) = 2*(w-3)
   - 从 w=10 开始，lr=0.1，记录每步的 w 和 loss
   - 多少步后 |w-3| < 0.001？

3. 【Sigmoid 梯度消失】
   观察 sigmoid 导数图：在 x>3 或 x<-3 时，导数接近 0。
   如果有 10 层网络，每层都用 sigmoid，
   第10层的梯度传到第1层时，大概缩小了多少倍？
   （提示：10层各自乘以 0.25 是多少？）
""")
