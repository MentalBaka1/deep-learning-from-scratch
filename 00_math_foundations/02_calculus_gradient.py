"""
====================================================================
第0章 · 第2节 · 微积分与梯度
====================================================================

【一句话总结】
导数告诉我们"改变输入，输出会怎么变"——这正是训练神经网络的核心：
知道怎么调整权重才能减小损失。

【为什么深度学习需要这个？】
- 训练 = 最小化损失函数，梯度告诉我们损失下降最快的方向
- 没有导数就没有梯度下降，没有梯度下降就没有深度学习
- 理解梯度消失/爆炸问题的前提

【核心概念】
1. 导数（Derivative）
   - 定义：函数在某点的变化率，即切线斜率
   - 直觉：开车时速度表显示的就是位置对时间的导数
   - 公式：f'(x) = lim[h→0] (f(x+h) - f(x)) / h

2. 偏导数（Partial Derivative）
   - 多变量函数对其中一个变量的导数（其他变量固定）
   - 在深度学习中：损失对每个权重的偏导数

3. 梯度（Gradient）
   - 所有偏导数组成的向量
   - 指向函数增长最快的方向（负梯度 = 下降最快）
   - 梯度的大小 = 变化的剧烈程度

4. 梯度下降（Gradient Descent）
   - w_new = w_old - learning_rate × gradient
   - 学习率太大：震荡发散；太小：收敛太慢

5. 数值梯度 vs 解析梯度
   - 数值：(f(x+h) - f(x-h)) / 2h，慢但可靠（用于验证）
   - 解析：手动推导公式，快且精确（用于训练）

【前置知识】
第0章第1节 - 向量与矩阵
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

# ════════════════════════════════════════════════════════════════
# 第1部分 · 导数的直觉
# ════════════════════════════════════════════════════════════════
# 导数 = 切线斜率 = 瞬时变化率
# 数值近似：f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

def numerical_derivative(f, x, h=1e-7):
    """用中心差分法计算 f 在 x 处的数值导数。"""
    return (f(x + h) - f(x - h)) / (2 * h)

def demo_derivative_intuition():
    """演示导数的几何直觉：曲线上某点的切线。"""
    f = lambda x: x**3 - 3*x + 1          # 目标函数
    f_prime = lambda x: 3*x**2 - 3         # 解析导数

    x0, y0, slope = 1.5, f(1.5), f_prime(1.5)
    print("=" * 50)
    print(f"第1部分：导数的直觉\n函数: f(x) = x³ - 3x + 1, 求导点 x₀={x0}")
    print(f"解析导数: {slope:.6f},  数值导数: {numerical_derivative(f, x0):.6f}")

    x = np.linspace(-2.5, 2.5, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 左图：函数 + 切线
    axes[0].plot(x, f(x), "b-", lw=2, label="f(x) = x³ - 3x + 1")
    axes[0].plot(x, slope*(x - x0) + y0, "r--", lw=1.5,
                 label=f"tangent, slope={slope:.2f}")
    axes[0].plot(x0, y0, "ro", ms=8, zorder=5)
    axes[0].set(xlim=(-2.5, 2.5), ylim=(-5, 8), xlabel="x", ylabel="f(x)",
                title="function & tangent line")
    axes[0].legend(fontsize=9)
    # 右图：导数函数
    axes[1].plot(x, f_prime(x), "g-", lw=2, label="f'(x) = 3x² - 3")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].plot(x0, slope, "ro", ms=8, zorder=5)
    axes[1].plot([-1, 1], [0, 0], "k^", ms=10, label="critical points (f'=0)")
    axes[1].set(xlabel="x", ylabel="f'(x)", title="derivative function")
    axes[1].legend(fontsize=9)
    plt.suptitle("Part 1: Derivative Intuition", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("02_part1_derivative.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第2部分 · 偏导数与梯度
# ════════════════════════════════════════════════════════════════
# f(x,y) 的梯度 ∇f = [∂f/∂x, ∂f/∂y]，指向最陡上升方向

def demo_gradient_field():
    """演示二维函数的梯度场与等高线。"""
    # f(x,y) = x² + 2y²,  ∇f = [2x, 4y]
    f = lambda x, y: x**2 + 2*y**2

    print("\n" + "=" * 50)
    print("第2部分：偏导数与梯度\n函数: f(x,y) = x² + 2y²,  ∇f = [2x, 4y]")
    for px, py in [(1, 1), (-2, 0.5), (0, -1)]:
        gx, gy = 2*px, 4*py
        print(f"  点({px:+},{py:+.1f}): ∇f=[{gx:+},{gy:+.1f}], "
              f"|∇f|={np.hypot(gx, gy):.2f}")

    g = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(g, g)
    ag = np.linspace(-2.5, 2.5, 12)
    AX, AY = np.meshgrid(ag, ag)

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X, Y, f(X, Y), levels=20, cmap="coolwarm", alpha=0.7)
    ax.contour(X, Y, f(X, Y), levels=20, colors="k", linewidths=0.3, alpha=0.5)
    plt.colorbar(cs, ax=ax, label="f(x, y)")
    ax.quiver(AX, AY, 2*AX, 4*AY, color="k", alpha=0.7, scale=60, width=0.004)
    ax.set(xlabel="x", ylabel="y", aspect="equal",
           title="Part 2: Gradient Field  (arrows = steepest ascent)")
    plt.tight_layout(); plt.savefig("02_part2_gradient_field.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第3部分 · 梯度下降实现
# ════════════════════════════════════════════════════════════════
# 核心：w_new = w_old - lr * ∇f(w_old)

def gradient_descent(f, grad_f, x0, lr=0.1, n_steps=50):
    """从零实现梯度下降。返回 (n_steps+1, 3) 的轨迹 [x, y, f]。"""
    x, y = float(x0[0]), float(x0[1])
    path = [[x, y, f(x, y)]]
    for _ in range(n_steps):
        gx, gy = grad_f(x, y)
        x, y = x - lr*gx, y - lr*gy
        path.append([x, y, f(x, y)])
    return np.array(path)

def demo_gradient_descent():
    """在 Rosenbrock 变体上演示梯度下降。"""
    # f(x,y) = (1-x)² + 10(y-x²)²，最小值在 (1,1)
    f = lambda x, y: (1-x)**2 + 10*(y - x**2)**2
    grad_f = lambda x, y: (-2*(1-x) - 40*x*(y-x**2), 20*(y-x**2))

    path = gradient_descent(f, grad_f, [-1.5, 2.0], lr=0.005, n_steps=500)
    print("\n" + "=" * 50)
    print(f"第3部分：梯度下降\n函数: (1-x)²+10(y-x²)²,  最优解: (1,1)")
    print(f"初始: ({path[0,0]:.1f},{path[0,1]:.1f}), f={path[0,2]:.2f}")
    print(f"最终: ({path[-1,0]:.4f},{path[-1,1]:.4f}), f={path[-1,2]:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    gx, gy = np.linspace(-2, 2, 300), np.linspace(-1, 3, 300)
    X, Y = np.meshgrid(gx, gy)
    # 左图：等高线 + 路径
    axes[0].contour(X, Y, f(X, Y), levels=np.logspace(-1, 3, 30),
                    cmap="viridis", alpha=0.7)
    axes[0].plot(path[:,0], path[:,1], "r.-", ms=2, lw=0.8, label="GD path")
    axes[0].plot(*path[0,:2], "gs", ms=10, label="start")
    axes[0].plot(1, 1, "r*", ms=15, label="optimum")
    axes[0].set(xlabel="x", ylabel="y", title="trajectory"); axes[0].legend()
    # 右图：损失曲线
    axes[1].semilogy(path[:,2], "b-", lw=1.5)
    axes[1].set(xlabel="iteration", ylabel="loss [log]", title="loss curve")
    plt.suptitle("Part 3: Gradient Descent", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("02_part3_gradient_descent.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第4部分 · 学习率的影响
# ════════════════════════════════════════════════════════════════
# 太大 → 震荡/发散   太小 → 收敛慢   合适 → 稳定快速

def demo_learning_rate_effects():
    """并排对比三种学习率。"""
    f = lambda x, y: x**2 + 5*y**2
    grad_f = lambda x, y: (2*x, 10*y)
    x0, n = [4.0, 2.0], 30

    configs = [
        (0.005, "lr=0.005 (too small)", "blue"),
        (0.08,  "lr=0.08  (just right)", "green"),
        (0.19,  "lr=0.19  (too large)",  "red"),
    ]
    print("\n" + "=" * 50)
    print(f"第4部分：学习率的影响\n函数: x²+5y²,  初始点: {x0}")

    g = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(g, g)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (lr, title, color) in zip(axes, configs):
        path = gradient_descent(f, grad_f, x0, lr=lr, n_steps=n)
        print(f"  {title}: 最终 f={path[-1,2]:.4f}")
        ax.contour(X, Y, f(X, Y), levels=20, cmap="gray", alpha=0.5)
        ax.plot(path[:,0], path[:,1], ".-", color=color, ms=5, lw=1.2)
        ax.plot(*x0, "ks", ms=8); ax.plot(0, 0, "r*", ms=12)
        ax.set(xlim=(-5,5), ylim=(-5,5), xlabel="x", ylabel="y",
               title=title, aspect="equal")
    plt.suptitle("Part 4: Learning Rate", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("02_part4_learning_rate.png"); plt.show()

    # 损失曲线对比
    fig, ax = plt.subplots(figsize=(8, 4))
    for lr, title, color in configs:
        p = gradient_descent(f, grad_f, x0, lr=lr, n_steps=n)
        ax.plot(np.clip(p[:,2], 0, 200), color=color, lw=2, label=title)
    ax.set(xlabel="iteration", ylabel="f(x,y)", title="loss comparison")
    ax.legend()
    plt.tight_layout(); plt.savefig("02_part4_loss_cmp.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第5部分 · 常见激活函数的导数
# ════════════════════════════════════════════════════════════════
# Sigmoid 和 Tanh 在极端值时导数趋近 0 → 梯度消失
# ReLU 正半轴导数恒为 1 → 缓解梯度消失

def sigmoid(x):
    """Sigmoid: 压缩到 (0,1)。"""
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    """σ'(x) = σ(x)(1-σ(x))。"""
    s = sigmoid(x); return s * (1 - s)

def relu(x):
    """ReLU: max(0, x)。"""
    return np.maximum(0, x)

def relu_deriv(x):
    """ReLU 导数: x>0 时为1，否则为0。"""
    return (x > 0).astype(float)

def tanh_deriv(x):
    """tanh'(x) = 1 - tanh²(x)。"""
    return 1 - np.tanh(x)**2

def demo_activation_derivatives():
    """绘制三种激活函数及其导数。"""
    x = np.linspace(-5, 5, 500)
    acts = [("Sigmoid", sigmoid, sigmoid_deriv, "blue"),
            ("Tanh",    np.tanh, tanh_deriv,    "green"),
            ("ReLU",    relu,    relu_deriv,     "red")]

    print("\n" + "=" * 50)
    print("第5部分：激活函数的导数")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for i, (name, fn, dfn, c) in enumerate(acts):
        # 上排：函数值
        axes[0, i].plot(x, fn(x), color=c, lw=2)
        axes[0, i].axhline(0, color="k", lw=0.3); axes[0, i].axvline(0, color="k", lw=0.3)
        axes[0, i].set_title(f"{name}(x)")
        # 下排：导数
        axes[1, i].plot(x, dfn(x), color=c, lw=2, ls="--")
        axes[1, i].axhline(0, color="k", lw=0.3); axes[1, i].axvline(0, color="k", lw=0.3)
        axes[1, i].set_title(f"{name}'(x)")
        # 打印关键值
        d0, d5 = dfn(np.array([0.0]))[0], dfn(np.array([5.0]))[0]
        print(f"  {name:8s}: f'(0)={d0:.4f}, f'(5)={d5:.6f}")

    plt.suptitle("Part 5: Activation Derivatives\n"
                 "(sigmoid/tanh vanish at extremes -> gradient vanishing)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.savefig("02_part5_activations.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第6部分 · 数值梯度验证
# ════════════════════════════════════════════════════════════════
# 解析梯度与数值梯度的相对误差 < 1e-5 → 实现正确

def numerical_gradient_2d(f, x, y, h=1e-5):
    """对 f(x,y) 用中心差分计算数值梯度。"""
    df_dx = (f(x+h, y) - f(x-h, y)) / (2*h)
    df_dy = (f(x, y+h) - f(x, y-h)) / (2*h)
    return df_dx, df_dy

def relative_error(a, b):
    """两个标量的相对误差，< 1e-5 视为通过。"""
    return abs(a - b) / (max(abs(a), abs(b)) + 1e-15)

def demo_gradient_check():
    """用数值梯度验证解析梯度的正确性。"""
    print("\n" + "=" * 50)
    print("第6部分：数值梯度验证")

    # 测试1：f = x² + 3xy + y²,  ∇f = [2x+3y, 3x+2y]
    print("\n测试1: f(x,y) = x² + 3xy + y²")
    f1 = lambda x, y: x**2 + 3*x*y + y**2
    for px, py in [(1.0, 2.0), (-3.0, 0.5)]:
        a = (2*px+3*py, 3*px+2*py)
        n = numerical_gradient_2d(f1, px, py)
        errs = [relative_error(a[i], n[i]) for i in range(2)]
        tag = "PASS" if max(errs) < 1e-5 else "FAIL"
        print(f"  ({px:+},{py:+}): 解析={a}, 数值=({n[0]:.4f},{n[1]:.4f}), "
              f"err={max(errs):.1e} [{tag}]")

    # 测试2：f = sin(x)cos(y) + x²,  ∇f = [cos(x)cos(y)+2x, -sin(x)sin(y)]
    print("测试2: f(x,y) = sin(x)cos(y) + x²")
    f2 = lambda x, y: np.sin(x)*np.cos(y) + x**2
    for px, py in [(0.5, 1.0), (np.pi, np.pi/4)]:
        a = (np.cos(px)*np.cos(py)+2*px, -np.sin(px)*np.sin(py))
        n = numerical_gradient_2d(f2, px, py)
        errs = [relative_error(a[i], n[i]) for i in range(2)]
        tag = "PASS" if max(errs) < 1e-5 else "FAIL"
        print(f"  ({px:.2f},{py:.2f}): err={max(errs):.1e} [{tag}]")

    # 测试3：验证激活函数导数
    print("测试3: 激活函数导数")
    for name, fn, dfn in [("Sigmoid", sigmoid, sigmoid_deriv),
                           ("Tanh", np.tanh, tanh_deriv)]:
        for xi in [-2.0, 0.0, 2.0]:
            a_d = dfn(np.array([xi]))[0]
            n_d = numerical_derivative(fn, xi)
            err = relative_error(a_d, n_d)
            print(f"  {name} x={xi:+}: err={err:.1e} "
                  f"[{'PASS' if err<1e-5 else 'FAIL'}]")

    # 可视化：Sigmoid 导数解析 vs 数值
    x = np.linspace(-4, 4, 200)
    ana = sigmoid_deriv(x)
    num = np.array([numerical_derivative(sigmoid, xi) for xi in x])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x, ana, "b-", lw=2, label="analytical")
    axes[0].plot(x, num, "r--", lw=2, alpha=0.7, label="numerical")
    axes[0].set_title("Sigmoid': analytical vs numerical"); axes[0].legend()
    axes[1].semilogy(x, np.abs(ana - num) + 1e-20, "g-", lw=1.5)
    axes[1].set_title("absolute error (log scale)")
    plt.suptitle("Part 6: Gradient Check", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig("02_part6_grad_check.png"); plt.show()

# ════════════════════════════════════════════════════════════════
# 第7部分 · 思考题
# ════════════════════════════════════════════════════════════════

def thinking_questions():
    """打印思考题与提示。"""
    qs = [
        ("为什么深度学习用梯度下降而不是直接解 ∇f=0？",
         "百万参数、非凸函数、无解析解。迭代逼近不需要全局形式。"),
        ("Sigmoid 导数最大值是多少？对深层网络意味着什么？",
         "σ'(0)=0.25。10层后梯度衰减到 0.25^10≈9.5e-7 → 梯度消失！"),
        ("学习率设为负数会怎样？",
         "w_new = w_old + |lr|*∇f → 梯度上升，用于最大化目标函数。"),
        ("中心差分为什么比前向差分更精确？",
         "Taylor展开：前向差分 O(h)，中心差分 O(h²)，精度高约5个数量级。"),
        ("ReLU 在 x=0 不可微，为什么实践中不影响？",
         "输入恰好为0的概率极低；深度学习只需「几乎处处可微」。"),
    ]
    print("\n" + "=" * 60)
    print("第7部分：思考题")
    print("=" * 60)
    for i, (q, hint) in enumerate(qs, 1):
        print(f"\n问题 {i}: {q}\n  提示: {hint}")

# ════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  第0章 · 第2节 · 微积分与梯度")
    print("  Calculus & Gradient for Deep Learning")
    print("=" * 56)

    demo_derivative_intuition()     # 1. 导数直觉
    demo_gradient_field()           # 2. 偏导数与梯度
    demo_gradient_descent()         # 3. 梯度下降
    demo_learning_rate_effects()    # 4. 学习率影响
    demo_activation_derivatives()   # 5. 激活函数导数
    demo_gradient_check()           # 6. 数值梯度验证
    thinking_questions()            # 7. 思考题

    print("\n" + "=" * 50)
    print("本节结束！下一节：概率与信息论")
    print("=" * 50)
