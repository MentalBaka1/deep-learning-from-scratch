"""
====================================================================
第1章 · 第1节 · 线性回归
====================================================================

【一句话总结】
线性回归是最简单的"学习"算法——它教会你损失函数、梯度下降、过拟合
这些贯穿整个深度学习的核心概念。

【为什么深度学习需要这个？】
- 神经网络的每一层本质上就是 y = Wx + b（线性变换）
- MSE 损失函数、梯度下降的训练循环——和训练 GPT 是同一个框架
- 从最简单的模型开始，理解"学习"到底在做什么

【核心概念】

1. 模型（Model）
   - 一个函数 y = wx + b，w 是权重（weight），b 是偏置（bias）
   - 类比：y = wx + b 就是一条直线，"学习"就是找最好的那条

2. 损失函数（Loss Function）
   - MSE = (1/N) Σ(y_pred - y_true)²
   - 衡量预测和真实值的差距，越小越好
   - 类比：考试分数，损失越小，学得越好

3. 梯度下降（Gradient Descent）
   - 对损失函数求导 → 得到梯度 → 沿负梯度方向更新参数
   - w = w - lr × ∂L/∂w
   - 类比：闭眼下山，摸到哪边低就往哪边走

4. 闭式解 vs 迭代解
   - 闭式解：w = (X^T X)^{-1} X^T y，一步到位但数据大时计算量爆炸
   - 迭代解：梯度下降，一步步逼近，可处理大数据

5. 过拟合的直觉
   - 模型太复杂（参数太多）→ 记住了训练数据的噪声
   - 训练误差很低但测试误差很高 → 过拟合

【前置知识】
第0章 - 数学基础（向量、梯度、梯度下降）
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 第1部分：生成数据
# ════════════════════════════════════════════════════════════════════
def generate_data(n_samples=100, w_true=3.5, b_true=2.0, noise_std=1.5):
    """
    生成带噪声的线性数据：y = w_true * x + b_true + 噪声

    参数:
        n_samples : 样本数量
        w_true    : 真实斜率（权重）
        b_true    : 真实截距（偏置）
        noise_std : 噪声的标准差，越大数据越散

    返回: X 形状 (n,) 的特征, y 形状 (n,) 的标签
    """
    X = np.random.uniform(-3, 3, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)  # 高斯噪声
    y = w_true * X + b_true + noise
    return X, y


print("=" * 60)
print("第1部分：生成数据")
print("=" * 60)

X_all, y_all = generate_data()

# --- 可视化原始数据 ---
plt.figure()
plt.scatter(X_all, y_all, alpha=0.6, edgecolors="k", linewidths=0.5, label="数据点")
plt.xlabel("x（特征）")
plt.ylabel("y（标签）")
plt.title("生成的线性数据 (y = 3.5x + 2.0 + 噪声)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 划分训练集和测试集（80% 训练，20% 测试）
indices = np.random.permutation(len(X_all))
split = int(0.8 * len(X_all))
train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]
print(f"训练集: {len(X_train)} 个样本 | 测试集: {len(X_test)} 个样本")


# ════════════════════════════════════════════════════════════════════
# 第2部分：闭式解（Normal Equation / 正规方程）
# ════════════════════════════════════════════════════════════════════
# 数学推导: L = ||Xw - y||², 令 ∂L/∂w = 0 → w = (X^T X)^{-1} X^T y

def normal_equation(X, y):
    """
    用正规方程直接求解线性回归最优参数。

    返回: w（权重/斜率）, b（偏置/截距）
    """
    # 构造设计矩阵：加一列全 1（对应偏置 b）
    X_design = np.column_stack([X, np.ones(len(X))])  # 形状 (n, 2)
    # 正规方程：theta = (X^T X)^{-1} X^T y
    theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return theta[0], theta[1]


print("\n" + "=" * 60)
print("第2部分：闭式解（正规方程）")
print("=" * 60)

w_closed, b_closed = normal_equation(X_train, y_train)
print(f"闭式解结果: w = {w_closed:.4f}, b = {b_closed:.4f}")
print(f"真实参数:   w = 3.5000, b = 2.0000")

# --- 可视化闭式解拟合效果 ---
plt.figure()
plt.scatter(X_train, y_train, alpha=0.5, label="训练数据", edgecolors="k", linewidths=0.5)
x_line = np.linspace(-3.5, 3.5, 100)
plt.plot(x_line, w_closed * x_line + b_closed, "r-", linewidth=2,
         label=f"闭式解: y={w_closed:.2f}x+{b_closed:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("闭式解（正规方程）拟合结果")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════════════════════════
# 第3部分：手写梯度下降
# ════════════════════════════════════════════════════════════════════
# 核心循环（和训练神经网络完全相同的框架！）:
#   for 每一步:
#       1. 前向传播：y_pred = w * x + b
#       2. 计算损失：L = mean((y_pred - y)²)
#       3. 计算梯度：dL/dw, dL/db
#       4. 更新参数：w -= lr * dL/dw,  b -= lr * dL/db

def compute_mse(y_pred, y_true):
    """计算均方误差（MSE）"""
    return np.mean((y_pred - y_true) ** 2)


def gradient_descent(X, y, lr=0.05, n_iters=200, verbose=True):
    """
    用梯度下降训练线性回归模型。

    参数:
        X, y    : 训练数据
        lr      : 学习率（步长大小）
        n_iters : 迭代次数
        verbose : 是否打印训练进度

    返回: w_hist, b_hist, loss_hist（每步的参数和损失记录）
    """
    n = len(X)
    # 随机初始化参数
    w, b = np.random.randn() * 0.5, np.random.randn() * 0.5
    w_hist, b_hist, loss_hist = [w], [b], []

    for i in range(n_iters):
        # 第1步：前向传播
        y_pred = w * X + b
        # 第2步：计算损失
        loss = compute_mse(y_pred, y)
        loss_hist.append(loss)
        # 第3步：计算梯度
        # ∂L/∂w = (2/N) Σ (y_pred - y) * x,  ∂L/∂b = (2/N) Σ (y_pred - y)
        error = y_pred - y
        dw = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)
        # 第4步：更新参数
        w, b = w - lr * dw, b - lr * db
        w_hist.append(w)
        b_hist.append(b)

        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"  迭代 {i+1:>4d}/{n_iters} | 损失: {loss:.4f} | "
                  f"w: {w:.4f} | b: {b:.4f}")

    return w_hist, b_hist, loss_hist


print("\n" + "=" * 60)
print("第3部分：手写梯度下降")
print("=" * 60)

w_hist, b_hist, loss_hist = gradient_descent(X_train, y_train, lr=0.05, n_iters=200)
w_gd, b_gd = w_hist[-1], b_hist[-1]
print(f"\n梯度下降结果: w = {w_gd:.4f}, b = {b_gd:.4f}")
print(f"闭式解结果:   w = {w_closed:.4f}, b = {b_closed:.4f}")
print("差距很小！两种方法殊途同归。")


# ════════════════════════════════════════════════════════════════════
# 第4部分：训练过程可视化
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("第4部分：训练过程可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：损失曲线 ---
axes[0].plot(loss_hist, color="steelblue", linewidth=1.5)
axes[0].set_xlabel("迭代次数")
axes[0].set_ylabel("MSE 损失")
axes[0].set_title("训练损失曲线")
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f"初始: {loss_hist[0]:.2f}", xy=(0, loss_hist[0]),
                 fontsize=9, xytext=(30, loss_hist[0] * 0.9),
                 arrowprops=dict(arrowstyle="->", color="gray"))
axes[0].annotate(f"最终: {loss_hist[-1]:.2f}",
                 xy=(len(loss_hist)-1, loss_hist[-1]), fontsize=9,
                 xytext=(len(loss_hist)*0.5, loss_hist[-1]+1),
                 arrowprops=dict(arrowstyle="->", color="gray"))

# --- 右图：不同迭代步的拟合直线 ---
axes[1].scatter(X_train, y_train, alpha=0.4, s=20, label="训练数据")
x_line = np.linspace(-3.5, 3.5, 100)
steps_to_show = [0, 5, 20, 50, 199]  # 选几个关键迭代步
colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(steps_to_show)))
for step, color in zip(steps_to_show, colors):
    axes[1].plot(x_line, w_hist[step] * x_line + b_hist[step],
                 color=color, linewidth=1.5, label=f"第 {step} 步", alpha=0.8)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("梯度下降过程：直线逐步逼近最优")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════════════════════════
# 第5部分：学习率实验
# ════════════════════════════════════════════════════════════════════
# 学习率（lr）是最重要的超参数之一：
#   太小 → 收敛太慢  |  刚好 → 稳步收敛  |  太大 → 震荡/发散

print("\n" + "=" * 60)
print("第5部分：学习率实验")
print("=" * 60)

learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：每个学习率的损失曲线
for lr in learning_rates:
    print(f"\n--- 学习率 lr = {lr} ---")
    _, _, losses = gradient_descent(X_train, y_train, lr=lr, n_iters=200)
    axes[0].plot(np.clip(losses, 0, 100), label=f"lr={lr}", linewidth=1.5)

axes[0].set_xlabel("迭代次数")
axes[0].set_ylabel("MSE 损失（裁剪到100以内）")
axes[0].set_title("不同学习率的损失曲线")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：最终损失柱状图
final_losses = []
for lr in learning_rates:
    _, _, losses = gradient_descent(X_train, y_train, lr=lr, n_iters=200, verbose=False)
    final_losses.append(min(losses[-1], 100))  # 裁剪发散情况

bar_colors = ["#2ecc71" if l < 5 else "#e74c3c" for l in final_losses]
axes[1].bar([f"lr={lr}" for lr in learning_rates], final_losses, color=bar_colors)
axes[1].set_ylabel("最终 MSE 损失")
axes[1].set_title("各学习率的最终损失对比")
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

print("\n关键观察：")
print("  - lr=0.001 太小，200步还没收敛")
print("  - lr=0.05  刚好，快速且稳定地收敛")
print("  - lr=0.5   太大，损失震荡甚至发散")


# ════════════════════════════════════════════════════════════════════
# 第6部分：多项式回归与过拟合
# ════════════════════════════════════════════════════════════════════
# 核心思想：加入 x², x³, ... 特征后能拟合曲线，但阶数太高 → 过拟合
# 过拟合表现：训练误差 ↓↓↓ 但 测试误差 ↑↑↑

def polynomial_features(X, degree):
    """将 X 扩展为多项式特征 [1, x, x², ..., x^degree]，形状 (n, degree+1)"""
    return np.column_stack([X ** d for d in range(degree + 1)])


def fit_polynomial(X_train, y_train, X_test, y_test, degree):
    """
    用正规方程拟合多项式回归，返回参数、训练MSE、测试MSE。
    """
    X_tr_poly = polynomial_features(X_train, degree)
    X_te_poly = polynomial_features(X_test, degree)
    # 加微小正则防止矩阵奇异
    I = np.eye(degree + 1)
    I[0, 0] = 0  # 不正则化偏置项
    theta = np.linalg.inv(X_tr_poly.T @ X_tr_poly + 1e-8 * I) @ X_tr_poly.T @ y_train
    train_mse = compute_mse(X_tr_poly @ theta, y_train)
    test_mse = compute_mse(X_te_poly @ theta, y_test)
    return theta, train_mse, test_mse


print("\n" + "=" * 60)
print("第6部分：多项式回归与过拟合")
print("=" * 60)

# --- 三张子图对比：阶数 1, 3, 15 ---
degrees = [1, 3, 15]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x_plot = np.linspace(X_train.min() - 0.5, X_train.max() + 0.5, 300)

for idx, degree in enumerate(degrees):
    theta, train_mse, test_mse = fit_polynomial(
        X_train, y_train, X_test, y_test, degree)
    y_plot = polynomial_features(x_plot, degree) @ theta

    axes[idx].scatter(X_train, y_train, alpha=0.4, s=20, label="训练数据", zorder=5)
    axes[idx].scatter(X_test, y_test, alpha=0.4, s=20, marker="^",
                      label="测试数据", color="orange", zorder=5)
    axes[idx].plot(x_plot, y_plot, "r-", linewidth=2, label="拟合曲线")
    axes[idx].set_ylim(y_all.min() - 3, y_all.max() + 3)
    axes[idx].set_title(f"阶数 = {degree}\n训练MSE={train_mse:.2f} | 测试MSE={test_mse:.2f}")
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)
    print(f"  阶数 {degree:>2d}: 训练MSE = {train_mse:.4f} | 测试MSE = {test_mse:.4f}")

plt.suptitle("多项式回归：合适 → 略复杂 → 过拟合", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# --- 训练/测试误差随阶数变化的曲线 ---
all_degrees = range(1, 16)
train_errors, test_errors = [], []
for d in all_degrees:
    _, tr_err, te_err = fit_polynomial(X_train, y_train, X_test, y_test, d)
    train_errors.append(tr_err)
    test_errors.append(te_err)

plt.figure(figsize=(10, 6))
plt.plot(list(all_degrees), train_errors, "o-", label="训练误差",
         color="steelblue", linewidth=2)
plt.plot(list(all_degrees), test_errors, "s-", label="测试误差",
         color="coral", linewidth=2)
plt.xlabel("多项式阶数")
plt.ylabel("MSE 损失")
plt.title("模型复杂度 vs 误差：过拟合的经典曲线")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(list(all_degrees))
plt.axvspan(0.5, 2.5, alpha=0.1, color="blue")   # 欠拟合区
plt.axvspan(7.5, 15.5, alpha=0.1, color="red")    # 过拟合区
plt.tight_layout()
plt.show()

print("\n关键观察：")
print("  - 阶数=1: 简单线性模型，拟合合理（数据本身就是线性的）")
print("  - 阶数=3: 稍微灵活一点，效果差不多")
print("  - 阶数=15: 训练误差很低，但曲线剧烈抖动 → 过拟合！")
print("  - 过拟合的标志：训练误差 ↓ 但测试误差 ↑")


# ════════════════════════════════════════════════════════════════════
# 第7部分：完整总结与思考题
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了线性回归的全部核心概念：
  1. 模型:      y = wx + b，最简单的参数化函数
  2. 损失函数:  MSE，衡量预测好坏的标准
  3. 闭式解:    正规方程，一步到位但不可扩展
  4. 梯度下降:  迭代优化，深度学习的通用训练框架
  5. 学习率:    太小收敛慢，太大会发散
  6. 过拟合:    模型太复杂 → 记住噪声 → 测试表现差

这些概念在后续每一章都会反复出现：
  - 神经网络 = 更复杂的模型（但训练循环一样！）
  - 交叉熵 = 分类任务的损失函数（替代 MSE）
  - Adam = 更聪明的梯度下降（替代 vanilla GD）
  - Dropout/正则化 = 对抗过拟合的武器
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【梯度下降 vs 闭式解】
   梯度下降需要迭代几百步才能逼近最优解，闭式解一步就到位。
   那为什么深度学习不用闭式解，而坚持用梯度下降？
   提示：想想当模型变成神经网络时，损失函数还是凸函数吗？
         当数据有十亿条时，(X^T X)^{-1} 的计算量是多少？

2. 【学习率调度】
   实验中我们用了固定学习率。如果训练初期用大学习率，
   后期用小学习率（"学习率衰减"），效果会怎样？
   请修改 gradient_descent 函数，实现 lr = lr_0 / (1 + decay * t)，
   观察损失曲线的变化。

3. 【批量 vs 随机梯度下降】
   我们的梯度下降每步用了全部训练数据（批量梯度下降）。
   如果每步只随机抽一个样本来计算梯度（随机梯度下降, SGD），
   损失曲线会有什么不同？收敛速度呢？
   提示：SGD 的梯度有噪声，但每步计算更快。

4. 【特征缩放的影响】
   如果 x 的范围是 [0, 1000]（比如房价面积），而不是 [-3, 3]，
   梯度下降还能正常工作吗？为什么很多教程都强调要做
   特征标准化（减均值除标准差）？
   提示：画出损失函数关于 w 和 b 的等高线图来理解。

5. 【从线性到非线性】
   线性回归只能拟合直线。如果我们在 y = wx + b 的基础上
   加一个非线性函数 y = σ(wx + b)（比如 σ = sigmoid），
   会发生什么？这和神经网络有什么关系？
   提示：这就是下一节——逻辑回归的核心思想！
""")

print("下一节预告: 第1章 · 第2节 · 逻辑回归与分类")
