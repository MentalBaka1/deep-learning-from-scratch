"""
==============================================================
第1章 第1节：线性回归 —— 用直线拟合数据
==============================================================

【为什么需要它？】
问题：给你过去100天的气温数据和冰淇淋销量，
     能预测明天气温35度时，冰淇淋会卖多少？

线性回归假设：销量 ≈ a × 气温 + b
训练目标：找到最好的 a 和 b，让预测尽量准。

这是机器学习最基础的模型，也是理解所有 ML 算法的入门。

【生活类比】
拿一把尺子，在散点图上找一条"最代表这些点"的直线。
"最代表" = 所有点到直线的距离之和最小。
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 生成数据 + 可视化问题
# ============================================================
print("=" * 50)
print("Part 1: 问题设定 —— 预测房价")
print("=" * 50)

"""
任务：根据房屋面积（平方米）预测房价（万元）
假设真实规律：房价 = 5 × 面积 + 30 + 噪声
我们要从数据中"发现"这个规律（不知道5和30！）
"""

# 生成真实数据
n_samples = 100
X = np.random.uniform(20, 150, n_samples)  # 面积：20-150平方米
true_w = 5.0   # 每平方米5万
true_b = 30.0  # 基础价格30万
noise = np.random.randn(n_samples) * 20    # 现实中有各种影响因素
y = true_w * X + true_b + noise            # 真实房价

print(f"数据：{n_samples}套房屋")
print(f"面积范围：{X.min():.0f} - {X.max():.0f} 平方米")
print(f"房价范围：{y.min():.0f} - {y.max():.0f} 万元")
print(f"真实参数（我们不知道！）：w={true_w}, b={true_b}")

# ============================================================
# Part 2: 损失函数 —— 衡量预测有多差
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 均方误差（MSE）损失")
print("=" * 50)

"""
损失函数 = 衡量模型预测有多差的"分数"（越小越好）

均方误差 MSE：
  L(w, b) = (1/N) * Σ (y_pred_i - y_true_i)²

为什么用平方？
  1. 正负误差都变成正数（不会互相抵消）
  2. 大误差被惩罚更多（平方放大了大误差）
  3. 数学上好求导（导数很干净）

为什么不用绝对值 |误差|？
  绝对值在0处不可导，梯度下降不够平滑
"""

def predict(X, w, b):
    return w * X + b

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# 测试几组不同的参数
print("不同参数的 MSE 损失：")
test_params = [(5, 30), (3, 50), (7, 10), (1, 100)]
for w, b in test_params:
    y_pred = predict(X, w, b)
    loss = mse_loss(y_pred, y)
    print(f"  w={w:3d}, b={b:3d} → MSE={loss:8.2f}", end="")
    if w == 5 and b == 30:
        print("  ← 真实参数，损失最小！")
    else:
        print()

# ============================================================
# Part 3: 梯度推导 —— 往哪个方向更新参数？
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 梯度推导")
print("=" * 50)

"""
L(w, b) = (1/N) * Σ (w*x_i + b - y_i)²

对 w 求偏导：
  ∂L/∂w = (2/N) * Σ (w*x_i + b - y_i) * x_i
         = (2/N) * Σ (y_pred_i - y_true_i) * x_i

对 b 求偏导：
  ∂L/∂b = (2/N) * Σ (w*x_i + b - y_i) * 1
         = (2/N) * Σ (y_pred_i - y_true_i)

直觉：
  - 如果 y_pred > y_true（预测偏高），误差 > 0
    - ∂L/∂w > 0（if x>0）→ w 要减小，让预测降低
  - 这正是梯度下降在做的事！
"""

def compute_gradients(X, y_true, w, b):
    """
    计算 MSE 损失对 w 和 b 的梯度
    """
    N = len(X)
    y_pred = predict(X, w, b)
    error = y_pred - y_true  # 预测误差

    # 链式法则：dL/dw = (dL/d_pred) * (d_pred/dw)
    # dL/d_pred = 2/N * error（MSE对预测值的导数）
    # d_pred/dw = x（预测值对w的导数）
    dw = (2 / N) * np.dot(error, X)  # = (2/N) * Σ error_i * x_i
    db = (2 / N) * np.sum(error)     # = (2/N) * Σ error_i

    return dw, db

# ============================================================
# Part 4: 梯度下降训练
# ============================================================
print("Part 4: 梯度下降训练循环")
print("=" * 50)

class LinearRegression:
    def __init__(self, learning_rate=0.0001, n_epochs=1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = 0.0  # 初始化为0
        self.b = 0.0
        self.loss_history = []

    def fit(self, X, y):
        for epoch in range(self.n_epochs):
            # 1. 前向传播：计算预测和损失
            y_pred = predict(X, self.w, self.b)
            loss = mse_loss(y_pred, y)
            self.loss_history.append(loss)

            # 2. 反向传播：计算梯度
            dw, db = compute_gradients(X, y, self.w, self.b)

            # 3. 更新参数（沿负梯度方向）
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d}: Loss={loss:.2f}, w={self.w:.3f}, b={self.b:.3f}")

    def predict(self, X):
        return predict(X, self.w, self.b)

model = LinearRegression(learning_rate=0.0001, n_epochs=1000)
model.fit(X, y)

print(f"\n训练结果：")
print(f"  学到的 w = {model.w:.3f}  （真实值：{true_w}）")
print(f"  学到的 b = {model.b:.3f}  （真实值：{true_b}）")

# ============================================================
# Part 5: 解析解 vs 梯度下降
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 解析解 —— 一步到位的数学解")
print("=" * 50)

"""
线性回归有精确的解析解（最小二乘法）：
  W* = (X^T X)^{-1} X^T y

为什么实际不用解析解？
  1. 矩阵求逆的计算量是 O(n³)，n是参数数量
  2. 神经网络有百万参数，矩阵逆根本算不了
  3. 梯度下降可以在线学习（来一个样本更新一次）
  4. 解析解需要完整数据集，不适合大数据

但对于线性回归，解析解是验证梯度下降的好工具！
"""

# 构建矩阵形式（加偏置列）
X_matrix = np.column_stack([X, np.ones(len(X))])  # shape (N, 2)
# W* = (X^T X)^{-1} X^T y
W_analytical = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y
print(f"解析解：w={W_analytical[0]:.4f}, b={W_analytical[1]:.4f}")
print(f"梯度下降：w={model.w:.4f}, b={model.b:.4f}")
print(f"结果非常接近！（梯度下降需要调学习率和epoch数）")

# ============================================================
# Part 6: 可视化结果
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：数据和拟合直线
ax = axes[0]
ax.scatter(X, y, alpha=0.5, label='真实数据', color='steelblue', s=30)
X_line = np.linspace(20, 150, 100)
ax.plot(X_line, predict(X_line, true_w, true_b),
       'g--', linewidth=2, label=f'真实直线 (w={true_w},b={true_b})')
ax.plot(X_line, predict(X_line, model.w, model.b),
       'r-', linewidth=2, label=f'学到的直线 (w={model.w:.2f},b={model.b:.2f})')
ax.set_xlabel('面积（平方米）')
ax.set_ylabel('房价（万元）')
ax.set_title('线性回归：拟合结果')
ax.legend()
ax.grid(True, alpha=0.3)

# 右图：训练损失曲线
ax = axes[1]
ax.plot(model.loss_history)
ax.set_xlabel('训练步数（Epoch）')
ax.set_ylabel('MSE 损失')
ax.set_title('训练过程：损失下降曲线')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('01_classical_ml/linear_regression.png', dpi=100, bbox_inches='tight')
print("\n图片已保存：01_classical_ml/linear_regression.png")
plt.show()

# ============================================================
# Part 7: 正则化 —— 防止过拟合
# ============================================================
print("\n" + "=" * 50)
print("Part 7: 正则化 —— 防止模型太"复杂"")
print("=" * 50)

"""
过拟合：模型把训练数据的"噪声"也记住了，在新数据上表现差。

解决方法：正则化 = 在损失函数里加一个"惩罚项"
  L2 正则化（Ridge）：L_total = MSE + λ * Σ w²
    - 惩罚过大的权重（让权重不要太极端）
    - 梯度：dL/dw += 2λ * w

  L1 正则化（Lasso）：L_total = MSE + λ * Σ |w|
    - 倾向于把小权重推到0（特征选择！）

λ（正则化强度）是超参数：
  太大：模型太简单，欠拟合
  太小：相当于没正则化
"""

# 多项式过拟合示例（只用5个点，但拟合高次多项式）
n_few = 10
X_few = np.sort(np.random.uniform(0, 1, n_few))
y_few = np.sin(2 * np.pi * X_few) + np.random.randn(n_few) * 0.2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('正则化：控制模型复杂度', fontsize=13)

X_plot = np.linspace(0, 1, 200)
degree = 9  # 高阶多项式（容易过拟合）

for ax, (lam, title) in zip(axes, [
    (0, '无正则化（过拟合）'),
    (0.001, 'L2正则化 λ=0.001'),
    (1.0, 'L2正则化 λ=1.0（欠拟合）'),
]):
    # 构造多项式特征
    X_poly = np.column_stack([X_few**i for i in range(degree+1)])
    X_plot_poly = np.column_stack([X_plot**i for i in range(degree+1)])

    # L2正则化的解析解：W = (X^T X + λI)^{-1} X^T y
    n_features = degree + 1
    W = np.linalg.inv(X_poly.T @ X_poly + lam * np.eye(n_features)) @ X_poly.T @ y_few

    y_fit = X_plot_poly @ W

    ax.scatter(X_few, y_few, color='red', s=50, zorder=5, label='训练数据')
    ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', label='真实函数')
    ax.plot(X_plot, np.clip(y_fit, -3, 3), 'b-', linewidth=2, label='模型预测')
    ax.set_ylim(-2, 2)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_classical_ml/regularization.png', dpi=100, bbox_inches='tight')
print("图片已保存：01_classical_ml/regularization.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【学习率实验】
   把 learning_rate 改成 0.001、0.00001，重新训练。
   观察：收敛速度有什么变化？损失最终能到多低？

2. 【多变量线性回归】
   如果还有"房龄"这个特征（老房子便宜），
   模型变成：房价 = w1*面积 + w2*房龄 + b
   修改代码支持多个特征（提示：X 变成矩阵，w 变成向量）

3. 【正则化强度】
   在多项式过拟合例子中，尝试 λ = 0.0001, 0.01, 0.1, 10。
   画出每种 λ 下的拟合曲线，找到最合适的 λ。
   如何客观地评估哪个 λ 最好？（提示：用"验证集"）
""")
