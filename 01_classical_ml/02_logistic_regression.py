"""
====================================================================
第1章 · 第2节 · 逻辑回归与交叉熵
====================================================================

【一句话总结】
逻辑回归 = 线性回归 + sigmoid + 交叉熵损失，是神经网络分类层的原型。

【为什么深度学习需要这个？】
- 神经网络最后一层（分类）就是逻辑回归
- sigmoid 是最经典的激活函数（虽然现在多用 ReLU/GELU）
- 交叉熵损失是分类任务的标准损失函数，从这里理解最自然
- 从回归到分类的跨越，引出概率建模的思想

【核心概念】

1. 从回归到分类
   - 回归：预测连续值（房价）
   - 分类：预测类别（猫/狗）
   - 关键转变：输出不再是任意数值，而是概率

2. Sigmoid 函数
   - σ(z) = 1 / (1 + e^(-z))，将任意实数压到 (0,1)
   - 导数：σ'(z) = σ(z)(1 - σ(z))，形状优美但有梯度消失问题
   - 阈值判断：> 0.5 → 类别1，< 0.5 → 类别0

3. 为什么分类不能用 MSE？
   - MSE + sigmoid → 损失函数非凸，容易困在局部最优
   - 交叉熵 + sigmoid → 损失函数凸，有唯一全局最优
   - 梯度形式更好：∂L/∂z = (σ(z) - y)，简洁高效

4. 交叉熵损失（Binary Cross-Entropy）
   - L = -[y·log(p) + (1-y)·log(1-p)]
   - 当 y=1 时，p 越大损失越小（奖励正确的高置信预测）
   - 当 y=0 时，p 越小损失越小
   - 信息论视角：用预测分布编码真实分布的代价

5. 决策边界
   - wx + b = 0 就是决策边界（2D 是一条线，3D 是一个面）
   - 线性模型只能画直线边界 → 非线性问题需要更复杂的模型

【前置知识】
第1章第1节 - 线性回归，第0章第3节 - 概率与信息论
"""

import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子，保证每次运行结果一致
np.random.seed(42)


# ====================================================================
# 第一部分：生成二分类数据
# ====================================================================
# 生成两个高斯分布的点簇，模拟最简单的二分类场景
# 类别 0：以 (-1, -1) 为中心的簇
# 类别 1：以 (1, 1) 为中心的簇

def generate_binary_data(n_samples=200, noise=0.8):
    """
    生成二分类数据集

    参数:
        n_samples: 总样本数（两个类别各一半）
        noise: 高斯噪声标准差，越大两类越难分

    返回:
        X: 形状 (n_samples, 2) 的特征矩阵
        y: 形状 (n_samples,) 的标签向量（0 或 1）
    """
    n_half = n_samples // 2

    # 类别 0：中心在 (-1, -1)
    X0 = np.random.randn(n_half, 2) * noise + np.array([-1, -1])
    # 类别 1：中心在 (1, 1)
    X1 = np.random.randn(n_half, 2) * noise + np.array([1, 1])

    # 拼接成完整数据集
    X = np.vstack([X0, X1])               # (n_samples, 2)
    y = np.hstack([np.zeros(n_half),       # 前半是 0
                   np.ones(n_half)])        # 后半是 1

    # 打乱顺序，防止训练时出现"先全是0、后全是1"的偏差
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


# 生成数据并可视化
X, y = generate_binary_data(n_samples=200, noise=0.8)

plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='类别 0', alpha=0.6, edgecolors='k', s=40)
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='类别 1', alpha=0.6, edgecolors='k', s=40)
plt.xlabel('特征 x₁')
plt.ylabel('特征 x₂')
plt.title('二分类数据集：两个高斯分布簇')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_binary_data.png', dpi=100)
plt.close()
print("[图1] 二分类数据集已保存 → 01_binary_data.png")


# ====================================================================
# 第二部分：Sigmoid 函数详解
# ====================================================================
# sigmoid 把任意实数映射到 (0, 1)，让输出具有"概率"的含义
#
# 公式: σ(z) = 1 / (1 + exp(-z))
#
# 关键性质:
# - σ(0) = 0.5（中间值）
# - z → +∞ 时 σ(z) → 1（饱和区：梯度趋近于 0 → 梯度消失！）
# - z → -∞ 时 σ(z) → 0（饱和区：同上）
# - σ(-z) = 1 - σ(z)（关于 (0, 0.5) 中心对称）

def sigmoid(z):
    """Sigmoid 激活函数：σ(z) = 1 / (1 + e^(-z))"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Sigmoid 的导数：σ'(z) = σ(z) * (1 - σ(z))

    这个性质非常优美——导数可以直接用函数值表达。
    但也暗藏问题：当 σ(z) 接近 0 或 1 时，导数接近 0 → 梯度消失。
    """
    s = sigmoid(z)
    return s * (1 - s)


# 可视化 sigmoid 及其导数
z = np.linspace(-8, 8, 500)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：sigmoid 函数
ax = axes[0]
ax.plot(z, sigmoid(z), 'b-', linewidth=2, label='σ(z)')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='y = 0.5')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
# 标注饱和区
ax.axvspan(-8, -4, alpha=0.1, color='red', label='饱和区（梯度≈0）')
ax.axvspan(4, 8, alpha=0.1, color='red')
# 标注线性区
ax.axvspan(-2, 2, alpha=0.1, color='green', label='近似线性区')
ax.set_xlabel('z')
ax.set_ylabel('σ(z)')
ax.set_title('Sigmoid 函数')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, alpha=0.3)

# 右图：sigmoid 导数
ax = axes[1]
ax.plot(z, sigmoid_derivative(z), 'r-', linewidth=2, label="σ'(z) = σ(z)(1-σ(z))")
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
# 导数最大值在 z=0 处，值为 0.25
ax.annotate('最大值 0.25', xy=(0, 0.25), xytext=(2, 0.22),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=11, color='black')
ax.set_xlabel('z')
ax.set_ylabel("σ'(z)")
ax.set_title('Sigmoid 导数 → 梯度消失的根源')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_sigmoid_detail.png', dpi=100)
plt.close()
print("[图2] Sigmoid 函数与导数已保存 → 02_sigmoid_detail.png")

# 数值验证：导数最大值只有 0.25
# 如果网络有 10 层，每层梯度乘以 0.25，最终梯度 = 0.25^10 ≈ 0.000001
# 这就是深层网络用 sigmoid 容易梯度消失的原因！
print(f"\n[数值] sigmoid 导数最大值 = {sigmoid_derivative(0):.4f}")
print(f"[数值] 10 层连乘: 0.25^10 = {0.25**10:.10f} → 梯度消失！")


# ====================================================================
# 第三部分：MSE vs 交叉熵 —— 为什么分类不能用 MSE？
# ====================================================================
# 核心结论：MSE + sigmoid 产生非凸损失面，梯度不好；
#           交叉熵 + sigmoid 产生凸损失面，梯度优雅。
#
# 数学推导:
#   MSE 对 z 的梯度: ∂L_mse/∂z = (σ(z) - y) · σ'(z)
#                    → 含有 σ'(z) 项，饱和时梯度消失
#
#   交叉熵对 z 的梯度: ∂L_ce/∂z = σ(z) - y
#                      → 简洁！不含 σ' 项，不会饱和

def binary_cross_entropy(p, y, eps=1e-15):
    """
    二元交叉熵损失

    参数:
        p: 预测概率（经过 sigmoid）
        y: 真实标签（0 或 1）
        eps: 防止 log(0) 的极小数

    公式: L = -[y·log(p) + (1-y)·log(1-p)]
    """
    p = np.clip(p, eps, 1 - eps)  # 数值稳定：避免 log(0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mse_loss(p, y):
    """均方误差损失: L = (p - y)²"""
    return (p - y) ** 2


# 可视化：固定 y=1，画出 L 关于 z 的函数图
z_range = np.linspace(-6, 6, 500)
p_range = sigmoid(z_range)
y_true = 1.0  # 真实标签 = 1

loss_ce = binary_cross_entropy(p_range, y_true)
loss_mse = mse_loss(p_range, y_true)

# 计算两种损失关于 z 的梯度
grad_ce = p_range - y_true                                  # σ(z) - y，简洁！
grad_mse = 2 * (p_range - y_true) * sigmoid_derivative(z_range)  # 多了 σ'(z) 项

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失函数对比
ax = axes[0]
ax.plot(z_range, loss_ce, 'b-', linewidth=2, label='交叉熵 (凸！)')
ax.plot(z_range, loss_mse, 'r--', linewidth=2, label='MSE (非凸)')
ax.set_xlabel('z（线性输出 = wx + b）')
ax.set_ylabel('损失值 L')
ax.set_title('损失函数对比（真实标签 y=1）')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# 右图：梯度对比
ax = axes[1]
ax.plot(z_range, np.abs(grad_ce), 'b-', linewidth=2, label='|∂L_ce/∂z| = |σ(z)-y|')
ax.plot(z_range, np.abs(grad_mse), 'r--', linewidth=2, label='|∂L_mse/∂z| ← 含 σ\'(z)')
ax.set_xlabel('z（线性输出 = wx + b）')
ax.set_ylabel('|梯度|')
ax.set_title('梯度对比 → MSE 在两端梯度消失')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_mse_vs_crossentropy.png', dpi=100)
plt.close()
print("\n[图3] MSE vs 交叉熵对比已保存 → 03_mse_vs_crossentropy.png")

# 关键观察:
# - 交叉熵在 z → -∞（预测完全错误）时梯度很大 → 犯大错时更新快
# - MSE 在 z → -∞ 时梯度反而很小 → 犯大错时更新不动（！）
print("\n关键观察:")
print(f"  z = -5 (预测完全错误) 时:")
print(f"    交叉熵梯度 |σ(z)-y| = {abs(sigmoid(-5) - 1):.4f} ← 很大，更新快")
print(f"    MSE 梯度             = {abs(2 * (sigmoid(-5) - 1) * sigmoid_derivative(-5)):.6f} ← 很小，更新不动！")


# ====================================================================
# 第四部分：手写逻辑回归（完整实现）
# ====================================================================
# 模型: p = σ(Xw + b)
# 损失: L = -1/N Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]
# 梯度:
#   ∂L/∂w = 1/N · Xᵀ(p - y)    ← 线性回归梯度中 (Xw+b-y) 替换成 (p-y)
#   ∂L/∂b = 1/N · Σ(p - y)

class LogisticRegression:
    """
    手写逻辑回归分类器

    与线性回归的三个关键区别:
    1. 输出层加了 sigmoid（将输出压到 0-1）
    2. 损失函数从 MSE 换成交叉熵
    3. 梯度中的误差项从 (z-y) 变成 (σ(z)-y)
    """

    def __init__(self, n_features):
        """
        初始化参数

        参数:
            n_features: 输入特征维度
        """
        # 小随机数初始化（不用零初始化，虽然逻辑回归零初始化也能工作）
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

        # 记录训练历史，用于可视化
        self.loss_history = []
        self.acc_history = []

    def predict_proba(self, X):
        """
        预测概率: p = σ(Xw + b)

        参数:
            X: 形状 (N, D) 的输入

        返回:
            形状 (N,) 的概率值，每个元素在 (0, 1) 之间
        """
        z = X @ self.w + self.b   # 线性部分：(N, D) @ (D,) + () → (N,)
        return sigmoid(z)          # 非线性部分：把实数压到 (0, 1)

    def predict(self, X, threshold=0.5):
        """
        预测类别: p > 阈值 → 类别 1，否则 → 类别 0

        阈值默认 0.5，但在某些场景（如医学诊断）可能需要调整
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_loss(self, X, y):
        """
        计算平均交叉熵损失

        公式: L = -1/N Σ [yᵢ·log(pᵢ) + (1-yᵢ)·log(1-pᵢ)]
        """
        p = self.predict_proba(X)
        eps = 1e-15  # 数值稳定
        p = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss

    def compute_gradients(self, X, y):
        """
        计算梯度（向量化版本）

        推导过程:
          z = Xw + b
          p = σ(z)
          L = -1/N Σ [y·log(p) + (1-y)·log(1-p)]

          令 error = p - y（预测概率 - 真实标签），则:
            ∂L/∂w = 1/N · Xᵀ · error     (D,) = (D, N) @ (N,)
            ∂L/∂b = 1/N · Σ error         标量

        注意：这个梯度形式和线性回归几乎一样！
        唯一区别是 error = σ(Xw+b) - y 而不是 (Xw+b) - y
        """
        N = len(y)
        p = self.predict_proba(X)
        error = p - y                        # (N,)

        dw = (1 / N) * (X.T @ error)         # (D,)
        db = (1 / N) * np.sum(error)          # 标量

        return dw, db

    def fit(self, X, y, lr=0.1, n_iters=200, verbose=True):
        """
        梯度下降训练

        参数:
            X: 训练数据 (N, D)
            y: 标签 (N,)
            lr: 学习率
            n_iters: 迭代次数
            verbose: 是否打印训练过程
        """
        self.loss_history = []
        self.acc_history = []

        for i in range(n_iters):
            # 1. 前向传播 + 计算损失
            loss = self.compute_loss(X, y)

            # 2. 计算精度
            preds = self.predict(X)
            acc = np.mean(preds == y)

            # 3. 记录历史
            self.loss_history.append(loss)
            self.acc_history.append(acc)

            # 4. 反向传播：计算梯度
            dw, db = self.compute_gradients(X, y)

            # 5. 参数更新
            self.w -= lr * dw
            self.b -= lr * db

            # 6. 打印训练进度
            if verbose and (i % 50 == 0 or i == n_iters - 1):
                print(f"  迭代 {i:>4d}/{n_iters} | 损失 = {loss:.4f} | 精度 = {acc:.2%}")

        return self


# 训练逻辑回归模型
print("\n" + "=" * 60)
print("训练逻辑回归模型")
print("=" * 60)
model = LogisticRegression(n_features=2)
model.fit(X, y, lr=0.5, n_iters=300, verbose=True)


# ====================================================================
# 第五部分：决策边界可视化
# ====================================================================
# 决策边界 = 满足 w₁x₁ + w₂x₂ + b = 0 的点的集合
# 对于 2D 情况，这是一条直线: x₂ = -(w₁x₁ + b) / w₂

def plot_decision_boundary(model, X, y, title="决策边界"):
    """
    可视化逻辑回归的决策边界

    方法：在特征空间上画出密集的网格点，
    对每个网格点进行预测，然后用颜色填充表示分类结果。
    """
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

    # 对每个网格点预测概率
    probs = model.predict_proba(grid).reshape(xx.shape)

    # 画等高线（概率热力图）
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, probs, levels=50, cmap='RdBu_r', alpha=0.7)
    plt.colorbar(label='预测概率 P(y=1)')

    # 画决策边界（p=0.5 的等高线）
    plt.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)

    # 画数据点
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='类别 0',
                alpha=0.6, edgecolors='k', s=40)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='类别 1',
                alpha=0.6, edgecolors='k', s=40)

    plt.xlabel('特征 x₁')
    plt.ylabel('特征 x₂')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()


plot_decision_boundary(model, X, y, title='逻辑回归决策边界（直线）')
plt.savefig('04_decision_boundary.png', dpi=100)
plt.close()
print("\n[图4] 决策边界已保存 → 04_decision_boundary.png")

# 打印学到的参数
print(f"\n学到的参数:")
print(f"  w = [{model.w[0]:.4f}, {model.w[1]:.4f}]")
print(f"  b = {model.b:.4f}")
print(f"  决策边界方程: {model.w[0]:.2f}·x₁ + {model.w[1]:.2f}·x₂ + {model.b:.2f} = 0")


# ====================================================================
# 第六部分：训练过程可视化（损失曲线 + 精度曲线）
# ====================================================================
# 观察要点:
# - 损失应该单调下降（凸优化保证）
# - 精度应该逐步上升
# - 如果损失震荡 → 学习率太大；如果下降太慢 → 学习率太小

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：损失曲线
ax = axes[0]
ax.plot(model.loss_history, 'b-', linewidth=1.5)
ax.set_xlabel('迭代次数')
ax.set_ylabel('交叉熵损失')
ax.set_title('训练损失曲线（应该单调下降）')
ax.grid(True, alpha=0.3)

# 右图：精度曲线
ax = axes[1]
ax.plot(model.acc_history, 'r-', linewidth=1.5)
ax.set_xlabel('迭代次数')
ax.set_ylabel('训练精度')
ax.set_title('训练精度曲线')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_training_curves.png', dpi=100)
plt.close()
print("\n[图5] 训练曲线已保存 → 05_training_curves.png")

# 学习率实验：对比不同学习率的收敛速度
print("\n" + "=" * 60)
print("学习率实验")
print("=" * 60)

learning_rates = [0.01, 0.1, 1.0, 5.0]
plt.figure(figsize=(10, 5))

for lr in learning_rates:
    m = LogisticRegression(n_features=2)
    m.fit(X, y, lr=lr, n_iters=200, verbose=False)
    plt.plot(m.loss_history, linewidth=1.5, label=f'lr = {lr}')

plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.title('不同学习率的收敛速度对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06_learning_rate_comparison.png', dpi=100)
plt.close()
print("[图6] 学习率对比已保存 → 06_learning_rate_comparison.png")


# ====================================================================
# 第七部分：多分类扩展 —— Softmax + 分类交叉熵
# ====================================================================
# 逻辑回归只能做二分类。当类别 > 2 时，需要推广：
#
# sigmoid → softmax
# 二元交叉熵 → 分类交叉熵（Categorical Cross-Entropy）
#
# Softmax 公式:
#   softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)
#
# 性质:
# - 每个输出都在 (0, 1) 之间
# - 所有输出之和 = 1（形成合法的概率分布）
# - 当只有两个类时，softmax 退化为 sigmoid

print("\n" + "=" * 60)
print("多分类扩展：Softmax + 分类交叉熵")
print("=" * 60)


def softmax(z):
    """
    Softmax 函数：将 K 维实数向量转换为概率分布

    数值稳定技巧: 减去最大值防止 exp 溢出
    exp(z - max) / Σ exp(z - max) = exp(z) / Σ exp(z)（数学上等价）

    参数:
        z: 形状 (N, K) 的 logits 矩阵，N 个样本，K 个类别

    返回:
        形状 (N, K) 的概率矩阵，每行之和为 1
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # 数值稳定
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def categorical_cross_entropy(probs, y_onehot, eps=1e-15):
    """
    分类交叉熵损失

    参数:
        probs: softmax 输出 (N, K)
        y_onehot: one-hot 编码的标签 (N, K)

    公式: L = -1/N Σᵢ Σₖ y_ik · log(p_ik)
    """
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_onehot * np.log(probs), axis=1))


# 演示 softmax 的行为
print("\nSoftmax 演示:")
demo_logits = np.array([[2.0, 1.0, 0.1]])
demo_probs = softmax(demo_logits)
print(f"  输入 logits: {demo_logits[0]}")
print(f"  Softmax 输出: [{demo_probs[0, 0]:.4f}, {demo_probs[0, 1]:.4f}, {demo_probs[0, 2]:.4f}]")
print(f"  概率之和: {demo_probs.sum():.4f}")

# 演示 softmax 温度效应（与后面 GPT 生成时的 temperature 概念相同！）
print("\n温度效应（预告：GPT 文本生成时也会用到 temperature 参数）:")
temperatures = [0.5, 1.0, 2.0, 5.0]
for T in temperatures:
    scaled_probs = softmax(demo_logits / T)
    print(f"  T={T:.1f} → [{scaled_probs[0, 0]:.4f}, {scaled_probs[0, 1]:.4f}, {scaled_probs[0, 2]:.4f}]"
          f"  {'← 更尖锐（更确定）' if T < 1 else '← 更平坦（更随机）' if T > 1 else '← 标准'}")

# 可视化 softmax 温度效应
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
categories = ['猫', '狗', '鸟']
for idx, T in enumerate(temperatures):
    scaled_probs = softmax(demo_logits / T)[0]
    axes[idx].bar(categories, scaled_probs, color=['#2196F3', '#FF5722', '#4CAF50'])
    axes[idx].set_title(f'Temperature = {T}')
    axes[idx].set_ylim(0, 1)
    axes[idx].set_ylabel('概率')
    for j, p in enumerate(scaled_probs):
        axes[idx].text(j, p + 0.02, f'{p:.2f}', ha='center', fontsize=10)
plt.suptitle('Softmax 温度效应：T 越小越确定，T 越大越随机', fontsize=13)
plt.tight_layout()
plt.savefig('07_softmax_temperature.png', dpi=100)
plt.close()
print("\n[图7] Softmax 温度效应已保存 → 07_softmax_temperature.png")

# Softmax 与 Sigmoid 的关系
# 当 K=2 时，softmax 退化为 sigmoid！证明：
#   softmax([z, 0]) = [exp(z)/(exp(z)+1), 1/(exp(z)+1)]
#                    = [sigmoid(z), 1-sigmoid(z)]
print("\n验证: Softmax(K=2) = Sigmoid")
z_test = 1.5
softmax_result = softmax(np.array([[z_test, 0.0]]))[0, 0]
sigmoid_result = sigmoid(z_test)
print(f"  softmax([{z_test}, 0])[0] = {softmax_result:.6f}")
print(f"  sigmoid({z_test})         = {sigmoid_result:.6f}")
print(f"  差异: {abs(softmax_result - sigmoid_result):.2e} → 数学上完全等价！")


# ====================================================================
# 第八部分：逻辑回归的局限性 —— 线性不可分问题
# ====================================================================
# 逻辑回归只能画直线决策边界，遇到非线性数据就无能为力了
# 这正是我们需要多层神经网络（MLP）的原因！

print("\n" + "=" * 60)
print("逻辑回归的局限：线性不可分问题（XOR）")
print("=" * 60)

# 生成一个环形数据（线性不可分）
def generate_circle_data(n_samples=200, noise=0.1):
    """
    生成环形二分类数据（线性不可分！）
    内圈为类别 0，外圈为类别 1
    """
    n_half = n_samples // 2
    # 内圈
    theta_inner = np.random.uniform(0, 2 * np.pi, n_half)
    r_inner = np.random.normal(0.5, noise, n_half)
    X_inner = np.column_stack([r_inner * np.cos(theta_inner),
                                r_inner * np.sin(theta_inner)])
    # 外圈
    theta_outer = np.random.uniform(0, 2 * np.pi, n_half)
    r_outer = np.random.normal(1.5, noise, n_half)
    X_outer = np.column_stack([r_outer * np.cos(theta_outer),
                                r_outer * np.sin(theta_outer)])

    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n_half), np.ones(n_half)])
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx]


X_circle, y_circle = generate_circle_data(n_samples=300, noise=0.15)

# 用逻辑回归尝试分类（注定失败）
model_circle = LogisticRegression(n_features=2)
model_circle.fit(X_circle, y_circle, lr=0.5, n_iters=300, verbose=False)

final_acc = np.mean(model_circle.predict(X_circle) == y_circle)
print(f"逻辑回归在环形数据上的精度: {final_acc:.2%} ← 接近随机猜测！")

# 可视化失败的决策边界
plot_decision_boundary(model_circle, X_circle, y_circle,
                       title=f'逻辑回归的局限：线性不可分数据（精度 {final_acc:.0%}）')
plt.savefig('08_linear_limitation.png', dpi=100)
plt.close()
print("[图8] 线性不可分示例已保存 → 08_linear_limitation.png")
print("\n启示：要解决非线性问题，需要多层神经网络 → 第2章！")


# ====================================================================
# 第九部分：思考题
# ====================================================================
print("\n" + "=" * 60)
print("思考题")
print("=" * 60)

questions = """
1. [基础] 当 sigmoid 的输入 z = 10 时，σ(z) 和 σ'(z) 分别约等于多少？
   如果网络有 5 层都用 sigmoid，梯度会衰减到什么量级？
   这对深度网络训练意味着什么？

2. [理解] 为什么交叉熵对 z 的梯度是 σ(z) - y，不含 σ'(z)？
   请从 L = -y·log(σ(z)) - (1-y)·log(1-σ(z)) 出发，
   利用 σ'(z) = σ(z)(1-σ(z)) 推导这个结论。
   提示：展开链式法则后会有巧妙的约分。

3. [应用] 如果把逻辑回归的阈值从 0.5 改为 0.3（即 p > 0.3 就判为
   阳性），对假阳性率和假阴性率分别有什么影响？
   在医学诊断中，你会倾向于调高还是调低阈值？为什么？

4. [扩展] 尝试给本节的逻辑回归加上 L2 正则化
   (损失 += λ/2 · ||w||²)，观察决策边界和权重的变化。
   提示：只需在梯度上加一项 λ·w，不改变 b 的梯度。

5. [连接] 把三分类 softmax 的输出 [0.7, 0.2, 0.1] 看作三个二分类
   sigmoid 的输出，它们之和还等于 1 吗？为什么 sigmoid 不能
   直接用于多分类？softmax 解决了什么关键问题？
"""
print(questions)


# ====================================================================
# 总结：从逻辑回归到神经网络的桥梁
# ====================================================================
print("=" * 60)
print("本节总结")
print("=" * 60)
print("""
逻辑回归是连接线性模型和神经网络的桥梁:

  线性回归           逻辑回归             神经网络
  z = wx + b    →   p = σ(wx+b)    →   多层 + 各种激活函数
  MSE 损失           交叉熵损失          交叉熵损失（分类层不变！）
  连续值预测         概率预测             复杂特征学习 + 概率预测

下一节预告: 第1章第3节 - 正则化与过拟合
  为什么模型在训练集上表现好、测试集上却差？
  L1/L2 正则化如何对抗过拟合？
""")
