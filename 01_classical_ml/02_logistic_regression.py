"""
==============================================================
第1章 第2节：逻辑回归 —— 把分数变成概率
==============================================================

【为什么需要它？】
线性回归输出的是连续值（-∞ 到 +∞），
但分类问题需要输出"概率"（0 到 1）或类别（是/否）。
直接用线性回归做分类有什么问题？
  - 预测值可能是 -5 或 10，不是合法的概率
  - 阈值 0.5 的设定很随意

逻辑回归的解决方案：用 Sigmoid 函数把线性输出"压缩"到 [0,1]

【生活类比】
面试官给候选人打分（0-100分），然后决定是否录用。
逻辑回归 = 先算一个"综合分"（线性），
再用 Sigmoid 把分数变成"录用概率"（0~1）。
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: Sigmoid 函数 —— 把任意值压到 [0,1]
# ============================================================
print("=" * 50)
print("Part 1: Sigmoid 函数")
print("=" * 50)

"""
Sigmoid(z) = 1 / (1 + e^{-z})

特性：
  z → +∞ 时，Sigmoid → 1
  z = 0 时，Sigmoid = 0.5
  z → -∞ 时，Sigmoid → 0

所以：分数越高 → 概率越接近1（录用）；分数越低 → 概率越接近0（不录用）

导数（很重要！后面要用）：
  d Sigmoid(z) / dz = Sigmoid(z) * (1 - Sigmoid(z))

这个导数最大值是 0.25（z=0 时），这就是"梯度消失问题"的来源。
"""

def sigmoid(z):
    """数值稳定版 Sigmoid"""
    # 避免 exp(-z) 溢出：对负数用不同公式
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

z = np.linspace(-8, 8, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(z, sigmoid(z), 'b-', linewidth=2.5, label='Sigmoid(z)')
ax.plot(z, sigmoid(z) * (1 - sigmoid(z)), 'r--', linewidth=2, label="Sigmoid'(z)（导数）")
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('z（线性分数）')
ax.set_ylabel('值')
ax.set_title('Sigmoid 函数及其导数')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(3, 0.55, 'z大 → 概率接近1', fontsize=9, color='blue')
ax.text(-7, 0.1, 'z小 → 概率接近0', fontsize=9, color='blue')

# ============================================================
# Part 2: 数据生成 —— 二分类问题
# ============================================================
print("Part 2: 生成二分类数据集")
print("=" * 50)

"""
场景：根据学习时长和睡眠时长，预测考试是否通过（1=通过，0=不通过）
"""

n = 200
# 通过的学生：学习时间长，睡眠充足
X_pass = np.column_stack([
    np.random.normal(7, 1.5, n//2),   # 学习时长（小时）
    np.random.normal(7, 1, n//2)       # 睡眠时长（小时）
])
y_pass = np.ones(n//2)

# 不通过的学生：学习时间短，睡眠少或乱
X_fail = np.column_stack([
    np.random.normal(3, 1.5, n//2),
    np.random.normal(5, 1.5, n//2)
])
y_fail = np.zeros(n//2)

X = np.vstack([X_pass, X_fail])
y = np.concatenate([y_pass, y_fail])

# 打乱
idx = np.random.permutation(n)
X, y = X[idx], y[idx]

# 特征标准化（让梯度下降更稳定）
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

print(f"数据集：{n}个学生，{(y==1).sum()}人通过，{(y==0).sum()}人不通过")

# 可视化数据
ax = axes[1]
ax.scatter(X[y==1, 0], X[y==1, 1], c='green', s=50, alpha=0.6, label='通过(1)')
ax.scatter(X[y==0, 0], X[y==0, 1], c='red', s=50, alpha=0.6, label='不通过(0)')
ax.set_xlabel('学习时长（小时）')
ax.set_ylabel('睡眠时长（小时）')
ax.set_title('二分类问题：考试预测')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_classical_ml/logistic_data.png', dpi=100, bbox_inches='tight')
plt.show()

# ============================================================
# Part 3: 损失函数 —— 为什么用交叉熵而不是MSE？
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 交叉熵损失 vs MSE 损失")
print("=" * 50)

"""
对于分类问题，用 MSE 的问题：
  - 预测概率 p=0.01，真实标签 y=1
  - MSE = (0.01 - 1)² = 0.9801
  - 梯度 = 2*(p-y)*p*(1-p) ≈ 2*(-0.99)*0.01*0.99 ≈ -0.02
  - 梯度非常小！即使预测严重错误，也学不快

用交叉熵：
  BCE = -[y*log(p) + (1-y)*log(1-p)]
  - 预测 p=0.01，真实 y=1：BCE = -log(0.01) ≈ 4.6  （损失很大！）
  - 梯度 = p - y = 0.01 - 1 = -0.99  （梯度很大，学习快！）

交叉熵 + Sigmoid 的组合非常优雅：
  梯度 dL/dz = sigmoid(z) - y = p - y
  当预测错且自信时，梯度大；预测对时，梯度小。完美！
"""

def bce_loss(y_pred_prob, y_true):
    """二元交叉熵损失"""
    eps = 1e-8
    return -np.mean(
        y_true * np.log(y_pred_prob + eps) +
        (1 - y_true) * np.log(1 - y_pred_prob + eps)
    )

# 对比 BCE 和 MSE 在不同预测下的梯度大小
print("预测值p，真实y=1时，两种损失的梯度对比：")
print(f"{'预测p':>8} {'BCE梯度':>12} {'MSE梯度':>12} {'谁学习更快？':>16}")
print("-" * 52)
for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
    # BCE 梯度（对z）：p - y = p - 1
    bce_grad = p - 1.0
    # MSE 梯度（对z）：2*(p-y)*p*(1-p)
    mse_grad = 2 * (p - 1.0) * p * (1 - p)
    ratio = abs(bce_grad) / (abs(mse_grad) + 1e-8)
    print(f"{p:>8.2f} {bce_grad:>12.4f} {mse_grad:>12.4f} {'BCE快' if ratio > 1 else 'MSE快':>16}（{ratio:.1f}x）")

# ============================================================
# Part 4: 手写逻辑回归
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 手写逻辑回归模型")
print("=" * 50)

class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_epochs=1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            # ===== 前向传播 =====
            z = X @ self.W + self.b    # 线性部分
            p = sigmoid(z)             # 概率预测

            # ===== 计算损失 =====
            loss = bce_loss(p, y)
            self.loss_history.append(loss)

            # ===== 反向传播 =====
            # 交叉熵 + Sigmoid 的组合梯度：dp - y
            # 然后链式传回线性层
            error = p - y              # 预测概率 - 真实标签

            dW = (1 / n_samples) * X.T @ error   # 梯度 w.r.t. W
            db = (1 / n_samples) * np.sum(error)  # 梯度 w.r.t. b

            # ===== 参数更新 =====
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 200 == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch:4d}: Loss={loss:.4f}, Acc={acc:.2%}")

    def predict_proba(self, X):
        """返回概率"""
        z = X @ self.W + self.b
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        """返回类别"""
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# 训练
model = LogisticRegression(learning_rate=0.1, n_epochs=1000)
model.fit(X_norm, y)

print(f"\n最终准确率：{model.accuracy(X_norm, y):.2%}")

# ============================================================
# Part 5: 可视化决策边界
# ============================================================
def plot_decision_boundary(model, X, y, X_mean, X_std):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：决策边界
    ax = axes[0]
    h = 0.05
    x1_min, x1_max = X[:, 0].min()-1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))

    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    X_grid_norm = (X_grid - X_mean) / X_std
    Z = model.predict_proba(X_grid_norm).reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, levels=50, cmap='RdYlGn', alpha=0.4)
    ax.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)

    ax.scatter(X[y==1, 0], X[y==1, 1], c='green', s=50, alpha=0.7, label='通过(1)', zorder=5)
    ax.scatter(X[y==0, 0], X[y==0, 1], c='red', s=50, alpha=0.7, label='不通过(0)', zorder=5)
    ax.set_xlabel('学习时长')
    ax.set_ylabel('睡眠时长')
    ax.set_title('决策边界（黑线=0.5概率分界）')
    ax.legend()

    # 右图：训练曲线
    ax = axes[1]
    ax.plot(model.loss_history, 'b-', linewidth=2)
    ax.set_xlabel('训练步数')
    ax.set_ylabel('交叉熵损失')
    ax.set_title('训练过程')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('01_classical_ml/logistic_boundary.png', dpi=100, bbox_inches='tight')
    print("图片已保存：01_classical_ml/logistic_boundary.png")
    plt.show()

plot_decision_boundary(model, X, y, X_mean, X_std)

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【Sigmoid 对称性】
   Sigmoid(0) = 0.5，说明当线性分数 z=0 时，模型完全不确定。
   如果想让决策阈值从 0.5 改为 0.7（更严格），
   不改代码的情况下，该如何调整 b（偏置）？

2. 【多分类扩展】
   逻辑回归只能做二分类。对于 3 分类问题：
   - Softmax = 把 K 个分数变成 K 个概率，和为1
   - Softmax(z_k) = e^{z_k} / Σ e^{z_j}
   手写 Softmax 函数，并验证输出之和确实为 1。

3. 【非线性边界】
   生成一个"圆形"数据集（内圈是一类，外圈是另一类），
   逻辑回归（线性决策边界）能分开吗？
   提示：可以加特征 x1², x2², x1*x2（多项式特征）
   这说明了什么？（神经网络通过非线性激活解决此问题）
""")
