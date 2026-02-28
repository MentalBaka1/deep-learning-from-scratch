"""
==============================================================
第1章 第4节：SVM 直觉 —— 找最宽的走廊
==============================================================

【为什么需要它？】
分类问题可以有很多条"能把数据分开的线"，哪条最好？
SVM 的回答：找让两类数据"离得最远"的那条线。

【生活类比】
两支军队（红vs蓝）在平原对峙，划定一条界线。
普通分类：只要线能把双方隔开就行
SVM：找一条线，使得双方最近的战士到这条线的距离都最大
       → 建立最宽的"缓冲区"（间隔）
       → 这样如果来了新战士，也不容易被分错

支持向量 = 距离界线最近的那些战士（关键样本）
超平面 = 那条界线
间隔最大化 = SVM 的核心目标

【存在理由】
解决问题：如何在多条可行分类线中找到"最安全"的一条？
核心思想：最大化决策边界到最近样本的距离（间隔最大化）
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 间隔（Margin）的直觉
# ============================================================
print("=" * 50)
print("Part 1: 间隔的直觉 —— 为什么要最宽？")
print("=" * 50)

"""
设分类超平面方程：w·x + b = 0
  - 对于正类：w·x + b ≥ 1
  - 对于负类：w·x + b ≤ -1

两个支持超平面（w·x+b=1 和 w·x+b=-1）之间的距离 = 2/||w||

最大化间隔 ↔ 最大化 2/||w|| ↔ 最小化 ||w||²

这就是 SVM 的优化目标！（对偶问题用拉格朗日乘数法求解）
"""

# 生成线性可分数据
n_per_class = 30
X_pos = np.random.randn(n_per_class, 2) + np.array([2, 2])
X_neg = np.random.randn(n_per_class, 2) + np.array([-2, -2])
X = np.vstack([X_pos, X_neg])
y = np.array([1] * n_per_class + [-1] * n_per_class)

# 可视化不同分类线和它们的间隔
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('SVM：哪条分类线最好？', fontsize=13)

# 3条都能分开数据的线，但 SVM 找第3条（间隔最大的）
lines = [
    (1, 0, 0, '任意可行线（间隔小）', 0.3),
    (0.8, -0.3, 0, '另一条可行线（间隔中）', 0.8),
    (1, 0.2, 0.3, 'SVM最优线（间隔最大）', 1.5),
]

x_range = np.linspace(-5, 5, 100)
for ax, (w1, w2, b, title, margin_width) in zip(axes, lines):
    ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=60, zorder=5, label='正类(+1)')
    ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=60, zorder=5, label='负类(-1)')

    # 决策边界：w1*x1 + w2*x2 + b = 0  →  x2 = -(w1*x1+b)/w2
    if abs(w2) > 0.001:
        y_boundary = -(w1 * x_range + b) / w2
    else:
        y_boundary = np.zeros_like(x_range)

    ax.plot(x_range, y_boundary, 'k-', linewidth=2.5, label='决策边界')

    # 画出间隔区域（走廊）
    norm = np.sqrt(w1**2 + w2**2)
    direction = np.array([-w2, w1]) / norm  # 垂直方向
    ax.fill_between(x_range,
                   y_boundary - margin_width / norm,
                   y_boundary + margin_width / norm,
                   alpha=0.2, color='yellow', label=f'间隔区域 ≈{margin_width:.1f}')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('01_classical_ml/svm_margin.png', dpi=100, bbox_inches='tight')
print("图片已保存：01_classical_ml/svm_margin.png")
plt.show()

# ============================================================
# Part 2: 软间隔 SVM —— 允许一点点错误
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 软间隔 SVM —— 允许少量误分类")
print("=" * 50)

"""
现实中，数据不总是线性可分的。
硬间隔 SVM（Hard Margin）：要求所有样本都在正确一侧，但可能无解。
软间隔 SVM（Soft Margin）：允许一些样本在间隔内或越界，但要惩罚。

目标函数：
  min_{w,b} (1/2)||w||² + C * Σ max(0, 1 - y_i(w·x_i + b))

  第一项：最大化间隔（让 ||w||² 最小）
  第二项：Hinge Loss，惩罚违规样本
  C：权衡系数
    C 大 → 更在乎分类正确（可能过拟合）
    C 小 → 更在乎间隔宽（可能欠拟合）

Hinge Loss = max(0, 1 - y*z)
  - z = w·x+b（原始预测分数）
  - 如果 y*z ≥ 1（分类正确且在间隔外）：loss = 0
  - 如果 y*z < 1（接近边界或分类错误）：loss > 0
"""

def hinge_loss(z, y):
    """Hinge Loss：max(0, 1 - y*z)"""
    return np.maximum(0, 1 - y * z)

# 可视化 Hinge Loss
z = np.linspace(-3, 3, 200)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(z, hinge_loss(z, y=1), 'b-', linewidth=2.5, label='y=+1 时的 Hinge Loss')
ax.plot(z, hinge_loss(z, y=-1), 'r-', linewidth=2.5, label='y=-1 时的 Hinge Loss')
ax.axvline(1, color='green', linestyle='--', alpha=0.5, label='间隔边界 z=1')
ax.axvline(-1, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('预测分数 z = w·x + b')
ax.set_ylabel('Hinge Loss')
ax.set_title('Hinge Loss：正确分类且置信时 loss=0')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 4)

# ============================================================
# Part 3: 手写线性 SVM（梯度下降版）
# ============================================================
print("Part 3: 手写 SVM（梯度下降训练）")
print("=" * 50)

class LinearSVM:
    """
    软间隔线性 SVM，用 SGD 训练
    目标：最小化 (1/2)||w||² + C * Σ max(0, 1 - y_i(w·x_i+b))
    """

    def __init__(self, C=1.0, learning_rate=0.01, n_epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            # 计算预测分数
            z = X @ self.w + self.b
            margins = y * z  # y_i * (w·x_i + b)

            # Hinge Loss
            hinge = np.maximum(0, 1 - margins)
            loss = 0.5 * np.dot(self.w, self.w) + self.C * np.mean(hinge)
            self.loss_history.append(loss)

            # 梯度计算
            # 对 w：正则项梯度 + Hinge Loss 梯度
            mask = (margins < 1).astype(float)  # 哪些样本有 hinge loss

            dw = self.w - self.C * (1/n_samples) * (y * mask) @ X
            db = -self.C * (1/n_samples) * np.sum(y * mask)

            # 更新
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                acc = self.score(X, y)
                print(f"  Epoch {epoch:4d}: Loss={loss:.4f}, Acc={acc:.2%}")

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# 训练 SVM
svm = LinearSVM(C=1.0, learning_rate=0.01, n_epochs=800)
svm.fit(X, y)

print(f"\n最终准确率：{svm.score(X, y):.2%}")

# 找支持向量（距决策边界最近的样本）
margins = y * svm.decision_function(X)
support_vector_mask = np.abs(margins - 1) < 0.5
n_sv = support_vector_mask.sum()
print(f"支持向量数量：{n_sv}（这些是最关键的样本！）")

# 可视化训练结果
ax = axes[1]
h = 0.05
xx, yy = np.meshgrid(np.arange(-5, 5, h), np.arange(-5, 5, h))
Z = svm.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=['lightblue', 'lightyellow', 'lightcoral'], alpha=0.4)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
          linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])

ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=60, zorder=5, label='正类')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=60, zorder=5, label='负类')

# 标出支持向量
sv_X = X[support_vector_mask]
ax.scatter(sv_X[:, 0], sv_X[:, 1], s=200, facecolors='none',
          edgecolors='green', linewidths=2.5, zorder=6, label='支持向量')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('SVM 决策边界\n黑线=决策边界，虚线=间隔边界，圈=支持向量')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('01_classical_ml/svm_result.png', dpi=100, bbox_inches='tight')
print("图片已保存：01_classical_ml/svm_result.png")
plt.show()

# ============================================================
# Part 4: 核技巧的直觉（升维让线性不可分变可分）
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 核技巧 —— 升维的魔法")
print("=" * 50)

"""
有些数据，在原始空间中线性不可分（如同心圆）。
但如果把数据"升维"，可能在高维空间中变得线性可分！

例子：同心圆数据
  原始特征：(x1, x2)
  添加特征：x1² + x2²（到原点的距离）
  → 内圈和外圈在第3维上就线性可分了！

核函数的妙处：不需要显式计算高维坐标（可能是无限维！），
而是用核函数直接计算高维空间中的内积。
常用核：
  - 线性核：K(x,z) = x·z
  - 多项式核：K(x,z) = (x·z + c)^d
  - RBF核（高斯核）：K(x,z) = exp(-||x-z||²/(2σ²))
"""

# 生成同心圆数据
n = 200
r_inner = np.random.uniform(0, 1, n//2)
r_outer = np.random.uniform(2, 3, n//2)
theta = np.random.uniform(0, 2*np.pi, n)

X_circle = np.column_stack([
    np.concatenate([r_inner, r_outer]) * np.cos(theta),
    np.concatenate([r_inner, r_outer]) * np.sin(theta)
])
y_circle = np.array([1]*(n//2) + [-1]*(n//2))

# 添加"距离"特征
X_circle_augmented = np.column_stack([X_circle, X_circle[:, 0]**2 + X_circle[:, 1]**2])

fig = plt.figure(figsize=(14, 5))

# 左：原始2D空间（线性不可分）
ax1 = fig.add_subplot(131)
ax1.scatter(X_circle[y_circle==1, 0], X_circle[y_circle==1, 1],
           c='red', s=30, alpha=0.6, label='内圈(+1)')
ax1.scatter(X_circle[y_circle==-1, 0], X_circle[y_circle==-1, 1],
           c='blue', s=30, alpha=0.6, label='外圈(-1)')
ax1.set_title('原始2D空间\n（线性不可分）')
ax1.legend(fontsize=8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 中：添加 r² 特征后的3D空间
ax2 = fig.add_subplot(132, projection='3d')
r_sq = X_circle[:, 0]**2 + X_circle[:, 1]**2
ax2.scatter(X_circle[y_circle==1, 0], X_circle[y_circle==1, 1],
           r_sq[y_circle==1], c='red', s=20, alpha=0.6, label='内圈')
ax2.scatter(X_circle[y_circle==-1, 0], X_circle[y_circle==-1, 1],
           r_sq[y_circle==-1], c='blue', s=20, alpha=0.6, label='外圈')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('r² = x1²+x2²')
ax2.set_title('添加r²特征后的3D空间\n（现在线性可分了！）')

# 右：用 r² 做 SVM
ax3 = fig.add_subplot(133)
# 只用 r² 这一个特征做线性分类
r_sq_vals = r_sq.reshape(-1, 1)
svm_1d = LinearSVM(C=1.0, learning_rate=0.1, n_epochs=500)
svm_1d.fit(r_sq_vals, y_circle)

r_range = np.linspace(0, 10, 200).reshape(-1, 1)
z_vals = svm_1d.decision_function(r_range)
ax3.axhline(0, color='black', linewidth=2, label='决策边界')
ax3.hist(r_sq[y_circle==1], bins=20, alpha=0.6, color='red', label='内圈 r²', density=True)
ax3.hist(r_sq[y_circle==-1], bins=20, alpha=0.6, color='blue', label='外圈 r²', density=True)
threshold = -svm_1d.b / svm_1d.w[0]
ax3.axvline(threshold, color='black', linewidth=2, linestyle='--', label=f'r²={threshold:.1f}')
ax3.set_xlabel('r² = x1² + x2²')
ax3.set_title(f'在r²上做线性分类\n准确率:{svm_1d.score(r_sq_vals, y_circle):.0%}')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_classical_ml/svm_kernel.png', dpi=100, bbox_inches='tight')
print("图片已保存：01_classical_ml/svm_kernel.png")
plt.show()

print("\n核技巧的本质：不显式升维，而是用核函数计算高维内积")
print("SVM + RBF核 可以处理任意非线性边界！")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【支持向量的重要性】
   训练完 SVM 后，如果随机删掉一个非支持向量的样本，
   决策边界会改变吗？为什么？
   这说明 SVM 对哪种样本更敏感？

2. 【C 参数实验】
   生成一个有少量异常点的数据集（大部分线性可分，但有几个越界点）。
   用 C = 0.01, 1, 100 训练 SVM，画出三个决策边界。
   总结 C 大小对边界和间隔的影响。

3. 【SVM vs 逻辑回归】
   两者都做线性分类，主要区别：
   - SVM：最大化间隔（Hinge Loss），只关心支持向量
   - 逻辑回归：最大化似然（交叉熵），所有样本都有贡献

   在什么场景下 SVM 更好？（提示：小数据集？高维数据？）
""")
