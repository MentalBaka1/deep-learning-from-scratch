"""
==============================================================
第2章 第1节：感知机 —— 最简单的神经元
==============================================================

【为什么需要它？】
1943年，McCulloch 和 Pitts 受大脑神经元启发，
提出了第一个计算模型：人工神经元。
1958年，Rosenblatt 提出感知机学习算法。

感知机是所有深度学习的起点，理解它是理解神经网络的基础。

【生活类比】
感知机 = 一个做决定的人（投票器）
  - 他关注 n 个因素（特征 x1, x2, ..., xn）
  - 每个因素的重要程度不同（权重 w1, w2, ..., wn）
  - 他把加权分数求和，如果超过阈值就说"是"

例子：决定明天是否去野餐
  x1=是否晴天(1/0)  × w1=2.0（很重要）
  x2=是否有钱(1/0)  × w2=1.0（比较重要）
  x3=是否有空(1/0)  × w3=3.0（非常重要）
  如果 2*x1 + 1*x2 + 3*x3 > 3，就去！

【存在理由】
解决问题：如何让计算机从例子中学会做决定？
核心思想：加权投票 + 错误驱动更新
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(42)

# ============================================================
# Part 1: 感知机模型
# ============================================================
print("=" * 50)
print("Part 1: 感知机的结构")
print("=" * 50)

"""
感知机的计算：
  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  （线性组合 + 偏置）
  ŷ = sign(z)  = +1 if z > 0 else -1  （激活函数：阶跃）

感知机学习规则（错误驱动）：
  如果预测对了：什么都不做
  如果预测错了：
    w = w + lr * y_true * x  （把权重往正确方向推）
    b = b + lr * y_true

直觉：如果把"+1"预测成了"-1"：
  - y_true = +1，当前 w·x < 0
  - 更新后 w_new = w + x，让 w_new·x = w·x + x·x > w·x（正向了！）
"""

class Perceptron:
    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.error_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.max_epochs):
            errors = 0
            for i in range(n_samples):
                # 预测
                z = np.dot(self.w, X[i]) + self.b
                y_pred = 1 if z > 0 else -1

                # 更新（只有预测错了才更新！）
                if y_pred != y[i]:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    errors += 1

            self.error_history.append(errors)
            if errors == 0:
                print(f"  在 epoch {epoch+1} 收敛！")
                break

        return self

    def predict(self, X):
        z = X @ self.w + self.b
        return np.where(z > 0, 1, -1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ============================================================
# Part 2: 在线性可分数据上测试
# ============================================================
print("\nPart 2: 感知机在线性可分数据上")
print("=" * 50)

# 生成线性可分数据
n = 100
X_pos = np.random.randn(n//2, 2) + np.array([2, 2])
X_neg = np.random.randn(n//2, 2) + np.array([-2, -2])
X_linear = np.vstack([X_pos, X_neg])
y_linear = np.array([1] * (n//2) + [-1] * (n//2))

perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
perceptron.fit(X_linear, y_linear)
print(f"准确率：{perceptron.score(X_linear, y_linear):.2%}")

# 可视化学习过程
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.scatter(X_pos[:, 0], X_pos[:, 1], c='red', s=40, alpha=0.6, label='+1')
ax.scatter(X_neg[:, 0], X_neg[:, 1], c='blue', s=40, alpha=0.6, label='-1')

# 决策边界：w[0]*x1 + w[1]*x2 + b = 0 → x2 = -(w[0]*x1 + b) / w[1]
x_range = np.linspace(-5, 5, 100)
if abs(perceptron.w[1]) > 1e-6:
    y_boundary = -(perceptron.w[0] * x_range + perceptron.b) / perceptron.w[1]
    ax.plot(x_range, y_boundary, 'k-', linewidth=2.5, label='学到的边界')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title('感知机：线性可分')
ax.legend()
ax.grid(True, alpha=0.3)

# 错误历史
ax = axes[1]
ax.plot(perceptron.error_history, 'r-o', markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('分类错误数')
ax.set_title('学习过程：错误逐渐减少到0')
ax.grid(True, alpha=0.3)

# ============================================================
# Part 3: 感知机的局限 —— XOR 问题
# ============================================================
print("\nPart 3: 感知机的致命弱点 —— XOR 问题")
print("=" * 50)

"""
XOR（异或）问题：
  x1=0, x2=0 → 0
  x1=0, x2=1 → 1
  x1=1, x2=0 → 1
  x1=1, x2=1 → 0

这个规律不能用一条直线分开（非线性！）
感知机（线性分类器）无法解决 XOR 问题。

这个"缺陷"在 1969 年被 Minsky 和 Papert 指出，
导致了第一次"AI 寒冬"！

解决方法：多层感知机（MLP），也就是深度神经网络。
"""

# XOR 数据
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1])  # 用 ±1 表示

print("XOR 数据：")
for x, label in zip(X_xor, y_xor):
    print(f"  x={x}，y={label}")

# 尝试用感知机学习 XOR
perceptron_xor = Perceptron(learning_rate=0.1, max_epochs=1000)
perceptron_xor.fit(X_xor, y_xor)
acc = perceptron_xor.score(X_xor, y_xor)
print(f"\n感知机在 XOR 上的准确率：{acc:.2%}  （无法达到100%！）")
print(f"收敛了吗？错误历史最后5个：{perceptron_xor.error_history[-5:]}")
print("\n结论：单层感知机无法解决非线性问题！需要多层（MLP）。")

# 可视化 XOR 无法线性分割
ax = axes[2]
colors = ['red' if y == 1 else 'blue' for y in y_xor]
markers = ['o', '^', 's', 'D']
for i, (x, y, c, m) in enumerate(zip(X_xor, y_xor, colors, markers)):
    ax.scatter(x[0], x[1], c=c, s=300, marker=m, zorder=5,
              label=f'({x[0]:.0f},{x[1]:.0f})→{y}')

# 尝试画几条直线，没有一条能分开
for w1, w2, b, color in [(1, 1, -1.5, 'gray'),
                          (1, -1, 0, 'purple'),
                          (0.5, 0.5, -0.8, 'brown')]:
    x_r = np.linspace(-0.5, 1.5, 100)
    if abs(w2) > 0.001:
        y_r = -(w1 * x_r + b) / w2
        ax.plot(x_r, y_r, '-', color=color, alpha=0.5, linewidth=1.5)

ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
ax.set_title('XOR 问题\n任何直线都无法分开\n→ 需要多层网络！')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

ax.text(0.5, -0.4, '这4个点无法被直线分成正确的两组', ha='center', fontsize=9, color='red')

plt.tight_layout()
plt.savefig('02_neural_networks/perceptron.png', dpi=100, bbox_inches='tight')
print("\n图片已保存：02_neural_networks/perceptron.png")
plt.show()

# ============================================================
# Part 4: 为什么需要多层？——可视化"隐藏层的魔法"
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 多层如何解决 XOR？（直觉）")
print("=" * 50)

"""
想法：如果我们先用两个神经元做"预处理"，
把 XOR 的数据变换到一个新的空间，
也许在新空间里就线性可分了？

手动设计的两层网络（不通过训练）：
  隐藏神经元1：h1 = step(x1 + x2 - 1.5)  （检测"至少一个为1"）
  隐藏神经元2：h2 = step(x1 + x2 - 0.5)  （检测"至少有一个为1"）
  输出：y = step(h1 - h2 + 0.5) （一个1但不是两个1）

新特征 (h1, h2) 在新空间里是线性可分的！
"""

def step(x):
    return (x > 0).astype(float)

# 手动设计的权重（通常由训练得到）
print("手动设计两层网络解决 XOR：")
for x in X_xor:
    h1 = step(x[0] + x[1] - 1.5)  # AND 门
    h2 = step(x[0] + x[1] - 0.5)  # OR 门（近似）
    output = step(-2 * h1 + h2 - 0.5)  # XOR = OR AND (NOT AND)
    print(f"  x={x.astype(int)}: h1={h1:.0f}, h2={h2:.0f} → output={output:.0f}  "
          f"（正确答案：{(x[0] != x[1]):.0f}）")

print("\n通过增加一个隐藏层，XOR 问题被解决了！")
print("这就是深度学习的核心：用多层网络学习复杂的非线性特征")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【感知机收敛定理】
   如果数据线性可分，感知机算法保证在有限步内收敛。
   如果数据线性不可分（如XOR），感知机会永远震荡。
   修改代码，画出 XOR 上前100个epoch的错误数曲线，
   验证它不会收敛到0。

2. 【特征空间变换】
   对 XOR 数据，添加特征 x3 = x1 * x2（交叉项）。
   在新的3D空间 (x1, x2, x1*x2) 里，用感知机能解决吗？
   （提示：尝试一下！这说明特征工程可以解决非线性问题）

3. 【神经元类比】
   真实生物神经元：
   - 树突（dendrite）= 接收输入信号
   - 细胞体 = 累加信号
   - 轴突 (axon) = 如果信号超过阈值，发出信号

   感知机的哪些部分对应以上三个结构？
   生物神经元的激活是"全有全无"（阶跃函数），
   为什么人工神经元要用平滑的 sigmoid/ReLU？
""")
