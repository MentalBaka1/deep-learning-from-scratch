"""
====================================================================
第2章 · 第1节 · 感知机
====================================================================

【一句话总结】
感知机是最简单的神经网络——一个单层的线性分类器。它的局限性（解不了XOR）
直接推动了多层网络和深度学习的诞生。

【为什么深度学习需要这个？】
- 感知机是理解神经元的起点：输入→加权求和→激活→输出
- XOR 问题证明了单层的局限，这是深度学习"深度"的根本动机
- 感知机学习算法是梯度下降的简化版

【核心概念】

1. 生物神经元与人工神经元
   - 生物：树突接收信号 → 胞体整合 → 超过阈值 → 轴突发射
   - 人工：输入×权重求和 → 加偏置 → 激活函数 → 输出
   - 对应：突触强度 = 权重，阈值 = 偏置的负数

2. 感知机模型
   - y = sign(w·x + b)
   - 权重 w：每个输入特征的重要程度
   - 偏置 b：决策阈值的调整
   - sign：阶跃函数（>0输出+1，<0输出-1）

3. 感知机学习算法
   - 如果分类正确：不更新
   - 如果分类错误：w = w + η·y·x, b = b + η·y
   - 保证：如果数据线性可分，有限步内必收敛

4. 线性可分性
   - 能被一条直线（超平面）完美分开的数据 → 线性可分
   - AND、OR 可以，XOR 不行！
   - XOR 问题证明了感知机的根本局限

5. 从感知机到多层网络
   - 解决方案：堆叠多层感知机（MLP）
   - 加上非线性激活函数 → 可以逼近任意函数
   - 这就是"深度学习"的起源

【前置知识】
第0章 - 向量点积，第1章 - 分类基础
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ====================================================================
# 第一部分：生物神经元与人工神经元的类比
# ====================================================================
print("=" * 60)
print("第一部分：生物神经元与人工神经元的类比")
print("=" * 60)

#   生物神经元                   人工神经元（感知机）
#   ──────────                   ──────────────────
#   树突接收信号                  输入 x1, x2, ..., xn
#   突触有不同强度                权重 w1, w2, ..., wn
#   胞体整合所有信号              加权求和 z = w·x + b
#   超过阈值 → 轴突发射          激活函数 y = sign(z)
#   发射/不发射（全有或全无）     输出 +1 或 -1
#
# 关键类比：突触强度 = 权重，阈值 = -偏置

print("""
  x1 ──w1──┐
  x2 ──w2──┼──→ [求和 z=w·x+b] ──→ [sign(z)] ──→ y
  x3 ──w3──┘
       +b(偏置)

  y = sign(w1*x1 + w2*x2 + ... + b)
  y = +1 若 z > 0（激活）;  y = -1 若 z <= 0（静默）
""")

# 具体计算示例
x_demo, w_demo, b_demo = np.array([0.6, 0.8]), np.array([0.5, -0.3]), 0.1
z_demo = np.dot(w_demo, x_demo) + b_demo
print(f"示例: x={x_demo}, w={w_demo}, b={b_demo}")
print(f"  z = 0.5*0.6+(-0.3)*0.8+0.1 = {z_demo:.2f}, y = sign({z_demo:.2f}) = {1 if z_demo>0 else -1}\n")


# ====================================================================
# 第二部分：感知机类的实现
# ====================================================================
print("=" * 60)
print("第二部分：感知机类的实现")
print("=" * 60)

class Perceptron:
    """
    感知机分类器：y = sign(w·x + b)
    学习规则：分类错误时 w += η*y*x, b += η*y；正确时不更新。
    """

    def __init__(self, lr=1.0, n_iter=100):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None
        self.errors_per_epoch = []  # 每轮错误数，用于观察收敛

    def fit(self, X, y):
        """训练感知机。X: (n_samples, n_features), y: +1/-1 标签。"""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors_per_epoch = []

        for epoch in range(self.n_iter):
            errors = 0
            for i in range(n_samples):
                y_hat = self.predict_one(X[i])
                if y_hat != y[i]:
                    # 朝正确方向"推"决策边界
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:  # 全部正确，提前停止
                break
        return self

    def predict_one(self, x):
        """单样本预测"""
        return 1 if np.dot(self.w, x) + self.b > 0 else -1

    def predict(self, X):
        """批量预测"""
        return np.array([self.predict_one(x) for x in X])

    def score(self, X, y):
        """计算准确率"""
        return np.mean(self.predict(X) == y)


print("Perceptron 类已定义: fit(X,y) 训练, predict(X) 预测, score(X,y) 评估\n")


# ====================================================================
# 第三部分：AND / OR 门学习
# ====================================================================
print("=" * 60)
print("第三部分：AND / OR 门学习")
print("=" * 60)

# AND: 只有 (1,1)→+1，其余→-1      OR: 只有 (0,0)→-1，其余→+1
X_logic = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([-1, -1, -1, 1])
y_or = np.array([-1, 1, 1, 1])

p_and = Perceptron(lr=1.0, n_iter=20).fit(X_logic, y_and)
p_or = Perceptron(lr=1.0, n_iter=20).fit(X_logic, y_or)

for name, model, y_true in [("AND", p_and, y_and), ("OR", p_or, y_or)]:
    print(f"{name}: w={model.w}, b={model.b}, "
          f"预测={model.predict(X_logic)}, 准确率={model.score(X_logic, y_true)*100:.0f}%, "
          f"收敛轮数={len(model.errors_per_epoch)}")
print()


# ====================================================================
# 第四部分：XOR 失败演示
# ====================================================================
print("=" * 60)
print("第四部分：XOR 失败演示——感知机的根本局限")
print("=" * 60)

# XOR: (0,0)→-1, (0,1)→+1, (1,0)→+1, (1,1)→-1
# 无论怎么画一条直线，都无法正确分开这四个点！
y_xor = np.array([-1, 1, 1, -1])

p_xor = Perceptron(lr=1.0, n_iter=100).fit(X_logic, y_xor)
print(f"XOR（训练100轮）: w={p_xor.w}, b={p_xor.b}")
print(f"  预测={p_xor.predict(X_logic)}, 真实={y_xor}")
print(f"  准确率={p_xor.score(X_logic, y_xor)*100:.0f}%")
print(f"  最后10轮错误数: {p_xor.errors_per_epoch[-10:]}")
print("结论: 错误数永远无法降到0——XOR 线性不可分！")
print("1969年 Minsky & Papert 的证明导致了第一次 AI 寒冬。\n")


# ====================================================================
# 第五部分：决策边界可视化
# ====================================================================
print("=" * 60)
print("第五部分：决策边界可视化")
print("=" * 60)


def plot_decision_boundary(ax, predict_fn, w, b, X, y, title):
    """绘制决策边界和数据点。w·x+b=0 在二维中是直线。"""
    for label, c, m in [(1, "tomato", "o"), (-1, "steelblue", "s")]:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=c, marker=m, s=120,
                   edgecolors="k", linewidths=1.2, label=f"{label:+d}", zorder=5)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-2, 0, 2], colors=["#AEC7E8", "#FFAEB9"], alpha=0.3)
    if w is not None and abs(w[1]) > 1e-10:  # 画决策边界线
        x1_ln = np.linspace(-0.5, 1.5, 100)
        x2_ln = -(w[0] * x1_ln + b) / w[1]
        ok = (x2_ln >= -0.5) & (x2_ln <= 1.5)
        ax.plot(x1_ln[ok], x2_ln[ok], "k--", linewidth=2)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(title, fontsize=12); ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)


# 决策边界三合一图
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, mdl, yt, name, ok in [(axes[0], p_and, y_and, "AND", True),
                                (axes[1], p_or, y_or, "OR", True),
                                (axes[2], p_xor, y_xor, "XOR", False)]:
    status = "学会了!" if ok else "失败!"
    plot_decision_boundary(ax, mdl.predict, mdl.w, mdl.b, X_logic, yt,
                           f"{name} 门 ({status})")

plt.suptitle("感知机决策边界：AND/OR 成功，XOR 失败", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("01_perceptron_boundaries.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 01_perceptron_boundaries.png")

# 收敛过程图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, mdl, name in [(axes[0], p_and, "AND"),
                        (axes[1], p_or, "OR"),
                        (axes[2], p_xor, "XOR")]:
    ax.plot(range(1, len(mdl.errors_per_epoch)+1),
            mdl.errors_per_epoch, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("训练轮数"); ax.set_ylabel("错误样本数")
    ax.set_title(f"{name} 收敛过程"); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_perceptron_convergence.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 01_perceptron_convergence.png\n")


# ====================================================================
# 第六部分：从感知机到 MLP——两层网络解决 XOR
# ====================================================================
print("=" * 60)
print("第六部分：从感知机到 MLP——两层网络解决 XOR")
print("=" * 60)

# 思路：一个感知机画一条线，两个感知机画两条线，组合起来圈出 XOR 区域
#   隐藏层神经元1: h1 = sign(x1 + x2 - 0.5)   → OR
#   隐藏层神经元2: h2 = sign(-x1 - x2 + 1.5)  → NAND
#   输出层:        y  = sign(h1 + h2 - 1.5)    → AND(h1, h2)
# 组合效果: XOR = AND(OR, NAND)

W1_mlp = np.array([[1.0, 1.0], [-1.0, -1.0]])  # 隐藏层权重
b1_mlp = np.array([-0.5, 1.5])                   # 隐藏层偏置
W2_mlp = np.array([1.0, 1.0])                    # 输出层权重
b2_mlp = -1.5                                     # 输出层偏置


def mlp_predict(X):
    """两层网络前向传播"""
    results = []
    for x in X:
        h = np.where(W1_mlp @ x + b1_mlp > 0, 1.0, -1.0)  # 隐藏层
        y = 1 if np.dot(W2_mlp, h) + b2_mlp > 0 else -1    # 输出层
        results.append(y)
    return np.array(results)


print("两层网络逐步计算 XOR:")
for i, xi in enumerate(X_logic):
    h = np.where(W1_mlp @ xi + b1_mlp > 0, 1, -1)
    yi = 1 if np.dot(W2_mlp, h.astype(float)) + b2_mlp > 0 else -1
    print(f"  {xi} → h={h} → y={yi:+d} (真实{y_xor[i]:+d}) {'OK' if yi==y_xor[i] else 'X'}")
print(f"准确率: {np.mean(mlp_predict(X_logic)==y_xor)*100:.0f}%\n")

# 对比可视化：单层 vs 两层
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

plot_decision_boundary(axes[0], p_xor.predict, p_xor.w, p_xor.b,
                       X_logic, y_xor, "单层感知机 vs XOR (失败)")

plot_decision_boundary(axes[1], mlp_predict, None, None,
                       X_logic, y_xor, "两层网络 vs XOR (成功!)")
# 在右图上叠画两条隐藏层决策线
x1_line = np.linspace(-0.5, 1.5, 100)
axes[1].plot(x1_line, -x1_line + 0.5, "g--", lw=2, alpha=0.7, label="隐藏线1(OR)")
axes[1].plot(x1_line, -x1_line + 1.5, "m--", lw=2, alpha=0.7, label="隐藏线2(NAND)")
axes[1].legend(fontsize=9, loc="upper left")

plt.suptitle("深度的力量：多层网络突破线性局限", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("01_perceptron_vs_mlp.png", dpi=100, bbox_inches="tight")
plt.show()
print("[图片已保存] 01_perceptron_vs_mlp.png")
print("核心启示：单层→一条线→线性问题; 多层→多条线组合→非线性问题; 更深→任意函数\n")


# ====================================================================
# 第七部分：思考题
# ====================================================================
print("=" * 60)
print("第七部分：思考题")
print("=" * 60)
print("""
1. 【决策边界的维度】
   二维输入时决策边界是直线。三维呢？n维呢？
   提示：搜索"超平面"(hyperplane)。

2. 【学习率实验】
   用 η=0.1, 1.0, 10.0 分别训练 AND 感知机，比较收敛速度和最终权重。
   >>> for lr in [0.1, 1.0, 10.0]:
   ...     p = Perceptron(lr=lr).fit(X_logic, y_and)
   ...     print(f"lr={lr}: {len(p.errors_per_epoch)}轮, w={p.w}, b={p.b}")

3. 【收敛定理】
   感知机收敛定理保证线性可分数据有限步收敛，但步数与什么有关？
   提示：与数据的"间隔"(margin)成反比——两类越远收敛越快。

4. 【自动学习权重】
   本节手工设定了 MLP 权重解 XOR。如何让网络自动学到这些权重？
   提示：这正是下一节"反向传播"要解决的问题。

5. 【更多逻辑门】
   NAND、NOR 是否线性可分？写出真值表并用感知机验证。
   提示：NAND 和 NOR 都是"通用门"——任何逻辑函数都可以仅用它们实现。
""")


# ====================================================================
# 总结
# ====================================================================
print("=" * 60)
print("总结：本节核心要点")
print("=" * 60)
print("""
  1. 感知机 = 最简单的神经网络：y = sign(w·x + b)
  2. 学习算法：错误时更新 w 和 b，正确时不动
  3. 能力边界：只能解线性可分问题（AND/OR 可以，XOR 不行）
  4. XOR 问题是深度学习的起源动机：单层不够，需要多层
  5. 两层网络就能解 XOR：组合多条决策线形成非线性边界

  下一节预告：第2章·第2节·反向传播与多层感知机
  → 学习如何让多层网络自动找到正确权重（梯度下降 + 链式法则）
""")
