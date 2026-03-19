"""
====================================================================
第0章 · 第1节 · 向量与矩阵
====================================================================

【一句话总结】
向量和矩阵是深度学习的"数据容器"和"变换工具"——神经网络的每一步计算
本质上都是矩阵乘法。

【为什么深度学习需要这个？】
- 输入数据（图像、文本嵌入）都是向量或矩阵
- 神经网络的权重就是矩阵，前向传播 = 矩阵乘法 + 非线性
- 不理解矩阵运算，就无法理解网络在做什么

【核心概念】

1. 向量（Vector）
   - 定义：有方向和大小的量，在代码中就是一维数组
   - 直觉：想象空间中的一支箭头，从原点指向某个方向
   - 在深度学习中：一个样本的特征、一个词的嵌入、一层的输出

2. 矩阵（Matrix）
   - 定义：二维数组，m行n列
   - 直觉：一张电子表格，或一个"变换机器"
   - 在深度学习中：权重矩阵W、一个batch的数据、注意力分数矩阵

3. 点积（Dot Product）
   - 定义：对应元素相乘再求和，结果是标量
   - 几何意义：衡量两个向量的"相似程度"
   - 在深度学习中：注意力机制中Q·K就是点积！

4. 矩阵乘法（Matrix Multiplication）
   - 定义：A(m×k) × B(k×n) = C(m×n)
   - 直觉：对B的每一列做线性组合
   - 在深度学习中：y = Wx + b 就是矩阵乘法

5. 广播（Broadcasting）
   - NumPy 自动扩展维度不同的数组进行运算
   - 在深度学习中：偏置b加到每个样本上就是广播

【前置知识】
Python 基础、NumPy 基本用法

====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# 设置中文字体显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ====================================================================
# 第一部分：向量基础
# ====================================================================
print("=" * 60, "\n第一部分：向量基础\n" + "=" * 60)

# --- 1.1 创建向量 ---
# 深度学习中的向量：MNIST 像素(784维)、词嵌入(300维)、隐藏层输出
a = np.array([3.0, 4.0])    # 二维向量，便于可视化
b = np.array([1.0, 5.0])
print(f"向量 a = {a}, 形状 = {a.shape}")
print(f"向量 b = {b}, 形状 = {b.shape}")

# --- 1.2 向量运算 ---
# 加法 → 残差连接(ResNet)  |  标量乘法 → 学习率×梯度  |  范数 → L2正则化
print(f"\n加法: a+b = {a + b}    标量乘: 2a = {2*a}")
norm_a = np.linalg.norm(a)
print(f"L2范数 |a| = {norm_a}")  # sqrt(9+16) = 5

# --- 1.3 点积与余弦相似度 ---
# 点积 = 对应元素相乘求和 → 注意力机制 Q·K 的核心
# 几何意义: a·b = |a||b|cos(θ)，衡量方向一致程度
dot_product = np.dot(a, b)   # 3*1 + 4*5 = 23
cos_sim = dot_product / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(np.clip(cos_sim, -1, 1))
print(f"\n点积 a·b = {dot_product}")
print(f"余弦相似度 = {cos_sim:.4f}，夹角 = {np.degrees(theta):.2f}°")

# --- 1.4 可视化向量与投影 ---
proj_length = dot_product / np.linalg.norm(b)
proj_vec = proj_length * (b / np.linalg.norm(b))  # a 在 b 上的投影

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, title, extra in [
    (axes[0], f"向量可视化 (夹角={np.degrees(theta):.1f}°)", None),
    (axes[1], "点积的几何意义：投影", proj_vec),
]:
    ax.quiver(0, 0, a[0], a[1], angles="xy", scale_units="xy", scale=1,
              color="steelblue", label=f"a={a}", linewidth=2)
    ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1,
              color="tomato", label=f"b={b}", linewidth=2)
    if extra is not None:  # 画投影向量
        ax.quiver(0, 0, extra[0], extra[1], angles="xy", scale_units="xy",
                  scale=1, color="green", label="a在b上的投影", linewidth=2.5)
        ax.plot([a[0], extra[0]], [a[1], extra[1]], "k--", alpha=0.4)
    ax.set_xlim(-1, 7); ax.set_ylim(-1, 7); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10); ax.set_title(title, fontsize=13)
    ax.axhline(y=0, color="k", lw=0.5); ax.axvline(x=0, color="k", lw=0.5)
plt.tight_layout(); plt.savefig("01_vectors.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 01_vectors.png\n")

# ====================================================================
# 第二部分：矩阵作为线性变换
# ====================================================================
print("=" * 60, "\n第二部分：矩阵作为线性变换\n" + "=" * 60)

# 核心直觉：矩阵是"变换机器"，输入向量 → 输出另一个向量
# 神经网络每一层 y = Wx 就是用 W 对输入做线性变换

# 旋转矩阵（逆时针 45°）
angle = np.radians(45)
R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]])
# 缩放矩阵（x 拉伸 2 倍，y 压缩一半）
S = np.array([[2.0, 0.0], [0.0, 0.5]])
print(f"旋转矩阵 R(45°):\n{R}\n缩放矩阵 S:\n{S}")

# 对单位圆上的点施加变换
t = np.linspace(0, 2 * np.pi, 100)
circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2, 100)
transforms = [R @ circle, S @ circle, S @ R @ circle]
titles = ["旋转 (45°)", "缩放 (x*2, y*0.5)", "组合 (旋转+缩放)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, title, c in zip(axes, transforms, titles, ["steelblue","tomato","green"]):
    ax.plot(circle[0], circle[1], "k--", alpha=0.3, label="原始圆")
    ax.plot(data[0], data[1], color=c, linewidth=2, label="变换后")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    ax.set_title(title, fontsize=13); ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
plt.suptitle("矩阵=线性变换：神经网络每一层都在做这件事", fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig("01_matrix_transform.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 01_matrix_transform.png")
# 启示：权重矩阵 W 把数据从一个空间"搬"到另一个空间，使分类更容易
print("启示: y = Wx + b 本质 = 线性变换(W) + 平移(b)\n")

# ====================================================================
# 第三部分：矩阵乘法——手写实现 vs NumPy
# ====================================================================
print("=" * 60, "\n第三部分：矩阵乘法手写实现\n" + "=" * 60)

# 规则: A(m*k) @ B(k*n) = C(m*n)，A 的列数必须等于 B 的行数
# C[i][j] = sum(A[i][p] * B[p][j] for p in range(k))

def matmul_naive(A, B):
    """手写三重循环矩阵乘法——理解原理用，实际中太慢绝不要用"""
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, f"维度不匹配! A列数({k}) != B行数({k2})"
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[p, j]
    return C

# 模拟前向传播: x(1,3) @ W(3,2) = y(1,2)
x = np.array([[1.0, 2.0, 3.0]])
W = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
y_naive = matmul_naive(x, W)
y_numpy = x @ W  # Python 3.5+ 的 @ 运算符，等价于 np.matmul
print(f"x{x.shape} @ W{W.shape} = y{y_numpy.shape}")
print(f"手写: {y_naive}  NumPy: {y_numpy}  一致: {np.allclose(y_naive, y_numpy)}")

# 性能对比
A_big = np.random.randn(64, 128)
B_big = np.random.randn(128, 256)
t0 = time.time(); C_naive = matmul_naive(A_big, B_big); t_naive = time.time() - t0
t0 = time.time(); C_numpy = A_big @ B_big; t_numpy = time.time() - t0
print(f"\n性能对比 (64*128 @ 128*256):")
print(f"  手写循环: {t_naive:.4f}s  |  NumPy: {t_numpy:.6f}s  |  加速: {t_naive/max(t_numpy,1e-9):.0f}x")
print(f"  结果一致: {np.allclose(C_naive, C_numpy)}")
print("结论: 永远用 NumPy/PyTorch 的矩阵运算，不要手写循环!\n")

# ====================================================================
# 第四部分：广播机制（Broadcasting）
# ====================================================================
print("=" * 60, "\n第四部分：广播机制\n" + "=" * 60)

# 规则：从右向左对齐维度，每个维度要么相同，要么其中一个为 1
# 最常见场景：y = Wx + b，偏置 b 广播到 batch 的每个样本

# 4.1 偏置广播: (4,3) + (3,) → (4,3)
batch_out = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.], [10.,11.,12.]])
bias = np.array([0.1, 0.2, 0.3])
result = batch_out + bias   # b 自动复制4份，分别加到每个样本
print(f"batch{batch_out.shape} + bias{bias.shape} → {result.shape}")
print(f"结果:\n{result}")

# 等价的手动循环（低效，仅为理解）
result_loop = np.zeros_like(batch_out)
for i in range(batch_out.shape[0]):
    result_loop[i] = batch_out[i] + bias
print(f"手动循环结果一致: {np.allclose(result, result_loop)}")

# 4.2 特征归一化: (5,3) - (3,) → 广播减均值
data = np.random.randn(5, 3)
normalized = data - data.mean(axis=0)  # 每列减去该列均值
print(f"\n归一化: {data.shape} - {data.mean(axis=0).shape} → 列均值接近0: "
      f"{normalized.mean(axis=0).round(10)}")

# 4.3 按样本加权: (5,3) * (5,1) → (5,3)
weights = np.array([[0.5],[1.0],[1.5],[2.0],[0.8]])
print(f"加权: {data.shape} * {weights.shape} → {(data * weights).shape}")

# 4.4 广播陷阱
try:
    _ = np.ones((3, 4)) + np.ones((5,))  # (3,4)+(5,) 不兼容!
except ValueError as e:
    print(f"\n广播陷阱 — 维度不兼容: {e}")
print("调试技巧: 遇到维度错误，先打印所有张量的 .shape\n")

# ====================================================================
# 第五部分：维度分析练习
# ====================================================================
print("=" * 60, "\n第五部分：维度分析练习\n" + "=" * 60)
# 深度学习最常犯的错误 = 维度不匹配。习惯：先写形状，再写代码！

# --- 练习1: 两层全连接网络 (MNIST 分类) ---
batch_size, in_dim, hid_dim, out_dim = 32, 784, 256, 10
X  = np.random.randn(batch_size, in_dim)   # (32, 784) 输入
W1 = np.random.randn(in_dim, hid_dim)      # (784, 256) 第一层权重
b1 = np.random.randn(hid_dim)              # (256,)     偏置
W2 = np.random.randn(hid_dim, out_dim)     # (256, 10)  第二层权重
b2 = np.random.randn(out_dim)              # (10,)      偏置

H = np.maximum(0, X @ W1 + b1)  # ReLU( (32,784)@(784,256)+(256,) ) → (32,256)
Y = H @ W2 + b2                 # (32,256)@(256,10)+(10,) → (32,10)
print("全连接网络维度流:")
print(f"  X{X.shape} @ W1{W1.shape} + b1{b1.shape} → H{H.shape}")
print(f"  H{H.shape} @ W2{W2.shape} + b2{b2.shape} → Y{Y.shape}")

# --- 练习2: 注意力机制 (Transformer 核心) ---
seq_len, d_model = 10, 64
Q = np.random.randn(seq_len, d_model)  # (10, 64)
K = np.random.randn(seq_len, d_model)  # (10, 64)
V = np.random.randn(seq_len, d_model)  # (10, 64)

scores = Q @ K.T / np.sqrt(d_model)    # (10,64)@(64,10) → (10,10)
# softmax: 每行归一化为概率分布
attn_w = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
output = attn_w @ V                    # (10,10)@(10,64) → (10,64)
print(f"\n注意力机制维度流:")
print(f"  Q{Q.shape} @ K^T{K.T.shape} / sqrt({d_model}) → scores{scores.shape}")
print(f"  softmax(scores) → attn{attn_w.shape} (每行和={attn_w[0].sum():.4f})")
print(f"  attn{attn_w.shape} @ V{V.shape} → output{output.shape}\n")

# ====================================================================
# 第六部分：实验与可视化
# ====================================================================
print("=" * 60, "\n第六部分：实验与可视化\n" + "=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 6.1 权重矩阵变换数据点（模拟二分类）
ax = axes[0]
np.random.seed(42)
c0 = np.random.randn(30, 2) * 0.5 + [1, 1]    # 类别0
c1 = np.random.randn(30, 2) * 0.5 + [-1, -1]   # 类别1
pts = np.vstack([c0, c1])
W_demo = np.array([[1.5, -0.5], [0.5, 1.2]])    # 模拟学到的权重
out = pts @ W_demo
ax.scatter(pts[:30,0], pts[:30,1], c="steelblue", alpha=.5, label="类0(原)")
ax.scatter(pts[30:,0], pts[30:,1], c="tomato", alpha=.5, label="类1(原)")
ax.scatter(out[:30,0], out[:30,1], c="steelblue", alpha=.8, marker="^", label="类0(变换后)")
ax.scatter(out[30:,0], out[30:,1], c="tomato", alpha=.8, marker="^", label="类1(变换后)")
ax.legend(fontsize=8); ax.set_title("权重矩阵对数据的变换", fontsize=12)
ax.grid(True, alpha=0.3); ax.set_aspect("equal")

# 6.2 权重矩阵热力图
ax = axes[1]
W_vis = np.random.randn(8, 8)
im = ax.imshow(W_vis, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("权重矩阵热力图", fontsize=12)
ax.set_xlabel("输出神经元"); ax.set_ylabel("输入神经元")
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{W_vis[i,j]:.1f}", ha="center", va="center",
                fontsize=7, color="w" if abs(W_vis[i,j]) > 1 else "k")

# 6.3 注意力权重热力图
ax = axes[2]
words = ["我", "喜", "欢", "深", "度", "学", "习"]
n = len(words)
attn_demo = np.eye(n)*0.4 + np.random.rand(n, n)*0.1
attn_demo /= attn_demo.sum(axis=-1, keepdims=True)
im = ax.imshow(attn_demo, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(words, fontsize=11); ax.set_yticklabels(words, fontsize=11)
ax.set_title("注意力权重矩阵\n(行=Query, 列=Key)", fontsize=12)
ax.set_xlabel("Key"); ax.set_ylabel("Query")

plt.tight_layout(); plt.savefig("01_visualizations.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 01_visualizations.png\n")

# ====================================================================
# 第七部分：思考题
# ====================================================================
print("=" * 60, "\n第七部分：思考题\n" + "=" * 60)
print("""
1.【向量维度与信息容量】
  Word2Vec 用 300 维，BERT 用 768 维。维度越高能记住的信息越多，
  但能无限增大吗？会有什么问题？（提示：维度灾难、计算成本）

2.【矩阵乘法不可交换】
  AB ≠ BA（一般情况）。这对神经网络层的顺序意味着什么？
  验证: A, B = np.random.randn(3,3), np.random.randn(3,3)
        print(np.allclose(A @ B, B @ A))  # 几乎一定 False

3.【广播的本质】
  y = X @ W + b 中 b 被广播。如果不用广播而用显式循环，
  代码怎么写？性能差多少？（提示：用 %%timeit 比较）

4.【从维度推断结构】
  输入 (64, 100)，输出 (64, 10)。W 和 b 的形状是什么？
  如果加一个 50 维隐藏层，所有参数的形状分别是？

5.【为什么要转置 K？】
  注意力机制算 Q @ K.T 而非 Q @ K。如果直接算 Q @ K 会怎样？
  为什么必须转置？（提示：维度要求 + 输出形状的含义）
""")

# ====================================================================
# 总结
# ====================================================================
print("=" * 60, "\n总结：本节核心要点\n" + "=" * 60)
print("""
  1. 向量是数据的载体，矩阵是变换的载体
  2. 点积衡量相似度 → 注意力机制的基石
  3. 矩阵乘法 = 线性变换 → 神经网络每一层的核心操作
  4. 广播让我们高效处理 batch 数据，避免手写循环
  5. 维度分析是深度学习工程师的必备技能：先写形状，再写代码

  下一节预告：第0章 · 第2节 · 导数与梯度
  → 有了矩阵运算，接下来学习如何计算"误差对参数的影响"
""")
