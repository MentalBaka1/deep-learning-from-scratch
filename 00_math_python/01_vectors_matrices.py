"""
==============================================================
第0章 第1节：向量与矩阵 —— 空间中的箭头与变换
==============================================================

【为什么需要它？】
深度学习里，所有的数据（图片、文字、音频）都被表示成向量或矩阵。
所有的神经网络计算，本质上都是一堆矩阵乘法。
所以，在学神经网络之前，必须先对"矩阵"有直觉。

【生活类比】
向量 = 地图上的一个箭头，告诉你"往东走3步，往北走4步"
矩阵 = 一种"空间变换"，可以把箭头旋转、缩放、拉伸
矩阵乘法 = 把一个变换应用到很多箭头上（批量处理！）
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Part 1: 向量基础
# ============================================================
print("=" * 50)
print("Part 1: 向量基础")
print("=" * 50)

# 向量就是一组有序的数字
# 在 2D 空间里，[3, 4] 表示"往右3步，往上4步"
v1 = np.array([3, 4])
v2 = np.array([1, -2])

print(f"向量 v1 = {v1}")
print(f"向量 v2 = {v2}")

# 向量加法：两个箭头首尾相接
v_sum = v1 + v2
print(f"v1 + v2 = {v_sum}  （两段路走下来的总位移）")

# 向量缩放
v_scaled = 2 * v1
print(f"2 * v1 = {v_scaled}  （把箭头变长一倍）")

# 向量长度（模）：勾股定理！
length_v1 = np.linalg.norm(v1)
print(f"|v1| = sqrt(3² + 4²) = {length_v1:.2f}  （就是中学学的勾股定理）")

# 单位向量：长度=1的向量，表示"方向"
unit_v1 = v1 / length_v1
print(f"v1的单位向量 = {unit_v1}  （|单位向量| = {np.linalg.norm(unit_v1):.2f}）")

# ============================================================
# Part 2: 点积（内积）—— 衡量两个向量有多"像"
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 点积 —— 两个向量有多相似？")
print("=" * 50)

"""
点积公式：a · b = a₁b₁ + a₂b₂ + ... = |a||b|cos(θ)

几何含义：
  - 点积 > 0：两个向量方向相似（夹角 < 90°）
  - 点积 = 0：两个向量垂直（夹角 = 90°）正交！
  - 点积 < 0：两个向量方向相反（夹角 > 90°）

这个在深度学习里极其重要！
注意力机制（Transformer）就是用点积来计算"两个词有多相关"
"""

a = np.array([1, 0])   # 向右
b = np.array([0, 1])   # 向上
c = np.array([1, 1])   # 右上45度

dot_ab = np.dot(a, b)  # np.dot(a,b)
dot_ac = np.dot(a, c)  

print(f"a = {a} (→), b = {b} (↑)")
print(f"a · b = {dot_ab}  （垂直，完全不相似）")
print(f"a · c = {dot_ac}  （c有向右分量，有些相似）")

# 余弦相似度：把长度归一化后的点积
cos_ab = dot_ab / (np.linalg.norm(a) * np.linalg.norm(b))
cos_ac = dot_ac / (np.linalg.norm(a) * np.linalg.norm(c))
print(f"cos(a,b) = {cos_ab:.2f}  (0=垂直)")
print(f"cos(a,c) = {cos_ac:.2f}  (0.707=45度，有相关性)")

# ============================================================
# Part 3: 矩阵 —— 线性变换
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 矩阵 —— 对空间做变换")
print("=" * 50)

"""
矩阵不只是一堆数字的表格。
矩阵 = 一种"空间变换"，可以：
  - 旋转向量
  - 缩放向量
  - 拉伸/压缩空间

矩阵乘以向量 = 把变换应用到这个向量上
"""

# 旋转矩阵（旋转90度）
theta = np.pi / 2  # 90度
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])
print("旋转90度的矩阵 R =")
print(R.round(2))

v = np.array([1, 0])  # 向右的向量
v_rotated = R @ v     # @ 是矩阵乘法符号
print(f"旋转前：v = {v}")
print(f"旋转后：Rv = {v_rotated.round(2)}  （原来向右，旋转后向上！）")

# 缩放矩阵
S = np.array([
    [2, 0],  # x方向放大2倍
    [0, 3]   # y方向放大3倍
])
v2 = np.array([1, 1])
print(f"\n缩放前：v2 = {v2}")
print(f"缩放后：Sv2 = {S @ v2}  （x变成2，y变成3）")

# ============================================================
# Part 4: 矩阵乘法 —— 批量处理数据
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 矩阵乘法 —— 神经网络的核心操作")
print("=" * 50)

"""
在神经网络里：
  X = 数据矩阵，shape (样本数, 特征数)
  W = 权重矩阵，shape (特征数, 输出数)
  X @ W = 所有样本同时做线性变换，shape (样本数, 输出数)

矩阵乘法 = 一次性处理整个数据集！这就是为什么GPU很快。
"""

# 4个样本，每个有3个特征
X = np.array([
    [1, 2, 3],   # 样本1
    [4, 5, 6],   # 样本2
    [7, 8, 9],   # 样本3
    [1, 0, 1],   # 样本4
])

# 权重矩阵：3个特征 -> 2个输出（类似神经网络的一层）
W = np.array([
    [0.1, 0.4],
    [0.2, 0.5],
    [0.3, 0.6],
])

output = X @ W
print(f"X.shape = {X.shape}  （4个样本，每个3个特征）")
print(f"W.shape = {W.shape}  （3个特征 → 2个输出）")
print(f"(X @ W).shape = {output.shape}  （4个样本，每个2个输出）")
print("\n一次矩阵乘法，同时处理了4个样本！")

# 手写矩阵乘法（理解底层）
def matmul_naive(A, B):
    """
    朴素版矩阵乘法（双重 for 循环）
    C[i,j] = sum_k A[i,k] * B[k,j]
    就是A的第i行 和 B的第j列 做点积！
    """
    m, k1 = A.shape
    k2, n = B.shape
    assert k1 == k2, "维度不匹配！"

    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            for k in range(k1):
                C[i, j] += A[i, k] * B[k, j]  # 点积累加
    return C

output_naive = matmul_naive(X, W)
print(f"\n手写矩阵乘法结果和numpy一致：{np.allclose(output, output_naive)}")

# ============================================================
# Part 5: 广播机制 —— numpy的"自动对齐"
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 广播（Broadcasting）—— 自动扩展维度")
print("=" * 50)

"""
广播规则：
  1. 维度数不同时，在左边补1
  2. 每个维度上，尺寸为1的那方会"扩展"到另一方的尺寸

例子：(4,3) + (3,) = (4,3)
  -> (3,) 被广播成 (4,3)，每一行都加同一个向量

在神经网络里，bias（偏置）的加法就是广播！
  X @ W 的形状是 (batch, output)
  bias 的形状是 (output,)
  X @ W + bias 自动广播，每个样本都加同一个 bias
"""

A = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)
bias = np.array([10, 20, 30])  # shape (3,)

result = A + bias  # bias 自动广播到每一行
print(f"A.shape = {A.shape}, bias.shape = {bias.shape}")
print(f"A + bias =\n{result}")
print("bias 被自动加到了 A 的每一行（就像神经网络加偏置）")

# ============================================================
# Part 6: 可视化 —— 看见矩阵变换
# ============================================================
def visualize_transformation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('矩阵变换的几何直觉', fontsize=14)

    # 原始向量场
    vectors = np.array([[1,0], [0,1], [1,1], [-1,1]])
    colors = ['red', 'blue', 'green', 'purple']
    labels = ['[1,0]', '[0,1]', '[1,1]', '[-1,1]']

    matrices = [
        (np.eye(2), '恒等变换（不变）'),
        (np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                   [np.sin(np.pi/4),  np.cos(np.pi/4)]]), '旋转45度'),
        (np.array([[2, 0], [0, 0.5]]), '缩放（x×2, y×0.5）'),
    ]

    for ax, (M, title) in zip(axes, matrices):
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_aspect('equal')

        for v, color, label in zip(vectors, colors, labels):
            v_transformed = M @ v
            # 原始向量（虚线）
            ax.annotate('', xy=v, xytext=[0,0],
                       arrowprops=dict(arrowstyle='->', color=color,
                                      lw=1.5, linestyle='dashed', alpha=0.4))
            # 变换后向量（实线）
            ax.annotate('', xy=v_transformed, xytext=[0,0],
                       arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.text(v_transformed[0]+0.1, v_transformed[1]+0.1,
                   label, color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig('00_math_python/vectors_transformation.png', dpi=100, bbox_inches='tight')
    print("\n图片已保存：00_math_python/vectors_transformation.png")
    plt.show()

visualize_transformation()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题（动手做！）")
print("=" * 50)
print("""
1. 【向量点积】
   有两个词向量（用随机初始化模拟）：
   word_cat  = np.random.randn(4)
   word_dog  = np.random.randn(4)
   word_car  = np.random.randn(4)

   计算 cat 和 dog 的余弦相似度，以及 cat 和 car 的余弦相似度。
   哪对更相似？（随机初始化下，两对应该差不多，
   但真实词向量 word2vec 中，cat 和 dog 会更相似）

2. 【矩阵乘法的维度】
   如果 X.shape = (32, 784)，W.shape = (784, 128)
   X @ W 的结果 shape 是什么？
   （提示：这就是一个全连接层：32张图片，每张784像素，输出128个特征）

3. 【广播验证】
   运行下面代码，解释为什么会报错，然后修复它：
   a = np.ones((3, 4))
   b = np.ones((4, 3))
   print(a + b)  # 会报错！
""")
