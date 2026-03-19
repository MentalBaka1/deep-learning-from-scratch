"""
====================================================================
第3章 · 第1节 · 卷积与池化
====================================================================

【一句话总结】
卷积是一种"局部连接+参数共享"的操作——它让网络能高效地从图像中
提取特征，而不需要像全连接那样连接每一个像素。

【为什么深度学习需要这个？】
- 全连接处理 224×224 图像：224×224×3 = 150,528 个输入，参数量爆炸
- 卷积利用图像的两个关键性质：局部性（邻近像素相关）、平移不变性（猫在哪都是猫）
- CNN 是计算机视觉的基础，理解卷积也有助于理解 Transformer 中的 1D 卷积

【核心概念】

1. 卷积操作（Convolution）
   - 一个小的"滤波器"（kernel）在图像上滑动
   - 每个位置：滤波器与局部区域做点积 → 一个输出值
   - 类比：用手电筒扫描图像，每次只看一小块
   - 不同的滤波器检测不同的特征（边缘、纹理、颜色...）

2. 关键超参数
   - 卷积核大小（Kernel Size）：通常 3×3 或 5×5
   - 步长（Stride）：滤波器每次移动多少像素
   - 填充（Padding）：在边缘补零，控制输出尺寸
   - 输出尺寸 = (输入 - 核 + 2×填充) / 步长 + 1

3. 特征图（Feature Map）
   - 一个卷积核产生一张特征图
   - N 个卷积核 → N 张特征图（N 个通道）
   - 浅层特征图：边缘、角点；深层特征图：纹理、物体部件

4. 池化（Pooling）
   - 最大池化（Max Pooling）：取局部区域最大值
   - 平均池化（Average Pooling）：取局部区域平均值
   - 作用：降低分辨率，减少参数，增大感受野

5. im2col 加速技巧
   - 将卷积操作转换为矩阵乘法
   - 虽然增加内存使用，但利用了高度优化的矩阵乘法库
   - 所有深度学习框架（PyTorch/TensorFlow）内部都用这个技巧

【前置知识】
第2章 - 神经网络基础
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.figsize"] = (10, 6)
np.random.seed(42)  # 固定随机种子，保证结果可复现


# ════════════════════════════════════════════════════════════════════
# 第1部分：卷积直觉 —— 滤波器在图像上滑动
# ════════════════════════════════════════════════════════════════════
#
# 卷积的核心操作非常简单：
#   1. 一个小矩阵（卷积核/滤波器）在大矩阵（图像）上滑动
#   2. 每到一个位置，做逐元素乘法然后求和 → 得到一个输出值
#   3. 所有位置的输出值组成"特征图"（feature map）
#
# 类比：拿着放大镜逐格扫描图像，每次只看放大镜大小的区域
#

print("=" * 60)
print("第1部分：卷积直觉 —— 滤波器滑动演示")
print("=" * 60)

# 创建一个 5×5 的小图像（灰度）
image_5x5 = np.array([
    [1, 2, 0, 1, 3],
    [0, 1, 3, 2, 1],
    [2, 0, 1, 0, 2],
    [1, 3, 2, 1, 0],
    [0, 1, 0, 3, 1]
], dtype=float)

# 一个 3×3 边缘检测卷积核
kernel_3x3 = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=float)

# 可视化卷积滑动过程（展示前4步）
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

step = 0
h_out = image_5x5.shape[0] - kernel_3x3.shape[0] + 1  # 输出高度 = 3
w_out = image_5x5.shape[1] - kernel_3x3.shape[1] + 1  # 输出宽度 = 3
output = np.zeros((h_out, w_out))

for i in range(h_out):
    for j in range(w_out):
        # 提取当前覆盖的区域
        region = image_5x5[i:i+3, j:j+3]
        # 逐元素乘法 + 求和 = 卷积输出
        value = np.sum(region * kernel_3x3)
        output[i, j] = value

        # 只画前4步的详细过程
        if step < 4:
            row, col = step // 4, step % 4

            # 上面一行：图像 + 卷积核覆盖区域高亮
            ax_img = axes[0, step]
            ax_img.imshow(image_5x5, cmap="Blues", vmin=0, vmax=3)
            # 画红色方框标记卷积核位置
            rect = patches.Rectangle((j - 0.5, i - 0.5), 3, 3,
                                     linewidth=2.5, edgecolor="red",
                                     facecolor="red", alpha=0.15)
            ax_img.add_patch(rect)
            # 在每个格子里写数值
            for ii in range(5):
                for jj in range(5):
                    ax_img.text(jj, ii, f"{image_5x5[ii, jj]:.0f}",
                                ha="center", va="center", fontsize=10)
            ax_img.set_title(f"步骤 {step+1}: 位置({i},{j})", fontsize=11)
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            # 下面一行：展示点积计算过程
            ax_calc = axes[1, step]
            ax_calc.axis("off")
            # 构造计算说明文字
            calc_parts = []
            for ki in range(3):
                for kj in range(3):
                    calc_parts.append(
                        f"{region[ki, kj]:.0f}*{kernel_3x3[ki, kj]:+.0f}"
                    )
            calc_str = " + ".join(calc_parts[:3]) + "\n"
            calc_str += "+ " + " + ".join(calc_parts[3:6]) + "\n"
            calc_str += "+ " + " + ".join(calc_parts[6:9])
            calc_str += f"\n= {value:.0f}"
            ax_calc.text(0.5, 0.5, calc_str, ha="center", va="center",
                         fontsize=9, family="monospace",
                         bbox=dict(boxstyle="round", facecolor="lightyellow"))

        step += 1

plt.suptitle("卷积滑动过程：3x3 核在 5x5 图像上滑动（前4步）",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"输入图像尺寸: 5x5")
print(f"卷积核尺寸:   3x3")
print(f"输出特征图尺寸: {h_out}x{w_out}")
print(f"公式验证: (5 - 3 + 2*0) / 1 + 1 = 3  ✓")
print(f"\n输出特征图:\n{output}")


# ════════════════════════════════════════════════════════════════════
# 第2部分：手写卷积 —— 朴素 for 循环实现
# ════════════════════════════════════════════════════════════════════
#
# 先写最直观的版本：四层 for 循环
# 虽然慢，但逻辑一目了然

print("\n" + "=" * 60)
print("第2部分：手写卷积（朴素 for 循环实现）")
print("=" * 60)


def conv2d_naive(image, kernel, stride=1, padding=0):
    """
    朴素的 2D 卷积实现（四重 for 循环）。

    参数:
        image   : 输入图像, 形状 (H, W)
        kernel  : 卷积核, 形状 (kH, kW)
        stride  : 步长, 默认 1
        padding : 零填充大小, 默认 0

    返回:
        output  : 卷积输出, 形状 (H_out, W_out)
    """
    # 第一步：如果需要，在图像四周补零
    if padding > 0:
        image = np.pad(image, padding, mode="constant", constant_values=0)

    H, W = image.shape
    kH, kW = kernel.shape

    # 第二步：计算输出尺寸（背下这个公式！）
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # 第三步：滑动卷积核，逐位置计算
    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            # 当前卷积核覆盖的区域
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start + kH, w_start:w_start + kW]
            # 逐元素相乘再求和
            output[i, j] = np.sum(region * kernel)

    return output


# 验证：和第1部分手算结果对比
result_naive = conv2d_naive(image_5x5, kernel_3x3)
print(f"朴素卷积输出:\n{result_naive}")
print(f"与手算结果一致: {np.allclose(result_naive, output)}")


# ════════════════════════════════════════════════════════════════════
# 第3部分：边缘检测 —— 用 Sobel 核检测图像边缘
# ════════════════════════════════════════════════════════════════════
#
# Sobel 算子是经典的边缘检测滤波器：
#   水平方向 Sobel：检测垂直边缘（左右像素差异大的地方）
#   垂直方向 Sobel：检测水平边缘（上下像素差异大的地方）
#
# 这就是 CNN 浅层自动学到的东西！

print("\n" + "=" * 60)
print("第3部分：边缘检测（Sobel 核）")
print("=" * 60)

# 创建一张合成图像：左半黑右半白 + 中心一个白色方块
test_image = np.zeros((32, 32))
test_image[:, 16:] = 1.0           # 右半部分为白色
test_image[10:22, 10:22] = 1.0     # 中心白色方块

# Sobel 算子
sobel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=float)  # 检测垂直边缘

sobel_y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=float)  # 检测水平边缘

# 用手写卷积做边缘检测
edges_x = conv2d_naive(test_image, sobel_x)
edges_y = conv2d_naive(test_image, sobel_y)
edges_combined = np.sqrt(edges_x ** 2 + edges_y ** 2)  # 边缘强度

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ["原始图像", "垂直边缘 (Sobel X)", "水平边缘 (Sobel Y)", "边缘强度 (合并)"]
images = [test_image, edges_x, edges_y, edges_combined]
cmaps = ["gray", "RdBu_r", "RdBu_r", "hot"]

for ax, img, title, cmap in zip(axes, images, titles, cmaps):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Sobel 边缘检测：CNN 浅层自动学到的就是这些！",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("Sobel X 核（检测垂直边缘）:")
print(sobel_x)
print("\nSobel Y 核（检测水平边缘）:")
print(sobel_y)
print("\n关键观察：不同的卷积核 = 不同的特征检测器")
print("  CNN 的训练过程就是自动学习最有用的卷积核！")


# ════════════════════════════════════════════════════════════════════
# 第4部分：步长和填充 —— 控制输出尺寸
# ════════════════════════════════════════════════════════════════════
#
# 输出尺寸公式（务必记住！）:
#   H_out = (H_in - kH + 2*padding) / stride + 1
#
# 常见配置：
#   - kernel=3, stride=1, padding=1 → 输出尺寸不变（"same"卷积）
#   - kernel=3, stride=2, padding=1 → 输出尺寸减半（下采样）

print("\n" + "=" * 60)
print("第4部分：步长和填充对输出尺寸的影响")
print("=" * 60)


def compute_output_size(h_in, kernel_size, stride, padding):
    """计算卷积输出的空间尺寸"""
    return (h_in - kernel_size + 2 * padding) // stride + 1


# 创建一个 8×8 的测试图像
img_8x8 = np.random.rand(8, 8)
kernel = np.ones((3, 3)) / 9.0  # 均值滤波核

# 测试不同的步长和填充组合
configs = [
    {"stride": 1, "padding": 0, "desc": "stride=1, pad=0 → 尺寸缩小"},
    {"stride": 1, "padding": 1, "desc": "stride=1, pad=1 → 尺寸不变 (same)"},
    {"stride": 2, "padding": 0, "desc": "stride=2, pad=0 → 尺寸大幅缩小"},
    {"stride": 2, "padding": 1, "desc": "stride=2, pad=1 → 尺寸减半"},
]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, cfg in enumerate(configs):
    s, p = cfg["stride"], cfg["padding"]
    out = conv2d_naive(img_8x8, kernel, stride=s, padding=p)
    out_size = compute_output_size(8, 3, s, p)

    axes[idx].imshow(out, cmap="viridis")
    axes[idx].set_title(f"{cfg['desc']}\n输出: {out.shape[0]}x{out.shape[1]}", fontsize=10)
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

    print(f"  {cfg['desc']}")
    print(f"    公式: ({8} - {3} + 2*{p}) / {s} + 1 = {out_size}")
    print(f"    实际输出: {out.shape}")

plt.suptitle("步长(Stride)和填充(Padding)对输出尺寸的影响",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n记忆技巧:")
print("  - padding=kernel//2, stride=1 → 输出尺寸 = 输入尺寸（same 卷积）")
print("  - stride=2 → 输出尺寸约为输入的一半（常用于下采样）")


# ════════════════════════════════════════════════════════════════════
# 第5部分：im2col —— 将卷积转换为矩阵乘法
# ════════════════════════════════════════════════════════════════════
#
# 朴素 for 循环太慢了！框架内部用 im2col 技巧：
#   1. 把每个卷积核覆盖的区域展平成一行 → 得到一个大矩阵
#   2. 卷积核也展平成一列
#   3. 矩阵乘法一步完成所有卷积操作
#
# 代价是内存增加（数据被复制了），但换来了 BLAS 矩阵乘法的极致速度

print("\n" + "=" * 60)
print("第5部分：im2col —— 卷积变矩阵乘法")
print("=" * 60)


def im2col(image, kernel_size, stride=1, padding=0):
    """
    将图像转换为列矩阵，使卷积可以用矩阵乘法实现。

    参数:
        image       : 输入图像, 形状 (H, W)
        kernel_size : 卷积核大小（正方形）
        stride      : 步长
        padding     : 零填充

    返回:
        col : 形状 (H_out*W_out, kH*kW) 的矩阵
              每一行对应一个卷积窗口展平后的向量
    """
    if padding > 0:
        image = np.pad(image, padding, mode="constant", constant_values=0)

    H, W = image.shape
    kH = kW = kernel_size
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # 每一行 = 一个窗口展平; 共 H_out * W_out 行
    col = np.zeros((H_out * W_out, kH * kW))

    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            patch = image[h_start:h_start + kH, w_start:w_start + kW]
            col[idx] = patch.flatten()  # 展平为一维向量
            idx += 1

    return col


def conv2d_im2col(image, kernel, stride=1, padding=0):
    """
    用 im2col + 矩阵乘法实现卷积。

    参数/返回同 conv2d_naive
    """
    kH, kW = kernel.shape
    if padding > 0:
        image_padded = np.pad(image, padding, mode="constant", constant_values=0)
    else:
        image_padded = image

    H, W = image_padded.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    # 第一步：im2col 展开
    col = im2col(image, kH, stride, padding)  # (H_out*W_out, kH*kW)

    # 第二步：卷积核展平为列向量
    kernel_flat = kernel.flatten()  # (kH*kW,)

    # 第三步：矩阵乘法！一行代码完成所有卷积
    output_flat = col @ kernel_flat  # (H_out*W_out,)

    # 第四步：reshape 回二维
    output = output_flat.reshape(H_out, W_out)
    return output


# 验证 im2col 结果与朴素实现一致
result_im2col = conv2d_im2col(image_5x5, kernel_3x3)
print(f"im2col 卷积输出:\n{result_im2col}")
print(f"与朴素实现一致: {np.allclose(result_im2col, result_naive)}")

# 可视化 im2col 的展开过程
col_matrix = im2col(image_5x5, 3)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 左：原始图像
axes[0].imshow(image_5x5, cmap="Blues")
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f"{image_5x5[i, j]:.0f}",
                     ha="center", va="center", fontsize=12)
axes[0].set_title("原始图像 (5x5)", fontsize=11)
axes[0].set_xticks([])
axes[0].set_yticks([])

# 中：im2col 展开的矩阵
axes[1].imshow(col_matrix, cmap="Blues", aspect="auto")
axes[1].set_title(f"im2col 矩阵 ({col_matrix.shape[0]}x{col_matrix.shape[1]})\n"
                  f"每行 = 一个 3x3 窗口展平", fontsize=11)
axes[1].set_ylabel("窗口索引")
axes[1].set_xlabel("像素位置")
for i in range(col_matrix.shape[0]):
    for j in range(col_matrix.shape[1]):
        axes[1].text(j, i, f"{col_matrix[i, j]:.0f}",
                     ha="center", va="center", fontsize=8)

# 右：核展平
kernel_flat = kernel_3x3.flatten().reshape(-1, 1)
axes[2].imshow(kernel_flat, cmap="RdBu_r", aspect="auto")
for i in range(len(kernel_flat)):
    axes[2].text(0, i, f"{kernel_flat[i, 0]:+.0f}",
                 ha="center", va="center", fontsize=10)
axes[2].set_title("核展平 (9x1)\n矩阵乘法得到输出", fontsize=11)
axes[2].set_xticks([])

plt.suptitle("im2col 技巧：把卷积变成矩阵乘法",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"\nim2col 展开: 图像(5x5) + 核(3x3) → 矩阵({col_matrix.shape[0]}x{col_matrix.shape[1]})")
print(f"矩阵乘法: ({col_matrix.shape[0]}x{col_matrix.shape[1]}) @ ({col_matrix.shape[1]}x1)")
print(f"         = ({col_matrix.shape[0]}x1) → reshape → (3x3) 输出特征图")
print(f"\n虽然内存占用从 25 增加到 {col_matrix.size}，但矩阵乘法快得多！")


# ════════════════════════════════════════════════════════════════════
# 第6部分：池化实现 —— 最大池化与平均池化
# ════════════════════════════════════════════════════════════════════
#
# 池化是 CNN 的另一个核心操作：
#   - 不涉及学习参数（与卷积不同）
#   - 在每个局部窗口内取最大值（Max）或平均值（Avg）
#   - 作用：降低分辨率、减少计算量、增大感受野、增加平移不变性

print("\n" + "=" * 60)
print("第6部分：池化（Max Pooling & Average Pooling）")
print("=" * 60)


def max_pool2d(image, pool_size=2, stride=2):
    """
    最大池化：在每个窗口中取最大值。

    参数:
        image     : 输入, 形状 (H, W)
        pool_size : 池化窗口大小（正方形）
        stride    : 步长, 通常等于 pool_size

    返回:
        output : 池化后的输出
    """
    H, W = image.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start + pool_size,
                           w_start:w_start + pool_size]
            output[i, j] = np.max(region)  # 取最大值

    return output


def avg_pool2d(image, pool_size=2, stride=2):
    """
    平均池化：在每个窗口中取平均值。

    参数/返回同 max_pool2d
    """
    H, W = image.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            region = image[h_start:h_start + pool_size,
                           w_start:w_start + pool_size]
            output[i, j] = np.mean(region)  # 取平均值

    return output


# 创建一个 6×6 的测试图像
pool_test = np.array([
    [1, 3, 2, 4, 1, 0],
    [0, 5, 1, 2, 3, 1],
    [2, 1, 4, 0, 2, 3],
    [3, 0, 2, 5, 1, 2],
    [1, 4, 0, 3, 2, 1],
    [2, 1, 3, 1, 0, 4]
], dtype=float)

max_out = max_pool2d(pool_test, pool_size=2, stride=2)
avg_out = avg_pool2d(pool_test, pool_size=2, stride=2)

# 可视化池化过程
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 原始图像
axes[0].imshow(pool_test, cmap="YlOrRd", vmin=0, vmax=5)
for i in range(6):
    for j in range(6):
        axes[0].text(j, i, f"{pool_test[i, j]:.0f}",
                     ha="center", va="center", fontsize=11, fontweight="bold")
# 画 2x2 网格线
for k in range(0, 7, 2):
    axes[0].axhline(k - 0.5, color="black", linewidth=2)
    axes[0].axvline(k - 0.5, color="black", linewidth=2)
axes[0].set_title("原始图像 (6x6)\n黑线划分池化窗口", fontsize=11)
axes[0].set_xticks([])
axes[0].set_yticks([])

# 最大池化结果
axes[1].imshow(max_out, cmap="YlOrRd", vmin=0, vmax=5)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{max_out[i, j]:.0f}",
                     ha="center", va="center", fontsize=14, fontweight="bold")
axes[1].set_title("最大池化 (2x2, stride=2)\n每个窗口取最大值", fontsize=11)
axes[1].set_xticks([])
axes[1].set_yticks([])

# 平均池化结果
axes[2].imshow(avg_out, cmap="YlOrRd", vmin=0, vmax=5)
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f"{avg_out[i, j]:.1f}",
                     ha="center", va="center", fontsize=14, fontweight="bold")
axes[2].set_title("平均池化 (2x2, stride=2)\n每个窗口取平均", fontsize=11)
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.suptitle("池化操作：降低分辨率，保留关键信息",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"输入尺寸: {pool_test.shape} → 池化后: {max_out.shape}")
print(f"参数量: 0 （池化没有可学习参数！）")
print(f"\n最大池化输出:\n{max_out}")
print(f"\n平均池化输出:\n{avg_out}")
print(f"\n最大池化保留最强的激活（常用），平均池化保留整体趋势")


# ════════════════════════════════════════════════════════════════════
# 第7部分：多通道卷积 —— 处理 RGB 彩色图像
# ════════════════════════════════════════════════════════════════════
#
# 真实图像通常是多通道的（RGB 3通道），卷积核也要相应扩展：
#   - 输入: (C_in, H, W)       例如 (3, 32, 32) 的 RGB 图像
#   - 一个卷积核: (C_in, kH, kW)  例如 (3, 3, 3)，深度必须匹配输入通道
#   - N 个卷积核 → 输出 N 通道特征图: (N, H_out, W_out)
#
# 关键理解：一个卷积核在所有输入通道上做卷积，结果相加 → 一个输出通道

print("\n" + "=" * 60)
print("第7部分：多通道卷积（RGB 输入）")
print("=" * 60)


def conv2d_multichannel(image, kernels, stride=1, padding=0):
    """
    多通道卷积（支持多个卷积核，输出多通道特征图）。

    参数:
        image   : 输入图像, 形状 (C_in, H, W)
        kernels : 卷积核, 形状 (C_out, C_in, kH, kW)
                  C_out 个卷积核，每个有 C_in 个通道
        stride  : 步长
        padding : 零填充

    返回:
        output  : 形状 (C_out, H_out, W_out) 的多通道特征图
    """
    C_out, C_in, kH, kW = kernels.shape
    _, H, W = image.shape

    # 对每个通道进行填充
    if padding > 0:
        image_padded = np.pad(image,
                              ((0, 0), (padding, padding), (padding, padding)),
                              mode="constant", constant_values=0)
    else:
        image_padded = image

    H_pad, W_pad = image_padded.shape[1], image_padded.shape[2]
    H_out = (H_pad - kH) // stride + 1
    W_out = (W_pad - kW) // stride + 1

    output = np.zeros((C_out, H_out, W_out))

    for n in range(C_out):           # 遍历每个输出通道（每个卷积核）
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                # 在所有输入通道上做卷积并求和
                region = image_padded[:, h_start:h_start + kH,
                                         w_start:w_start + kW]
                output[n, i, j] = np.sum(region * kernels[n])

    return output


# 创建一个 3 通道 (RGB) 8x8 合成图像
rgb_image = np.zeros((3, 8, 8))
rgb_image[0, 2:6, 2:6] = 1.0   # 红色通道中心有个方块
rgb_image[1, :, 4:] = 0.7      # 绿色通道右半部分
rgb_image[2, 4:, :] = 0.5      # 蓝色通道下半部分

# 定义 2 个卷积核: 形状 (2, 3, 3, 3) → 2个输出通道, 3个输入通道, 3x3
conv_kernels = np.random.randn(2, 3, 3, 3) * 0.5
# 第一个核：倾向于检测红色通道的特征
conv_kernels[0, 0] = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 红通道边缘
conv_kernels[0, 1] *= 0.1  # 弱化绿通道
conv_kernels[0, 2] *= 0.1  # 弱化蓝通道

# 执行多通道卷积
multi_output = conv2d_multichannel(rgb_image, conv_kernels, stride=1, padding=1)

# 可视化
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 上排：输入 RGB 通道 + 合成图
channel_names = ["R 通道", "G 通道", "B 通道"]
for c in range(3):
    axes[0, c].imshow(rgb_image[c], cmap="gray", vmin=0, vmax=1)
    axes[0, c].set_title(f"输入: {channel_names[c]}", fontsize=11)
    axes[0, c].set_xticks([])
    axes[0, c].set_yticks([])

# 合成 RGB 图像
rgb_display = np.transpose(rgb_image, (1, 2, 0))  # (H, W, C)
axes[0, 3].imshow(rgb_display)
axes[0, 3].set_title("输入: RGB 合成", fontsize=11)
axes[0, 3].set_xticks([])
axes[0, 3].set_yticks([])

# 下排：输出特征图
for n in range(2):
    axes[1, n].imshow(multi_output[n], cmap="RdBu_r")
    axes[1, n].set_title(f"输出: 特征图 {n+1}", fontsize=11)
    axes[1, n].set_xticks([])
    axes[1, n].set_yticks([])

# 隐藏多余的子图
axes[1, 2].axis("off")
axes[1, 3].axis("off")

# 在空白区域写参数统计
info_text = (
    f"输入形状: (3, 8, 8)\n"
    f"卷积核形状: (2, 3, 3, 3)\n"
    f"输出形状: {multi_output.shape}\n\n"
    f"参数量: 2 * 3 * 3 * 3 = {2*3*3*3}\n"
    f"(不含偏置)"
)
axes[1, 2].text(0.1, 0.5, info_text, fontsize=12, va="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))
axes[1, 2].axis("off")

# 与全连接对比
fc_params = 3 * 8 * 8 * 2 * 8 * 8  # 如果用全连接：输入*输出
axes[1, 3].text(0.1, 0.5,
                f"如果用全连接层:\n"
                f"参数量 = {3*8*8} * {2*8*8}\n"
                f"       = {fc_params}\n\n"
                f"卷积参数量 = {2*3*3*3}\n"
                f"节省了 {fc_params // (2*3*3*3)}x !",
                fontsize=12, va="center", family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightcyan"))
axes[1, 3].axis("off")

plt.suptitle("多通道卷积：3通道输入, 2个卷积核 → 2通道输出",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print(f"输入:  {rgb_image.shape}  (C_in=3, H=8, W=8)")
print(f"卷积核: {conv_kernels.shape}  (C_out=2, C_in=3, kH=3, kW=3)")
print(f"输出:  {multi_output.shape}  (C_out=2, H_out=8, W_out=8)")
print(f"\n参数量对比:")
print(f"  卷积层: {2 * 3 * 3 * 3} 个参数（+偏置 2 个 = {2*3*3*3 + 2} 个）")
print(f"  全连接: {3*8*8} × {2*8*8} = {fc_params} 个参数")
print(f"  卷积节省了 {fc_params // (2*3*3*3)}x 的参数！这就是 CNN 的威力。")


# ════════════════════════════════════════════════════════════════════
# 第8部分：完整总结与思考题
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了 CNN 的两个核心操作：

  1. 卷积操作:    滤波器滑动 → 局部连接 + 参数共享 → 高效提取特征
  2. 关键超参数:  卷积核大小、步长、填充 → 控制输出尺寸
  3. 边缘检测:    Sobel 算子是最经典的手工滤波器，CNN 自动学习更好的
  4. im2col:      把卷积转成矩阵乘法，牺牲内存换取速度
  5. 池化:        降分辨率、减参数、增大感受野（无可学习参数）
  6. 多通道卷积:  RGB 3通道 → N 个卷积核 → N 通道特征图
  7. 参数效率:    卷积比全连接少几个数量级的参数

下一节将把这些积木组合成完整的 CNN 网络（LeNet/VGG/ResNet）！
""")

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【感受野计算】
   两层 3x3 卷积（stride=1, padding=1）堆叠起来，
   输出的每个像素"看到"了输入的多大区域？
   那和一层 5x5 卷积相比呢？为什么 VGG 选择堆叠 3x3？
   提示：计算参数量 —— 2*(3*3) vs 1*(5*5)

2. 【1x1 卷积的作用】
   如果卷积核大小是 1x1，它还有意义吗？
   在多通道情况下，1x1 卷积实际上在做什么操作？
   提示：1x1 卷积 = 逐像素的全连接层 = 通道间的线性组合
   这就是 Network in Network 和 ResNet 中大量使用的 "bottleneck" 结构。

3. 【池化 vs 步长卷积】
   现代 CNN（如 ResNet）越来越少用最大池化，
   而改用 stride=2 的卷积来降低分辨率。两者有什么区别？
   提示：stride=2 卷积有可学习参数，池化没有。
   哪个更"灵活"？哪个更"稳定"？

4. 【im2col 的内存代价】
   对于输入 (3, 224, 224)、卷积核 (64, 3, 3, 3)、stride=1、padding=1，
   im2col 展开后的矩阵尺寸是多少？比原图大了多少倍？
   提示：每个窗口展平后长度 = 3*3*3 = 27，窗口总数 = 224*224

5. 【转置卷积（反卷积）】
   卷积通常会缩小空间尺寸。如果我们想"放大"特征图
   （比如在图像分割中），应该怎么做？
   提示：转置卷积（Transposed Convolution）本质上是卷积的"逆操作"，
   它也可以用 im2col 的逆过程 (col2im) 来理解。
""")

print("下一节预告: 第3章 · 第2节 · 经典 CNN 架构（LeNet → VGG → ResNet）")
