"""
==============================================================
第3章 第1节：卷积直觉 —— 用手电筒扫描图像
==============================================================

【为什么需要它？】
问题1：全连接层处理图像的问题
  一张 224×224 彩色图片 = 224*224*3 = 150,528 个像素
  如果用全连接层，第一层1000个神经元 = 1.5亿个参数！
  而且每个参数是独立的，根本学不到图片的空间结构。

问题2：平移不变性
  猫在图片左上角 vs 右下角，是同一只猫。
  全连接层每次都要重新学！太浪费了。

卷积的解决方案：
  1. 参数共享：同一个卷积核扫描整张图片，参数数量大幅减少
  2. 局部感受野：每个神经元只看一小块图像（符合图像的局部结构）
  3. 平移不变性：卷积核移到哪，都在做同一种特征检测

【生活类比】
用一个手电筒扫描一张照片：
  手电筒形状 = 卷积核（滤波器）
  手电筒能照到的区域 = 感受野
  手电筒的"模板"决定检测什么特征：
    斜线模板 → 检测斜线
    边缘模板 → 检测边缘
    圆形模板 → 检测圆形

【存在理由】
解决问题：全连接层处理图像参数爆炸，且不具备平移不变性
核心思想：共享参数的局部卷积操作，每个核检测一种特征
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.random.seed(42)

# ============================================================
# Part 1: 全连接 vs 卷积的参数数量对比
# ============================================================
print("=" * 50)
print("Part 1: 全连接 vs 卷积 —— 参数爆炸问题")
print("=" * 50)

image_sizes = [(28, 28), (64, 64), (224, 224)]
n_channels = 3  # RGB
n_filters = 64
filter_size = 3

print(f"\n{'图像尺寸':<15} {'全连接参数':<20} {'卷积参数':<15} {'压缩比':<10}")
print("-" * 60)
for h, w in image_sizes:
    input_size = h * w * n_channels
    fc_params = input_size * 1000  # 假设第一层1000个神经元
    conv_params = filter_size * filter_size * n_channels * n_filters
    ratio = fc_params / conv_params
    print(f"{h}x{w}x{n_channels:<9} {fc_params:<20,} {conv_params:<15,} {ratio:<10.0f}x")

print(f"\n卷积层只有 {filter_size}x{filter_size}x{n_channels}x{n_filters} = {filter_size**2*n_channels*n_filters} 个参数")
print("全连接层随图像尺寸增大，参数爆炸！")
print("卷积层参数数量与图像尺寸无关（参数共享！）")

# ============================================================
# Part 2: 卷积操作的直觉
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 卷积操作 —— 手电筒扫描")
print("=" * 50)

"""
卷积操作：
  将卷积核（filter）在图像上滑动
  每个位置：卷积核与覆盖区域做对应元素相乘再求和

输出尺寸公式：
  output_h = (input_h + 2*padding - kernel_h) / stride + 1
  output_w = (input_w + 2*padding - kernel_w) / stride + 1

概念：
  stride（步长）：每次滑动多少格（越大，输出越小）
  padding（填充）：在图像边缘填0（保持输出尺寸不变，或其他用途）
"""

def conv2d_naive(image, kernel, stride=1, pad=0):
    """
    朴素卷积：双重 for 循环版本（清晰直观）
    image: (H, W)
    kernel: (kH, kW)
    返回: 卷积结果
    """
    if pad > 0:
        image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')

    H, W = image.shape
    kH, kW = kernel.shape
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # 提取感受野区域
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW]
            # 逐元素相乘求和（卷积核与感受野的点积）
            output[i, j] = np.sum(patch * kernel)

    return output

# ============================================================
# Part 3: 已知卷积核的可视化 —— 卷积核是特征检测器
# ============================================================
print("Part 3: 卷积核作为特征检测器")
print("=" * 50)

"""
卷积神经网络中，卷积核是"学习"出来的。
但我们可以先用已知的卷积核，看看卷积在"检测什么"。
"""

# 生成测试图像（一个简单的几何图形）
def make_test_image(size=32):
    img = np.zeros((size, size))
    # 画一个矩形
    img[8:24, 8:24] = 0.8
    # 画一条对角线
    for i in range(size):
        img[i, i] = 1.0
    # 画一个圆
    cx, cy, r = 16, 8, 5
    for i in range(size):
        for j in range(size):
            if (i-cx)**2 + (j-cy)**2 < r**2:
                img[i, j] = 0.6
    return img

test_image = make_test_image(32)

# 经典卷积核
kernels = {
    '均值模糊（平均池化效果）': np.ones((3, 3)) / 9,

    '垂直边缘检测（Sobel-x）': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]) / 4.0,

    '水平边缘检测（Sobel-y）': np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ]) / 4.0,

    '对角线检测': np.array([
        [ 2, -1, -1],
        [-1,  2, -1],
        [-1, -1,  2]
    ]) / 3.0,

    '锐化': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]),

    '高斯模糊': np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0,
}

n_kernels = len(kernels)
fig, axes = plt.subplots(2, n_kernels + 1, figsize=(22, 7))
fig.suptitle('不同卷积核的效果：卷积核 = 特征检测器', fontsize=13)

# 原始图像
axes[0][0].imshow(test_image, cmap='gray', vmin=0, vmax=1)
axes[0][0].set_title('原始图像', fontsize=10)
axes[0][0].axis('off')
axes[1][0].text(0.5, 0.5, '手动设计的\n卷积核\n（CNN训练\n时自动学习）',
               ha='center', va='center', fontsize=9, transform=axes[1][0].transAxes)
axes[1][0].axis('off')

for idx, (name, kernel) in enumerate(kernels.items()):
    col = idx + 1
    # 卷积结果
    result = conv2d_naive(test_image, kernel, pad=1)
    result_display = np.clip(result, 0, 1)

    axes[0][col].imshow(result_display, cmap='gray', vmin=0, vmax=1)
    axes[0][col].set_title(name, fontsize=8)
    axes[0][col].axis('off')

    # 显示卷积核
    im = axes[1][col].imshow(kernel, cmap='RdBu_r',
                              vmin=kernel.min(), vmax=kernel.max())
    axes[1][col].set_title(f'核（{kernel.shape[0]}×{kernel.shape[1]}）', fontsize=8)
    # 在每个格子里写数字
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            axes[1][col].text(j, i, f'{kernel[i,j]:.2f}',
                            ha='center', va='center', fontsize=7)
    axes[1][col].axis('off')

plt.tight_layout()
plt.savefig('03_cnn/convolution_viz.png', dpi=80, bbox_inches='tight')
print("图片已保存：03_cnn/convolution_viz.png")
plt.show()

# ============================================================
# Part 4: 动画 —— 卷积核在图像上滑动
# ============================================================
print("\nPart 4: 步长和填充的效果")
print("=" * 50)

def compute_output_size(input_size, kernel_size, stride, pad):
    return (input_size + 2*pad - kernel_size) // stride + 1

print("输出尺寸计算（图像=28×28，卷积核=3×3）：")
print(f"{'stride':>8} {'padding':>10} {'输出尺寸':>12} {'说明':>20}")
print("-" * 55)
configs = [(1, 0, '无填充，步长1（缩小）'),
           (1, 1, '有填充，步长1（保持尺寸）'),
           (2, 0, '无填充，步长2（缩小一半）'),
           (2, 1, '有填充，步长2（缩小约一半）')]
for s, p, desc in configs:
    out = compute_output_size(28, 3, s, p)
    print(f"{s:>8} {p:>10} {out:>12}×{out:<12} {desc}")

# ============================================================
# Part 5: 多通道卷积 —— RGB图像的处理
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 多通道卷积 —— RGB图像")
print("=" * 50)

"""
彩色图像有3个通道（R、G、B）。
多通道卷积：卷积核也有 C_in 个通道（每个通道一个 kH×kW 的核）
  输出的一个位置 = Σ_c (输入通道c × 核通道c) 的和

多个卷积核 → 多个输出通道
  n_filters 个卷积核 → 输出有 n_filters 个特征图

整体参数量：n_filters × C_in × kH × kW
"""

def conv2d_multichannel(X, W, stride=1, pad=0):
    """
    多通道卷积
    X: (C_in, H, W)
    W: (C_out, C_in, kH, kW)
    输出: (C_out, out_H, out_W)
    """
    C_out, C_in, kH, kW = W.shape
    C_in_x, H, W_size = X.shape

    if pad > 0:
        X = np.pad(X, ((0,0), (pad,pad), (pad,pad)), mode='constant')

    out_H = (H + 2*pad - kH) // stride + 1
    out_W = (W_size + 2*pad - kW) // stride + 1
    output = np.zeros((C_out, out_H, out_W))

    for f in range(C_out):        # 每个卷积核
        for i in range(out_H):
            for j in range(out_W):
                # 感受野：C_in × kH × kW
                patch = X[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
                output[f, i, j] = np.sum(patch * W[f])  # 点积

    return output

# 演示
C_in, H, W_size = 3, 8, 8   # RGB 8×8 图像
C_out = 4                    # 4个卷积核
kH = kW = 3

X = np.random.randn(C_in, H, W_size)
W = np.random.randn(C_out, C_in, kH, kW) * 0.1

output = conv2d_multichannel(X, W, stride=1, pad=1)
print(f"输入 X: shape = {X.shape}  （3个通道，8×8）")
print(f"权重 W: shape = {W.shape}  （4个核，每核3×3×3）")
print(f"输出:   shape = {output.shape}  （4个特征图，8×8）")
print(f"参数数量：{W.size}  （= 4×3×3×3）")
print(f"对比全连接：{C_in*H*W_size * C_out*output.shape[1]*output.shape[2]}  （=全部输入输出连接）")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【输出尺寸计算】
   输入图像：32×32×3（CIFAR-10尺寸）
   经过以下操作后，输出尺寸是多少？
   操作1：Conv(32 filters, 3×3, stride=1, pad=1)
   操作2：MaxPool(2×2, stride=2)
   操作3：Conv(64 filters, 3×3, stride=1, pad=1)
   操作4：MaxPool(2×2, stride=2)

2. 【自制特征检测器】
   设计一个能检测"左上到右下对角线"的 5×5 卷积核（手写数字3×3没有）。
   在 make_test_image() 的结果上应用它，画出结果。
   斜线是否被高亮了？

3. 【感受野计算】
   经过以下网络后，最终输出的每个像素"看到"了原始图像多大区域？
   Conv(3×3, stride=1) → Conv(3×3, stride=1) → Conv(3×3, stride=1)
   （提示：第一层感受野=3×3，每加一层扩大2）
   感受野大小 = 理解网络"看到多远"的关键。
""")
