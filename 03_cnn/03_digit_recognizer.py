"""
==============================================================
第3章 第3节：数字识别实战 —— 识别手写数字 1-6
==============================================================

【为什么需要它？】
学了一堆理论，现在来做一个真实的端到端项目！
从数据生成 → 模型构建 → 训练 → 评估 → 可视化卷积核

这个文件整合了前两章的所有知识：
  - 第0章：数学（卷积、梯度、链式法则）
  - 第2章：神经网络（反向传播、优化器）
  - 第3章：卷积层、池化层

纯 numpy 实现，不用 PyTorch/TensorFlow！

【存在理由】
解决问题：如何让计算机"看懂"手写数字？
核心思想：用卷积提取局部特征，用全连接做分类
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 生成合成数字数据集（1-6）
# ============================================================
print("=" * 50)
print("Part 1: 生成手写数字数据集（1-6）")
print("=" * 50)

"""
因为我们不用外部库，自己合成数字图像：
  - 每个数字用像素模板表示（16×16 的格子）
  - 加入高斯噪声模拟手写的不规则性
  - 随机旋转/平移增加多样性
"""

# 16×16 像素的数字模板（1=墨水，0=空白）
DIGIT_TEMPLATES = {}

def make_template(pixels, size=16):
    """将(row, col)像素点转为图像"""
    img = np.zeros((size, size))
    for r, c in pixels:
        if 0 <= r < size and 0 <= c < size:
            img[r, c] = 1.0
    return img

# 数字1：竖线
ones = [(r, 8) for r in range(3, 13)] + [(4, 7), (5, 6)]
DIGIT_TEMPLATES[1] = make_template(ones)

# 数字2：顶弧 + 对角 + 底横
twos = ([(3, c) for c in range(6, 11)] +  # 顶横
        [(4, 10), (5, 10), (6, 9)] +        # 右侧
        [(7, 8), (8, 7), (9, 6)] +          # 对角
        [(10, c) for c in range(6, 11)])    # 底横
DIGIT_TEMPLATES[2] = make_template(twos)

# 数字3：两个弯弧
threes = ([(3, c) for c in range(6, 11)] +
          [(4, 10), (5, 10), (6, 9)] +
          [(6, c) for c in range(7, 11)] +
          [(7, 10), (8, 10), (9, 10)] +
          [(10, c) for c in range(6, 11)])
DIGIT_TEMPLATES[3] = make_template(threes)

# 数字4：竖 + 横 + 竖
fours = ([(r, 10) for r in range(3, 13)] +   # 右竖
         [(r, 7) for r in range(3, 8)] +      # 左竖（上半）
         [(7, c) for c in range(7, 11)])       # 横
DIGIT_TEMPLATES[4] = make_template(fours)

# 数字5：顶横 + 左竖 + 中横 + 右竖 + 底横
fives = ([(3, c) for c in range(6, 11)] +    # 顶横
         [(r, 6) for r in range(3, 7)] +      # 左竖（上）
         [(6, c) for c in range(6, 11)] +     # 中横
         [(7, 10), (8, 10), (9, 10)] +        # 右竖（下）
         [(10, c) for c in range(6, 11)])     # 底横
DIGIT_TEMPLATES[5] = make_template(fives)

# 数字6：顶横 + 左竖 + 中横 + 右竖（下）+ 底横
sixes = ([(3, c) for c in range(6, 11)] +    # 顶横
         [(r, 6) for r in range(3, 11)] +     # 左竖（全）
         [(6, c) for c in range(6, 11)] +     # 中横
         [(7, 10), (8, 10), (9, 10)] +        # 右竖（下）
         [(10, c) for c in range(6, 11)])     # 底横
DIGIT_TEMPLATES[6] = make_template(sixes)

def generate_digit(digit, noise=0.15, size=16):
    """在模板基础上加噪声，模拟手写"""
    template = DIGIT_TEMPLATES[digit].copy()

    # 高斯噪声
    noisy = template + np.random.randn(size, size) * noise

    # 随机小幅平移（-1到1像素）
    shift_r = np.random.randint(-1, 2)
    shift_c = np.random.randint(-1, 2)
    noisy = np.roll(np.roll(noisy, shift_r, axis=0), shift_c, axis=1)

    # 裁剪到[0,1]
    noisy = np.clip(noisy, 0, 1)
    return noisy

def generate_dataset(n_per_class=200, size=16, noise=0.15):
    """生成完整数据集"""
    X_list, y_list = [], []
    for digit in range(1, 7):  # 1-6
        for _ in range(n_per_class):
            img = generate_digit(digit, noise=noise, size=size)
            X_list.append(img.reshape(1, size, size))  # (1, H, W) = (通道,高,宽)
            y_list.append(digit - 1)  # 类别 0-5

    X = np.array(X_list)  # (N, 1, 16, 16)
    y = np.array(y_list)  # (N,)

    # 打乱
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 8:1:1 分割
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    return (X[:n_train], y[:n_train],
            X[n_train:n_train+n_val], y[n_train:n_train+n_val],
            X[n_train+n_val:], y[n_train+n_val:])

X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset(n_per_class=200)
print(f"数据集生成：")
print(f"  训练集：{X_train.shape}，{len(y_train)} 个样本")
print(f"  验证集：{X_val.shape}，{len(y_val)} 个样本")
print(f"  测试集：{X_test.shape}，{len(y_test)} 个样本")
print(f"  类别分布：{[np.sum(y_train==c) for c in range(6)]}")

# 可视化部分数据
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
fig.suptitle('合成手写数字 1-6（每列5个样本的平均）', fontsize=12)
for digit_idx, digit in enumerate(range(1, 7)):
    samples = X_train[y_train == digit-1][:5]
    axes[0][digit_idx].imshow(np.mean(samples, axis=0).reshape(16, 16), cmap='gray')
    axes[0][digit_idx].set_title(f'数字{digit}（均值）', fontsize=9)
    axes[0][digit_idx].axis('off')

    axes[1][digit_idx].imshow(samples[0].reshape(16, 16), cmap='gray')
    axes[1][digit_idx].set_title(f'数字{digit}（单样本）', fontsize=9)
    axes[1][digit_idx].axis('off')

plt.tight_layout()
plt.savefig('03_cnn/digit_dataset.png', dpi=100, bbox_inches='tight')
print("图片已保存：03_cnn/digit_dataset.png")
plt.show()

# ============================================================
# Part 2: 构建 CNN（使用第2节实现的层）
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 构建 CNN 模型")
print("=" * 50)

"""
网络结构：
  输入：(batch, 1, 16, 16)

  Conv(8 filters, 3×3, pad=1) → ReLU → MaxPool(2×2)
  → (batch, 8, 8, 8)

  Conv(16 filters, 3×3, pad=1) → ReLU → MaxPool(2×2)
  → (batch, 16, 4, 4)

  Flatten → (batch, 256)

  FC(256 → 64) → ReLU
  → (batch, 64)

  FC(64 → 6) → Softmax
  → (batch, 6)
"""

def relu(x):
    return np.maximum(0, x)

def relu_backward(d_out, cache_x):
    return d_out * (cache_x > 0)

def softmax(x):
    shifted = x - x.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    n = len(y_true)
    return -np.mean(np.log(probs[np.arange(n), y_true] + 1e-8))

# --- 卷积层 ---
def conv_forward(X, W, b, stride=1, pad=0):
    """
    X: (N, C_in, H, W)
    W: (C_out, C_in, kH, kW)
    b: (C_out,)
    """
    N, C_in, H, W_size = X.shape
    C_out, _, kH, kW = W.shape

    if pad > 0:
        X_padded = np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    else:
        X_padded = X

    out_H = (H + 2*pad - kH) // stride + 1
    out_W = (W_size + 2*pad - kW) // stride + 1

    # im2col：将每个感受野展开为列向量，加速矩阵运算
    output = np.zeros((N, C_out, out_H, out_W))

    for n in range(N):
        for f in range(C_out):
            for i in range(out_H):
                for j in range(out_W):
                    patch = X_padded[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                    output[n, f, i, j] = np.sum(patch * W[f]) + b[f]

    return output, (X_padded, W, b, stride, pad, X.shape)

def conv_backward(d_out, cache):
    X_padded, W, b, stride, pad, orig_shape = cache
    N, C_in, H, W_size = orig_shape
    C_out, _, kH, kW = W.shape
    _, _, out_H, out_W = d_out.shape

    dX_padded = np.zeros_like(X_padded)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(C_out):
            for i in range(out_H):
                for j in range(out_W):
                    patch = X_padded[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
                    dW[f] += d_out[n, f, i, j] * patch
                    dX_padded[n, :, i*stride:i*stride+kH, j*stride:j*stride+kW] += d_out[n, f, i, j] * W[f]
            db[f] += np.sum(d_out[:, f, :, :])

    if pad > 0:
        dX = dX_padded[:, :, pad:-pad, pad:-pad]
    else:
        dX = dX_padded

    return dX, dW, db

def maxpool_forward(X, pool_size=2, stride=2):
    """最大池化"""
    N, C, H, W = X.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1

    output = np.zeros((N, C, out_H, out_W))
    mask = np.zeros_like(X, dtype=bool)

    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    patch = X[n, c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                    max_val = np.max(patch)
                    output[n, c, i, j] = max_val
                    # 记录最大值位置（反向传播用）
                    local_mask = (patch == max_val)
                    mask[n, c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size] |= local_mask

    return output, (mask, X.shape, pool_size, stride)

def maxpool_backward(d_out, cache):
    """最大池化反向：只有最大值位置获得梯度"""
    mask, input_shape, pool_size, stride = cache
    N, C, H, W = input_shape
    _, _, out_H, out_W = d_out.shape

    dX = np.zeros(input_shape)
    for n in range(N):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    dX[n, c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size] += \
                        d_out[n, c, i, j] * mask[n, c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
    return dX

class DigitCNN:
    def __init__(self):
        # Conv1：1→8 通道，3×3
        self.W1 = np.random.randn(8, 1, 3, 3) * np.sqrt(2.0 / (1*3*3))
        self.b1 = np.zeros(8)

        # Conv2：8→16 通道，3×3
        self.W2 = np.random.randn(16, 8, 3, 3) * np.sqrt(2.0 / (8*3*3))
        self.b2 = np.zeros(16)

        # FC1：256→64
        self.W3 = np.random.randn(256, 64) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(64)

        # FC2：64→6
        self.W4 = np.random.randn(64, 6) * np.sqrt(2.0 / 64)
        self.b4 = np.zeros(6)

        self.cache = {}

    def forward(self, X, y=None):
        # Conv1 → ReLU → MaxPool
        out1, cache1 = conv_forward(X, self.W1, self.b1, stride=1, pad=1)
        relu1 = relu(out1)
        pool1, cache_pool1 = maxpool_forward(relu1, pool_size=2, stride=2)

        # Conv2 → ReLU → MaxPool
        out2, cache2 = conv_forward(pool1, self.W2, self.b2, stride=1, pad=1)
        relu2 = relu(out2)
        pool2, cache_pool2 = maxpool_forward(relu2, pool_size=2, stride=2)

        # Flatten
        N = X.shape[0]
        flat = pool2.reshape(N, -1)  # (N, 16*4*4=256)

        # FC1 → ReLU
        fc1 = flat @ self.W3 + self.b3
        relu3 = relu(fc1)

        # FC2（输出层，不加激活）
        logits = relu3 @ self.W4 + self.b4

        self.cache = {
            'X': X, 'out1': out1, 'relu1': relu1, 'cache1': cache1,
            'cache_pool1': cache_pool1, 'pool1': pool1,
            'out2': out2, 'relu2': relu2, 'cache2': cache2,
            'cache_pool2': cache_pool2, 'pool2': pool2,
            'flat': flat, 'fc1': fc1, 'relu3': relu3,
            'logits': logits, 'N': N
        }

        if y is not None:
            probs = softmax(logits)
            loss = cross_entropy_loss(probs, y)
            return probs, loss
        return softmax(logits)

    def backward(self, y):
        c = self.cache
        N = c['N']

        # Softmax + CrossEntropy 反向
        probs = softmax(c['logits'])
        d_logits = probs.copy()
        d_logits[np.arange(N), y] -= 1
        d_logits /= N

        # FC2 反向
        self.dW4 = c['relu3'].T @ d_logits
        self.db4 = d_logits.sum(axis=0)
        d_relu3 = d_logits @ self.W4.T

        # ReLU3 反向
        d_fc1 = relu_backward(d_relu3, c['fc1'])

        # FC1 反向
        self.dW3 = c['flat'].T @ d_fc1
        self.db3 = d_fc1.sum(axis=0)
        d_flat = d_fc1 @ self.W3.T

        # Flatten 反向
        d_pool2 = d_flat.reshape(c['pool2'].shape)

        # MaxPool2 反向
        d_relu2 = maxpool_backward(d_pool2, c['cache_pool2'])

        # ReLU2 反向
        d_out2 = relu_backward(d_relu2, c['out2'])

        # Conv2 反向
        d_pool1, self.dW2, self.db2 = conv_backward(d_out2, c['cache2'])

        # MaxPool1 反向
        d_relu1 = maxpool_backward(d_pool1, c['cache_pool1'])

        # ReLU1 反向
        d_out1 = relu_backward(d_relu1, c['out1'])

        # Conv1 反向
        _, self.dW1, self.db1 = conv_backward(d_out1, c['cache1'])

    def update(self, lr):
        for W, b, dW, db in [
            (self.W1, self.b1, self.dW1, self.db1),
            (self.W2, self.b2, self.dW2, self.db2),
            (self.W3, self.b3, self.dW3, self.db3),
            (self.W4, self.b4, self.dW4, self.db4),
        ]:
            W -= lr * dW
            b -= lr * db

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# ============================================================
# Part 3: 训练
# ============================================================
print("\nPart 3: 训练 CNN（这需要几分钟，请耐心等待...）")
print("=" * 50)

model = DigitCNN()
lr = 0.01
batch_size = 32
n_epochs = 30

train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(n_epochs):
    # 打乱训练数据
    idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[idx]
    y_train_shuffled = y_train[idx]

    epoch_losses = []
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        _, loss = model.forward(X_batch, y_batch)
        model.backward(y_batch)
        model.update(lr)
        epoch_losses.append(loss)

    # 评估（用小批次避免内存问题）
    train_loss = np.mean(epoch_losses)

    val_loss_list = []
    for i in range(0, len(X_val), batch_size):
        _, vloss = model.forward(X_val[i:i+batch_size], y_val[i:i+batch_size])
        val_loss_list.append(vloss)
    val_loss = np.mean(val_loss_list)

    train_acc = model.accuracy(X_train[:200], y_train[:200])  # 用子集加速
    val_acc = model.accuracy(X_val, y_val)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")

# ============================================================
# Part 4: 评估与可视化
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 测试集评估与可视化")
print("=" * 50)

test_acc = model.accuracy(X_test, y_test)
print(f"测试集准确率：{test_acc:.2%}")

# 混淆矩阵
preds_test = model.predict(X_test)
confusion = np.zeros((6, 6), dtype=int)
for true, pred in zip(y_test, preds_test):
    confusion[true, pred] += 1

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'CNN 手写数字识别结果（测试准确率：{test_acc:.2%}）', fontsize=14)

# 训练曲线（loss）
ax = axes[0][0]
ax.plot(train_losses, 'b-', label='训练Loss', linewidth=2)
ax.plot(val_losses, 'r-', label='验证Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss 曲线')
ax.legend()
ax.grid(True, alpha=0.3)

# 训练曲线（accuracy）
ax = axes[0][1]
ax.plot(train_accs, 'b-', label='训练准确率', linewidth=2)
ax.plot(val_accs, 'r-', label='验证准确率', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('准确率')
ax.set_title('准确率曲线')
ax.legend()
ax.grid(True, alpha=0.3)

# 混淆矩阵
ax = axes[0][2]
im = ax.imshow(confusion, cmap='Blues')
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xticklabels([str(i+1) for i in range(6)])
ax.set_yticklabels([str(i+1) for i in range(6)])
ax.set_xlabel('预测数字')
ax.set_ylabel('真实数字')
ax.set_title('混淆矩阵（行=真实，列=预测）')
for i in range(6):
    for j in range(6):
        ax.text(j, i, str(confusion[i, j]), ha='center', va='center',
               fontsize=12, color='white' if confusion[i, j] > 20 else 'black')
plt.colorbar(im, ax=ax)

# 预测示例（正确 + 错误）
ax = axes[1][0]
ax.axis('off')
correct_idx = np.where(preds_test == y_test)[0][:8]
wrong_idx = np.where(preds_test != y_test)[0][:4]

imgs = []
titles = []
for idx in correct_idx:
    imgs.append(X_test[idx, 0])
    titles.append(f'真:{y_test[idx]+1} 预:{preds_test[idx]+1} ✓')
for idx in wrong_idx:
    imgs.append(X_test[idx, 0])
    titles.append(f'真:{y_test[idx]+1} 预:{preds_test[idx]+1} ✗')

ax2 = fig.add_axes([0.01, 0.01, 0.64, 0.45])
ax2.axis('off')
for k, (img, title) in enumerate(zip(imgs, titles)):
    ax_sub = fig.add_axes([0.01 + (k % 6) * 0.105, 0.01 + (k // 6) * 0.22, 0.10, 0.20])
    ax_sub.imshow(img, cmap='gray')
    color = 'green' if '✓' in title else 'red'
    ax_sub.set_title(title, fontsize=8, color=color)
    ax_sub.axis('off')

# 第一层卷积核可视化
ax = axes[1][2]
kernels_vis = model.W1[:, 0, :, :]  # (8, 3, 3)
grid = np.zeros((3*4, 3*2))  # 4行2列（只显示8个核）
for k in range(8):
    r, c = k // 2, k % 2
    kmin, kmax = kernels_vis[k].min(), kernels_vis[k].max()
    if kmax > kmin:
        norm_k = (kernels_vis[k] - kmin) / (kmax - kmin)
    else:
        norm_k = kernels_vis[k]
    grid[r*3:(r+1)*3, c*3:(c+1)*3] = norm_k

ax.imshow(grid, cmap='RdBu_r')
ax.set_title('学到的第一层卷积核（8个 3×3）\n每个核检测不同的局部特征')
ax.axis('off')

plt.savefig('03_cnn/digit_cnn_results.png', dpi=100, bbox_inches='tight')
print("图片已保存：03_cnn/digit_cnn_results.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【网络深度实验】
   在现有网络基础上，再加一个 Conv(32 filters) + MaxPool 层。
   修改 FC1 的输入尺寸（从256改为新尺寸）。
   训练后，准确率有提升吗？为什么（更多层 = 更多特征？）

2. 【过拟合实验】
   把 n_per_class 从 200 减少到 20，故意造成过拟合。
   观察训练准确率 vs 验证准确率的差异。
   如何解决过拟合？（提示：Dropout，数据增强，减小网络）

3. 【卷积核解读】
   观察第一层卷积核（已可视化）。
   蓝色区域和红色区域分别代表什么？
   尝试判断：哪个卷积核可能是"水平边缘检测器"？
   哪个可能是"角点检测器"？
""")
