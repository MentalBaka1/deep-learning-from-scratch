"""
==============================================================
第6章 第1节：自编码器 —— 压缩与重建
==============================================================

【为什么需要它？】
有监督学习需要大量标注数据（贵！难！）。
很多时候数据有，但标签没有：
  - 大量图片，但没有人一张张打标签
  - 大量文本，但没有人标注情感/主题
  - 大量传感器数据，但没有"正常/异常"标签

自编码器（Autoencoder）是一种无监督方法：
  不需要标签，只要原始数据！
  让网络学会"把数据压缩后再重建"，如果重建得好，
  说明压缩表示捕获了数据的本质特征

【生活类比：文件压缩】
  AutoEncoder ≈ 压缩软件（如 WinRAR、ZIP）

  编码器（Encoder）= 压缩：把大文件变小文件
  潜空间（Latent Space）= 压缩后的文件（占用内存少）
  解码器（Decoder）= 解压：把小文件还原成大文件

  好的压缩算法：解压后和原文件几乎一样
  好的自编码器：重建后和原输入几乎一样

  但不同的是：自编码器的"压缩规则"是从数据中学出来的！
  WinRAR 用固定算法，而 AE 会针对你的数据类型自适应。

【存在理由】
解决问题：无监督学习、降维、特征提取、异常检测、数据压缩
核心思想：强迫网络学习数据的低维表示，再从低维重建原始数据
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

# ============================================================
# Part 1: 自编码器的结构
# ============================================================
print("=" * 50)
print("Part 1: 自编码器结构解析")
print("=" * 50)

"""
自编码器结构：

  输入 x (d_input)
      ↓
  [编码器 Encoder]
      ↓
  潜向量 z (d_latent)  ← 瓶颈层（bottle neck），维度 << 输入维度
      ↓
  [解码器 Decoder]
      ↓
  重建 x_hat (d_input)

  损失：重建误差 = ||x - x_hat||²（MSE）

信息瓶颈（Bottleneck）的作用：
  潜空间维度 << 输入维度，强迫网络压缩信息
  如果潜空间够大，网络可以直接"记住"输入（没有压缩）
  必须让潜空间足够小，才能学到有用的特征

  例：输入784维（28×28图像），潜空间2维
    → 网络必须用2个数字描述图像
    → 这2个数字会自然地编码图像的"主要变化方向"
    → 类似于 PCA，但是非线性的！
"""

class AutoEncoder:
    """
    全连接自编码器
    Encoder: d_input → d_hidden → d_latent
    Decoder: d_latent → d_hidden → d_input
    """
    def __init__(self, d_input, d_latent, d_hidden=64):
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden

        # 编码器权重
        scale1 = np.sqrt(2.0 / d_input)
        scale2 = np.sqrt(2.0 / d_hidden)
        self.W_enc1 = np.random.randn(d_input, d_hidden) * scale1
        self.b_enc1 = np.zeros(d_hidden)
        self.W_enc2 = np.random.randn(d_hidden, d_latent) * scale2
        self.b_enc2 = np.zeros(d_latent)

        # 解码器权重（结构对称）
        scale3 = np.sqrt(2.0 / d_latent)
        self.W_dec1 = np.random.randn(d_latent, d_hidden) * scale3
        self.b_dec1 = np.zeros(d_hidden)
        self.W_dec2 = np.random.randn(d_hidden, d_input) * scale2
        self.b_dec2 = np.zeros(d_input)

        self.cache = None

    def encode(self, x):
        """编码：输入 → 潜向量"""
        h1 = np.maximum(0, x @ self.W_enc1 + self.b_enc1)  # ReLU
        z = np.tanh(h1 @ self.W_enc2 + self.b_enc2)        # Tanh（限制潜空间范围）
        return z, h1

    def decode(self, z):
        """解码：潜向量 → 重建输入"""
        h2 = np.maximum(0, z @ self.W_dec1 + self.b_dec1)  # ReLU
        x_hat = h2 @ self.W_dec2 + self.b_dec2             # 线性输出
        return x_hat, h2

    def forward(self, x):
        """前向传播"""
        z, h_enc = self.encode(x)
        x_hat, h_dec = self.decode(z)
        self.cache = (x, h_enc, z, h_dec, x_hat)
        return x_hat, z

    def backward(self, x, x_hat, lr):
        """反向传播（MSE 损失）"""
        x_in, h_enc, z, h_dec, _ = self.cache
        batch = len(x)

        # MSE 损失的梯度
        d_out = 2 * (x_hat - x) / batch  # (batch, d_input)

        # 解码器第二层
        dW_dec2 = h_dec.T @ d_out
        db_dec2 = d_out.sum(0)
        d_h_dec = d_out @ self.W_dec2.T

        # ReLU 反向
        d_h_dec *= (h_dec > 0)

        # 解码器第一层
        dW_dec1 = z.T @ d_h_dec
        db_dec1 = d_h_dec.sum(0)
        d_z = d_h_dec @ self.W_dec1.T

        # Tanh 反向
        d_z_pre = d_z * (1 - z**2)  # tanh 导数

        # 编码器第二层
        dW_enc2 = h_enc.T @ d_z_pre
        db_enc2 = d_z_pre.sum(0)
        d_h_enc = d_z_pre @ self.W_enc2.T

        # ReLU 反向
        d_h_enc *= (h_enc > 0)

        # 编码器第一层
        dW_enc1 = x_in.T @ d_h_enc
        db_enc1 = d_h_enc.sum(0)

        # 更新参数
        for W, dW, b, db in [
            (self.W_enc1, dW_enc1, self.b_enc1, db_enc1),
            (self.W_enc2, dW_enc2, self.b_enc2, db_enc2),
            (self.W_dec1, dW_dec1, self.b_dec1, db_dec1),
            (self.W_dec2, dW_dec2, self.b_dec2, db_dec2),
        ]:
            W -= lr * dW
            b -= lr * db

# ============================================================
# Part 2: 生成合成数据集
# ============================================================
print("Part 2: 生成数据并训练自编码器")
print("=" * 50)

"""
数据集：手写数字样式的合成图像（8×8 = 64 维）
  - 不使用任何外部库，完全合成
  - 用高斯噪声 + 结构性模式模拟不同"类别"
"""

def generate_digit_like_data(n_samples=800, img_size=8):
    """
    生成 3 类简单的合成"图像"数据（不依赖任何外部数据集）
    类别0：左侧亮（竖条纹）
    类别1：上方亮（横条纹）
    类别2：对角亮（对角线）
    """
    d = img_size * img_size
    X = []
    labels = []

    n_per_class = n_samples // 3
    for cls in range(3):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size))
            if cls == 0:  # 竖条纹
                img[:, :img_size//2] = 1.0
            elif cls == 1:  # 横条纹
                img[:img_size//2, :] = 1.0
            else:  # 对角线
                for i in range(img_size):
                    for j in range(img_size):
                        if abs(i - j) <= 1:
                            img[i, j] = 1.0

            # 加入随机噪声
            img += np.random.randn(img_size, img_size) * 0.2
            img = np.clip(img, 0, 1)
            X.append(img.ravel())
            labels.append(cls)

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)

    # 打乱
    idx = np.random.permutation(len(X))
    return X[idx], labels[idx]

img_size = 8
d_input = img_size * img_size  # 64
X_data, labels = generate_digit_like_data(n_samples=600, img_size=img_size)
X_train, y_train = X_data[:480], labels[:480]
X_test, y_test = X_data[480:], labels[480:]

print(f"数据集形状：{X_data.shape}  （{img_size}×{img_size}=64维图像）")
print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

# ============================================================
# Part 3: 训练自编码器
# ============================================================
print("\nPart 3: 训练（64维 → 2维 → 64维）")
print("=" * 50)

ae = AutoEncoder(d_input=64, d_latent=2, d_hidden=32)

n_epochs = 80
batch_size = 32
lr = 0.001
train_losses = []

for epoch in range(n_epochs):
    idx = np.random.permutation(len(X_train))
    epoch_losses = []

    for start in range(0, len(X_train), batch_size):
        batch_idx = idx[start:start+batch_size]
        X_batch = X_train[batch_idx]

        x_hat, z = ae.forward(X_batch)
        loss = np.mean((x_hat - X_batch)**2)
        epoch_losses.append(loss)
        ae.backward(X_batch, x_hat, lr)

    train_losses.append(np.mean(epoch_losses))

    if epoch % 20 == 0:
        test_hat, _ = ae.forward(X_test)
        test_loss = np.mean((test_hat - X_test)**2)
        print(f"  Epoch {epoch:3d}: train_loss={train_losses[-1]:.4f}, test_loss={test_loss:.4f}")

# ============================================================
# Part 4: 可视化潜空间和重建
# ============================================================
print("\nPart 4: 可视化潜空间")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('自编码器：压缩与重建', fontsize=14)

# 1. 训练损失
ax = axes[0][0]
ax.semilogy(train_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('重建损失（MSE，对数）')
ax.set_title('训练损失曲线')
ax.grid(True, alpha=0.3)

# 2. 潜空间可视化（2D 散点图）
ax = axes[0][1]
_, z_test = ae.forward(X_test)
colors = ['red', 'blue', 'green']
class_names = ['竖条纹', '横条纹', '对角线']
for c, (color, name) in enumerate(zip(colors, class_names)):
    mask = y_test == c
    ax.scatter(z_test[mask, 0], z_test[mask, 1], c=color, s=30, alpha=0.6, label=name)
ax.legend()
ax.set_xlabel('潜向量维度1')
ax.set_ylabel('潜向量维度2')
ax.set_title('潜空间（64维→2维）\n不同类别自然分离！')
ax.grid(True, alpha=0.3)

# 3. 重建质量展示
ax = axes[0][2]
# 选取3个测试样本
sample_idx = [0, len(X_test)//3, 2*len(X_test)//3]
x_sample = X_test[sample_idx]
x_hat_sample, _ = ae.forward(x_sample)

# 原始 vs 重建（拼成一行）
original_row = np.hstack([x.reshape(img_size, img_size) for x in x_sample])
reconstructed_row = np.hstack([x.reshape(img_size, img_size) for x in x_hat_sample])
comparison = np.vstack([original_row, np.ones((1, original_row.shape[1])),
                        reconstructed_row])
ax.imshow(comparison, cmap='gray', aspect='auto', vmin=0, vmax=1)
ax.set_xticks([])
ax.set_yticks([img_size//2, img_size + 1 + img_size//2])
ax.set_yticklabels(['原始', '重建'], fontsize=10)
ax.set_title('原始 vs 重建（64→2→64）\n压缩 32 倍后的重建效果')

# 4. 潜空间插值（在两个点之间插值）
ax = axes[1][0]
# 选择类别0和类别1的典型样本
z_class0 = z_test[y_test == 0][0]
z_class1 = z_test[y_test == 1][0]

n_interp = 8
interp_images = []
for alpha in np.linspace(0, 1, n_interp):
    z_interp = (1 - alpha) * z_class0 + alpha * z_class1
    x_interp, _ = ae.decode(z_interp.reshape(1, -1))
    interp_images.append(x_interp[0].reshape(img_size, img_size))

row = np.hstack(interp_images)
ax.imshow(row, cmap='gray', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(n_interp) * img_size + img_size//2)
ax.set_xticklabels([f'{i/(n_interp-1):.1f}' for i in range(n_interp)], fontsize=7)
ax.set_yticks([])
ax.set_xlabel('插值比例（0=竖条纹，1=横条纹）')
ax.set_title('潜空间插值\n两类图像在潜空间连续过渡')

# 5. 不同潜空间维度的重建质量
ax = axes[1][1]
latent_dims = [1, 2, 4, 8, 16, 32]
test_mses = []
for d_lat in latent_dims:
    ae_tmp = AutoEncoder(d_input=64, d_latent=d_lat, d_hidden=32)
    # 快速训练（较少 epoch）
    for ep in range(40):
        idx = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), 32):
            Xb = X_train[idx[start:start+32]]
            xh, _ = ae_tmp.forward(Xb)
            ae_tmp.backward(Xb, xh, 0.002)
    xh_test, _ = ae_tmp.forward(X_test)
    test_mses.append(np.mean((xh_test - X_test)**2))

ax.plot(latent_dims, test_mses, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.set_xlabel('潜空间维度')
ax.set_ylabel('重建 MSE')
ax.set_title('潜空间维度 vs 重建质量\n维度越高，重建越好（但压缩越少）')
ax.grid(True, alpha=0.3)
ax.axvline(64, color='red', linestyle='--', alpha=0.5, label='输入维度（无压缩）')
ax.legend()

# 6. 异常检测应用
ax = axes[1][2]
"""
自编码器做异常检测的原理：
  训练时：用正常数据训练，AE 学会重建正常数据
  推理时：
    - 正常数据：重建误差小（模型"认识"它）
    - 异常数据：重建误差大（模型"不认识"它，压缩后信息丢失）
"""
# 正常样本的重建误差
normal_hat, _ = ae.forward(X_test)
normal_errors = np.mean((normal_hat - X_test)**2, axis=1)

# 生成"异常"样本（随机噪声图像）
anomaly_data = np.random.randn(len(X_test)//3, d_input) * 0.5 + 0.5
anomaly_data = np.clip(anomaly_data, 0, 1)
anomaly_hat, _ = ae.forward(anomaly_data)
anomaly_errors = np.mean((anomaly_hat - anomaly_data)**2, axis=1)

ax.hist(normal_errors, bins=30, alpha=0.6, color='blue', label=f'正常样本\n(均值={normal_errors.mean():.3f})')
ax.hist(anomaly_errors, bins=30, alpha=0.6, color='red', label=f'异常样本\n(均值={anomaly_errors.mean():.3f})')

threshold = normal_errors.mean() + 2 * normal_errors.std()
ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'异常阈值={threshold:.3f}')
ax.set_xlabel('重建误差（MSE）')
ax.set_ylabel('频率')
ax.set_title('异常检测\n正常 vs 异常样本的重建误差分布')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

anomaly_detection_rate = (anomaly_errors > threshold).mean()
false_alarm_rate = (normal_errors > threshold).mean()
print(f"\n异常检测结果（阈值={threshold:.3f}）：")
print(f"  异常检出率：{anomaly_detection_rate:.1%}")
print(f"  误报率：{false_alarm_rate:.1%}")

plt.tight_layout()
plt.savefig('06_generative/autoencoder.png', dpi=80, bbox_inches='tight')
print("\n图片已保存：06_generative/autoencoder.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【为什么潜空间维度不能太小也不能太大？】
   太小（如 d_latent=1）：
     信息丢失太多，无法重建细节
   太大（如 d_latent=64，等于输入维度）：
     网络可以"记住"输入，不需要学习有效压缩
   如何在实践中选择合适的潜空间维度？

2. 【稀疏自编码器（Sparse Autoencoder）】
   在损失函数中加入惩罚项：
     loss = MSE(x, x_hat) + λ * ||z||₁
   这个 L1 范数惩罚让潜向量稀疏（大多数维度接近0）。
   稀疏性为什么有助于学习更有意义的特征？
   （提示：大脑的神经元也是稀疏激活的）

3. 【去噪自编码器（Denoising AE）】
   把干净输入加噪声 x_noisy = x + noise，
   让模型重建干净的 x（而不是重建 x_noisy）：
     loss = MSE(x, decoder(encoder(x_noisy)))
   去噪 AE 为什么能学到更鲁棒的特征？
   实现去噪 AE 并测试去噪效果。
""")
