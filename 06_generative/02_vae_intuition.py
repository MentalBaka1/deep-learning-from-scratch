"""
==============================================================
第6章 第2节：变分自编码器（VAE）—— 学会"生成"
==============================================================

【为什么需要它？】
普通自编码器（AE）的问题：
  潜空间是"不连续的"！

  什么意思？假设 AE 把 "猫的图片" 编码到潜空间点 z=(1.2, 0.8)
           把 "狗的图片" 编码到潜空间点 z=(-0.9, 1.5)

  如果我们在 z=(0.15, 1.15) 采样（两点之间），解码出的是什么？
  → 可能是完全无意义的噪声图片！
  → AE 只在"见过的点"附近有意义，其他区域是"空洞"

变分自编码器（VAE）的解决方案：
  不把输入编码成"一个点"，而是编码成"一个分布"！

  编码器输出：μ（均值）和 σ（标准差）
  潜向量：从 N(μ, σ²) 中采样 z = μ + σ * ε（ε~N(0,1)）
  解码：z → 重建输入

  好处：
  1. 潜空间连续且有结构（附近的点解码出相似的内容）
  2. 可以从 N(0,1) 随机采样，生成全新的样本
  3. 不同类别在潜空间里有重叠，插值有意义

【生活类比：学艺术风格 vs 记住每幅画】
  普通 AE = 把每幅画记住，只能"复原"见过的画
  VAE     = 学会了"画风规律"，可以创作全新的画

  具体类比：
    普通 AE 的潜空间 ≈ 精确的 GPS 坐标（记住每个位置）
    VAE 的潜空间 ≈ 模糊的"地区"（猫大约在东北角，狗大约在西北角）
    从"猫区域"采样，解码出不同的猫（而不是同一只猫）

【存在理由】
解决问题：普通 AE 的潜空间不连续，无法生成新样本
核心思想：用概率分布代替确定性编码，用 KL 散度约束潜空间结构
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: VAE 的数学基础
# ============================================================
print("=" * 50)
print("Part 1: VAE 的数学直觉")
print("=" * 50)

"""
VAE 的损失函数：ELBO（Evidence Lower Bound）

  Loss = 重建损失 + KL 散度
       = E[||x - x_hat||²]  +  KL(N(μ,σ²) || N(0,1))

重建损失：和普通 AE 一样，让重建越逼真越好

KL 散度（KL Divergence）：
  衡量两个概率分布的"距离"
  KL(q||p) = 0 当且仅当 q 和 p 完全相同

  KL(N(μ,σ²) || N(0,1)) = ½ * Σ(σ² + μ² - 1 - log(σ²))

  这一项的作用：
  - 让编码器输出的分布靠近标准正态 N(0,1)
  - 防止 μ 离 0 太远（避免潜空间"塌陷"）
  - 防止 σ 太小（避免变成普通 AE 的点编码）
  - 让不同样本的编码分布在 0 附近有重叠 → 空间连续！

两项的权衡：
  重建损失大 → 模型更关注重建质量
  KL 散度大  → 潜空间更结构化，但重建质量可能下降
  超参数 β 控制平衡（β-VAE）
"""

def kl_divergence(mu, log_var):
    """
    KL(N(mu, exp(log_var)) || N(0, 1))
    = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    使用 log_var = log(σ²) 提高数值稳定性
    """
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=-1)

# 演示 KL 散度
print("KL 散度直觉演示：")
configs = [
    (0.0, 0.0, "mu=0, sigma=1（完全匹配N(0,1)）"),
    (0.5, 0.0, "mu=0.5, sigma=1（均值偏移）"),
    (0.0, np.log(0.1**2), "mu=0, sigma=0.1（非常窄，接近点编码）"),
    (2.0, np.log(4.0), "mu=2, sigma=2（偏移且宽）"),
]
for mu, lv, desc in configs:
    kl = kl_divergence(np.array([mu]), np.array([lv]))
    print(f"  {desc}")
    print(f"    KL = {kl[0]:.3f}")

print("\n→ KL 散度越大，编码器的分布越偏离标准正态")
print("→ 损失函数中 KL 项约束编码器不能过于偏离")

# ============================================================
# Part 2: 重参数化技巧 —— 让随机采样可以反向传播
# ============================================================
print("\nPart 2: 重参数化技巧（Reparameterization Trick）")
print("=" * 50)

"""
问题：从 N(μ, σ²) 采样 z 是随机操作，不可微分！
  反向传播需要计算 dLoss/dμ 和 dLoss/dσ
  但通过随机节点的梯度无法直接计算

重参数化技巧（Kingma & Welling, 2014）：
  把随机性"移出"到一个固定的噪声变量 ε

  原来：z ~ N(μ, σ²)
  重参数化：z = μ + σ * ε，其中 ε ~ N(0, 1)

  关键：ε 是和 μ、σ 无关的随机变量！
       z = μ + σ * ε 是 μ 和 σ 的确定性函数（给定 ε）
       对 μ 和 σ 的梯度可以正常反向传播！

  类比：
    原来：让机器人随机决定每步走多远（不可控）
    重参数化：随机决定一个"方向系数 ε"，机器人走 μ + σ*ε 步
              机器人走了多少步（z）是 μ 和 σ 的函数 → 可求导！
"""

def reparameterize(mu, log_var):
    """
    重参数化采样：z = mu + eps * sigma
    eps ~ N(0, 1) 是独立的随机变量
    """
    sigma = np.exp(0.5 * log_var)  # sigma = exp(log_var / 2)
    eps = np.random.randn(*mu.shape)  # 独立随机噪声
    z = mu + sigma * eps  # 确定性函数！
    return z, eps  # 同时返回 eps（反向传播需要）

print("重参数化演示：")
mu_demo = np.array([[1.0, 2.0]])
lv_demo = np.array([[0.0, -1.0]])  # log(sigma²): 0→sigma=1, -1→sigma≈0.6

print(f"  编码器输出 μ = {mu_demo[0]}")
print(f"  编码器输出 log_var = {lv_demo[0]}")
print(f"  sigma = exp(log_var/2) = {np.exp(0.5*lv_demo[0]).round(3)}")
print(f"  采样 3 次（ε ~ N(0,1)）：")
for _ in range(3):
    z_sample, eps = reparameterize(mu_demo, lv_demo)
    print(f"    z = μ + σ*ε = {z_sample[0].round(3)}  (ε = {eps[0].round(3)})")

# ============================================================
# Part 3: 完整 VAE 实现
# ============================================================
print("\nPart 3: VAE 完整实现")
print("=" * 50)

class VAE:
    """
    变分自编码器（Variational Autoencoder）

    编码器：x → μ, log_var
    采样：z = μ + σ * ε（重参数化）
    解码器：z → x_hat
    损失：重建损失 + β * KL散度
    """
    def __init__(self, d_input, d_latent, d_hidden=64, beta=1.0):
        self.d_input = d_input
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.beta = beta  # KL 权重

        # 编码器（共享层 → 分支为 μ 和 log_var）
        s1 = np.sqrt(2.0 / d_input)
        s2 = np.sqrt(2.0 / d_hidden)
        self.W_enc = np.random.randn(d_input, d_hidden) * s1
        self.b_enc = np.zeros(d_hidden)

        # μ 分支
        self.W_mu = np.random.randn(d_hidden, d_latent) * s2
        self.b_mu = np.zeros(d_latent)

        # log_var 分支
        self.W_lv = np.random.randn(d_hidden, d_latent) * s2
        self.b_lv = np.zeros(d_latent)

        # 解码器
        s3 = np.sqrt(2.0 / d_latent)
        self.W_dec1 = np.random.randn(d_latent, d_hidden) * s3
        self.b_dec1 = np.zeros(d_hidden)
        self.W_dec2 = np.random.randn(d_hidden, d_input) * s2
        self.b_dec2 = np.zeros(d_input)

        self.cache = None

    def encode(self, x):
        """编码：x → (μ, log_var)"""
        h = np.maximum(0, x @ self.W_enc + self.b_enc)  # ReLU
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_lv + self.b_lv
        # 限制 log_var 范围，防止数值不稳定
        log_var = np.clip(log_var, -10, 10)
        return mu, log_var, h

    def decode(self, z):
        """解码：z → x_hat"""
        h = np.maximum(0, z @ self.W_dec1 + self.b_dec1)
        x_hat = h @ self.W_dec2 + self.b_dec2
        return x_hat, h

    def forward(self, x):
        """前向传播：返回重建、潜向量、μ、log_var"""
        mu, log_var, h_enc = self.encode(x)
        z, eps = reparameterize(mu, log_var)  # 重参数化采样
        x_hat, h_dec = self.decode(z)

        self.cache = (x, h_enc, mu, log_var, z, eps, h_dec, x_hat)
        return x_hat, z, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        """
        ELBO 损失：重建损失 + β * KL散度
        """
        batch = len(x)
        recon_loss = np.mean((x_hat - x)**2)  # MSE 重建损失
        kl_loss = np.mean(kl_divergence(mu, log_var))  # 平均 KL
        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    def backward(self, lr):
        """反向传播"""
        x, h_enc, mu, log_var, z, eps, h_dec, x_hat = self.cache
        batch = len(x)
        sigma = np.exp(0.5 * log_var)

        # 重建损失对 x_hat 的梯度
        d_xhat = 2 * (x_hat - x) / batch

        # 解码器第二层
        dW_dec2 = h_dec.T @ d_xhat
        db_dec2 = d_xhat.sum(0)
        d_h_dec = d_xhat @ self.W_dec2.T
        d_h_dec *= (h_dec > 0)  # ReLU 反向

        dW_dec1 = z.T @ d_h_dec
        db_dec1 = d_h_dec.sum(0)
        d_z = d_h_dec @ self.W_dec1.T

        # 重参数化的梯度：z = mu + sigma * eps
        # d_recon/d_mu = d_z/d_mu * d_loss/d_z = 1 * d_z
        # d_recon/d_sigma = eps * d_z
        d_mu_recon = d_z
        d_sigma = eps * d_z  # d_recon/d_sigma

        # KL 损失对 μ 和 log_var 的梯度（解析解）
        # d_KL/d_mu = mu / batch
        # d_KL/d_log_var = 0.5 * (exp(log_var) - 1) / batch
        d_mu_kl = mu / batch
        d_lv_kl = 0.5 * (np.exp(log_var) - 1) / batch

        # 将 d_sigma 转换为 d_log_var
        # sigma = exp(0.5 * log_var)  → d_sigma/d_log_var = 0.5 * sigma
        d_lv_recon = d_sigma * 0.5 * sigma

        # 合并梯度
        d_mu = d_mu_recon + self.beta * d_mu_kl
        d_lv = d_lv_recon + self.beta * d_lv_kl

        # 编码器 μ 分支
        dW_mu = h_enc.T @ d_mu
        db_mu = d_mu.sum(0)

        # 编码器 log_var 分支
        dW_lv = h_enc.T @ d_lv
        db_lv = d_lv.sum(0)

        # 编码器共享层
        d_h_enc = d_mu @ self.W_mu.T + d_lv @ self.W_lv.T
        d_h_enc *= (h_enc > 0)  # ReLU 反向

        dW_enc = x.T @ d_h_enc
        db_enc = d_h_enc.sum(0)

        # 梯度裁剪
        for grad in [dW_enc, dW_mu, dW_lv, dW_dec1, dW_dec2]:
            np.clip(grad, -5, 5, out=grad)

        # 更新参数
        for W, dW, b, db in [
            (self.W_enc, dW_enc, self.b_enc, db_enc),
            (self.W_mu, dW_mu, self.b_mu, db_mu),
            (self.W_lv, dW_lv, self.b_lv, db_lv),
            (self.W_dec1, dW_dec1, self.b_dec1, db_dec1),
            (self.W_dec2, dW_dec2, self.b_dec2, db_dec2),
        ]:
            W -= lr * dW
            b -= lr * db

# ============================================================
# Part 4: 生成合成数据并训练
# ============================================================
print("生成训练数据...")

def generate_two_class_data(n_samples=600, d_input=20):
    """
    生成两类简单数据（无外部依赖）：
    类别0：偏正值特征
    类别1：偏负值特征，带不同的协方差结构
    """
    n_half = n_samples // 2
    # 类别0：均值在正方向
    X0 = np.random.randn(n_half, d_input) * 0.5 + 0.8
    # 类别1：均值在负方向
    X1 = np.random.randn(n_half, d_input) * 0.5 - 0.8
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_half), np.ones(n_half)])
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

d_input = 20
X_data, y_data = generate_two_class_data(n_samples=500, d_input=d_input)
X_train, y_train = X_data[:400], y_data[:400]
X_test, y_test = X_data[400:], y_data[400:]

vae = VAE(d_input=d_input, d_latent=2, d_hidden=32, beta=1.0)

n_epochs = 60
batch_size = 32
lr = 0.002
train_losses = []
recon_losses = []
kl_losses = []

print("训练 VAE...")
for epoch in range(n_epochs):
    idx = np.random.permutation(len(X_train))
    epoch_total = epoch_recon = epoch_kl = 0

    for start in range(0, len(X_train), batch_size):
        Xb = X_train[idx[start:start+batch_size]]
        x_hat, z, mu, lv = vae.forward(Xb)
        total, recon, kl = vae.loss(Xb, x_hat, mu, lv)
        epoch_total += total
        epoch_recon += recon
        epoch_kl += kl
        vae.backward(lr)

    n_batches = (len(X_train) + batch_size - 1) // batch_size
    train_losses.append(epoch_total / n_batches)
    recon_losses.append(epoch_recon / n_batches)
    kl_losses.append(epoch_kl / n_batches)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: total={train_losses[-1]:.4f}, "
              f"recon={recon_losses[-1]:.4f}, kl={kl_losses[-1]:.4f}")

# ============================================================
# Part 5: 可视化
# ============================================================
print("\n可视化 VAE 结果...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('变分自编码器（VAE）：生成式学习', fontsize=14)

# 1. 损失曲线（重建损失 + KL 散度）
ax = axes[0][0]
ax.plot(train_losses, 'b-', linewidth=2, label='总损失')
ax.plot(recon_losses, 'g--', linewidth=2, label='重建损失')
ax.plot(kl_losses, 'r-.', linewidth=2, label='KL 散度')
ax.set_xlabel('Epoch')
ax.set_ylabel('损失')
ax.set_title('VAE 训练损失\n（重建损失 + KL散度）')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 潜空间分布（VAE vs AE 对比）
ax = axes[0][1]
_, _, mu_test, lv_test = vae.forward(X_test)
colors_map = {0: 'blue', 1: 'red'}
labels_map = {0: '类别0', 1: '类别1'}
for c in [0, 1]:
    mask = y_test == c
    ax.scatter(mu_test[mask, 0], mu_test[mask, 1],
               c=colors_map[c], s=30, alpha=0.5, label=labels_map[c])
# 画出标准正态的轮廓
theta = np.linspace(0, 2*np.pi, 100)
for r in [1, 2, 3]:
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'k--', alpha=0.2, linewidth=1)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.legend()
ax.set_xlabel('潜维度 1 (μ₁)')
ax.set_ylabel('潜维度 2 (μ₂)')
ax.set_title('VAE 潜空间\n（KL约束使分布集中在N(0,1)附近）')
ax.grid(True, alpha=0.3)

# 3. 生成新样本
ax = axes[0][2]
"""
VAE 生成新样本的过程：
1. 从标准正态采样：z ~ N(0, I)
2. 通过解码器：x_new = Decoder(z)

因为 KL 散度约束了潜空间分布接近 N(0, I)，
这个采样过程会生成有意义的样本！
"""
n_generate = 200
z_samples = np.random.randn(n_generate, 2)  # 从标准正态采样
x_generated, _ = vae.decode(z_samples)

# 用第一个和第二个特征的分布对比真实数据和生成数据
ax.hist(X_test[:, 0], bins=20, alpha=0.5, color='blue', density=True, label='真实数据(特征1)')
ax.hist(x_generated[:, 0], bins=20, alpha=0.5, color='red', density=True, label='生成样本(特征1)')
ax.set_xlabel('特征值')
ax.set_ylabel('密度')
ax.set_title(f'生成样本 vs 真实数据\n（从N(0,1)采样z，解码生成）')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 潜空间网格采样（理解潜空间结构）
ax = axes[1][0]
grid_size = 10
z1_vals = np.linspace(-2.5, 2.5, grid_size)
z2_vals = np.linspace(-2.5, 2.5, grid_size)

# 在潜空间网格采样并解码
grid_points = []
for z2 in z2_vals:
    for z1 in z1_vals:
        grid_points.append([z1, z2])
grid_z = np.array(grid_points)
grid_decoded, _ = vae.decode(grid_z)

# 可视化第1个特征的解码值
grid_feature1 = grid_decoded[:, 0].reshape(grid_size, grid_size)
im = ax.imshow(grid_feature1, cmap='RdBu_r', extent=[-2.5, 2.5, -2.5, 2.5],
               origin='lower', aspect='auto')
plt.colorbar(im, ax=ax)
ax.set_xlabel('潜维度1 (z₁)')
ax.set_ylabel('潜维度2 (z₂)')
ax.set_title('潜空间网格解码\n（颜色=解码出的第1个特征值）')

# 叠加潜空间点
for c in [0, 1]:
    mask = y_test == c
    ax.scatter(mu_test[mask, 0], mu_test[mask, 1],
               c=colors_map[c], s=20, alpha=0.3, zorder=5)

# 5. AE vs VAE 潜空间对比（概念说明）
ax = axes[1][1]
ax.axis('off')
comparison_text = """
AE vs VAE 潜空间对比：

【普通 AE】
  编码：x → z（单个确定点）

  潜空间：稀疏、不连续
  ●          ●    类别0


        ●  ●      类别1

  两类之间有"空洞"，采样出无意义样本
  → 不能用于生成！

【VAE】
  编码：x → μ,σ → z~N(μ,σ²)

  KL 约束使所有编码向 N(0,1) 聚集
  类别0和1有重叠区域

     类0  类1
    ○○○○○○○○
    ○○ ●●●○○
    ○○○○○○○○

  → 潜空间连续，采样有意义！
"""
ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('AE vs VAE 核心区别')

# 6. β 参数的影响（实验对比）
ax = axes[1][2]
beta_values = [0.001, 0.1, 1.0, 5.0]
recon_vs_kl = []

for beta in beta_values:
    vae_tmp = VAE(d_input=d_input, d_latent=2, d_hidden=32, beta=beta)
    for ep in range(30):
        idx = np.random.permutation(len(X_train))
        for start in range(0, len(X_train), 32):
            Xb = X_train[idx[start:start+32]]
            xh, _, mu_b, lv_b = vae_tmp.forward(Xb)
            vae_tmp.backward(0.003)
    xh_test, _, mu_b, lv_b = vae_tmp.forward(X_test)
    r_loss = np.mean((xh_test - X_test)**2)
    k_loss = np.mean(kl_divergence(mu_b, lv_b))
    recon_vs_kl.append((r_loss, k_loss))

recon_v = [r for r, k in recon_vs_kl]
kl_v = [k for r, k in recon_vs_kl]
ax.scatter(kl_v, recon_v, c=['blue', 'green', 'orange', 'red'],
           s=100, zorder=5, edgecolor='black')
for i, (beta, (r, k)) in enumerate(zip(beta_values, recon_vs_kl)):
    ax.annotate(f'β={beta}', (k, r), textcoords="offset points",
                xytext=(5, 5), fontsize=9)
ax.set_xlabel('KL 散度（潜空间结构质量）')
ax.set_ylabel('重建损失')
ax.set_title('β 参数：生成质量 vs 重建质量\nβ大→结构好但重建差')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('06_generative/vae_intuition.png', dpi=80, bbox_inches='tight')
print("图片已保存：06_generative/vae_intuition.png")
plt.show()

# ============================================================
# 总结：AE vs VAE vs GAN
# ============================================================
print("\n" + "=" * 50)
print("生成模型全景对比")
print("=" * 50)
print("""
三种主要生成模型：

【AE (Autoencoder)】
  目标：重建输入（无监督降维）
  潜空间：确定性点，不连续
  生成能力：弱（不能采样新样本）
  主要用途：降维、特征提取、异常检测

【VAE (Variational Autoencoder)】
  目标：学习数据的概率分布
  潜空间：连续分布（N(μ,σ²)），接近标准正态
  生成能力：中（可以采样，但图像可能模糊）
  主要用途：生成、插值、表征学习
  损失：重建损失 + KL 散度

【GAN (Generative Adversarial Network, 未包含在本课程）】
  目标：生成器和判别器博弈
  潜空间：隐式分布（不显式估计）
  生成能力：强（生成图像往往非常清晰）
  主要用途：高质量图像/视频生成
  损失：生成器想骗过判别器，判别器想区分真假

现代趋势：扩散模型（Diffusion Models，如 Stable Diffusion）
  以更稳定的方式生成高质量图像，超越了传统 GAN 和 VAE
""")

# ============================================================
# 思考题
# ============================================================
print("=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【β-VAE 的直觉】
   β-VAE 在损失中加大 KL 权重（β > 1）：
     loss = recon + β * KL
   大的 β 会让潜空间更"解缠绕"（disentangled）：
   不同潜维度控制不同的生成因素（如维度1=颜色，维度2=形状）
   但重建质量会下降。
   这和 L1/L2 正则化的权衡有什么相似之处？

2. 【条件 VAE（CVAE）】
   如果想生成特定类别的样本（不是随机生成），可以用条件 VAE：
   编码器：q(z|x, y)（给定输入和标签，编码潜向量）
   解码器：p(x|z, y)（给定潜向量和标签，重建）
   修改本节的 VAE 代码实现 CVAE（把标签 y 拼接到输入和潜向量）

3. 【VAE 的后验坍缩（Posterior Collapse）】
   训练 VAE 时有时会出现：解码器变强后，完全忽视潜向量 z
   （因为解码器够强，不用 z 也能重建），此时 KL 散度趋近于 0。
   这叫"后验坍缩"。
   有什么策略可以避免？
   （提示：KL annealing，即 β 从0开始慢慢增大）
""")
