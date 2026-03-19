"""
====================================================================
第3章 · 第2节 · CNN 架构演进：从 LeNet 到 ResNet
====================================================================

【一句话总结】
CNN 架构的演进史就是"如何把网络做得更深更好"的探索史——
ResNet 的跳跃连接是其中最重要的突破，它直接启发了 Transformer。

【为什么深度学习需要这个？】
- 理解架构设计的思路：每个架构为什么这样设计？解决了什么问题？
- ResNet 的残差连接被 Transformer 直接继承
- 了解"更深 ≠ 更好"的退化问题，理解为什么需要跳跃连接

【核心概念】

1. LeNet-5 (1998, LeCun)
   - 第一个成功的 CNN，用于手写数字识别
   - 结构：Conv→Pool→Conv→Pool→FC→FC→Output
   - 贡献：证明了卷积+池化的有效性

2. AlexNet (2012, Krizhevsky)
   - ImageNet 竞赛冠军，深度学习的"iPhone 时刻"
   - 关键创新：ReLU激活、Dropout正则化、GPU训练
   - 比 LeNet 更深更宽

3. VGGNet (2014, Simonyan)
   - 核心思想：用多个 3×3 小卷积核代替大卷积核
   - 两个 3×3 = 一个 5×5 的感受野，但参数更少、非线性更多
   - 证明了"深度很重要"

4. 退化问题（Degradation Problem）
   - 理论上更深的网络应该不比浅网络差（至少可以学恒等映射）
   - 但实际中 56 层比 20 层效果更差！
   - 原因：优化困难，梯度在传播过程中衰减

5. ResNet (2015, He) — 最重要！
   - 残差连接：y = F(x) + x（输出 = 变换 + 原始输入）
   - 直觉：网络只需学习"残差"（增量），而不是完整映射
   - 为什么有效：梯度可以通过跳跃连接直接流回早期层
   - 影响：使训练 100+ 层网络成为可能
   - 对 Transformer 的影响：Transformer 的每个子层都有残差连接

【前置知识】
第3章第1节 - 卷积与池化
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)


# ====================================================================
# 第一部分：架构演进时间线
# ====================================================================
print("=" * 70, "\n第一部分：CNN 架构演进时间线\n" + "=" * 70)
print("""
  1998        2012         2014         2015         2017
   |           |            |            |            |
   v           v            v            v            v
+--------+ +----------+ +----------+ +----------+ +--------------+
| LeNet-5| | AlexNet  | | VGGNet   | | ResNet   | | Transformer  |
| 5 层   | | 8 层     | | 19 层    | | 152 层   | | 残差连接继承 |
| 手写   | | ImageNet | | 3x3堆叠  | | 跳跃连接 | | + 注意力机制 |
| 数字   | | ReLU     | | 更深更好 | | 解决退化 | | = 现代LLM    |
+--------+ +----------+ +----------+ +----------+ +--------------+
                                           |
                                      最关键突破!
                                    "跳跃连接"让深层网络终于能训练

  深度:    5 -----> 8 -----> 19 -----> 152 层
  错误率:  ~1%    ~16.4%    ~7.3%     ~3.6%   (ImageNet Top-5)
          (MNIST) ----- ImageNet 分类任务 ----->
""")


# ====================================================================
# 第二部分：LeNet 简易实现（纯 NumPy 前向传播）
# ====================================================================
print("=" * 70, "\n第二部分：LeNet 简易实现\n" + "=" * 70)


def conv2d(x, kernel, stride=1):
    """二维卷积（单通道，无填充）"""
    h, w = x.shape
    kh, kw = kernel.shape
    out_h, out_w = (h - kh) // stride + 1, (w - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(x[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel)
    return out


def avg_pool2d(x, size=2):
    """平均池化"""
    h, w = x.shape
    oh, ow = h // size, w // size
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.mean(x[i*size:(i+1)*size, j*size:(j+1)*size])
    return out


relu = lambda x: np.maximum(0, x)

# LeNet-5 前向传播: Conv(5x5,6)→Pool→Conv(5x5,16)→Pool→FC→FC→FC
test_image = np.random.randn(28, 28)
np.random.seed(1)

# 第 1 层: 28x28 → Conv 24x24 → Pool 12x12
k1 = np.random.randn(6, 5, 5) * 0.1
conv1 = np.stack([relu(conv2d(test_image, k)) for k in k1])   # (6,24,24)
pool1 = np.stack([avg_pool2d(fm) for fm in conv1])              # (6,12,12)

# 第 2 层: 12x12 → Conv 8x8 → Pool 4x4（简化：对所有输入通道求和）
k2 = np.random.randn(16, 5, 5) * 0.1
conv2 = np.stack([relu(sum(conv2d(pool1[c], k) for c in range(6))) for k in k2])
pool2 = np.stack([avg_pool2d(fm) for fm in conv2])              # (16,4,4)

# 展平 → 全连接层
flat = pool2.flatten()                                          # (256,)
fc1 = relu(flat @ (np.random.randn(256, 120) * 0.05))          # (120,)
fc2 = relu(fc1 @ (np.random.randn(120, 84) * 0.05))            # (84,)
logits = fc2 @ (np.random.randn(84, 10) * 0.05)                # (10,)

print(f"LeNet-5 维度流:")
print(f"  输入 {test_image.shape} → Conv1 {conv1.shape} → Pool1 {pool1.shape}")
print(f"  → Conv2 {conv2.shape} → Pool2 {pool2.shape} → 展平 {flat.shape} → 输出 {logits.shape}")

# 可视化第一层特征图
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for i in range(6):
    for row, (data, prefix) in enumerate([(conv1, "Conv"), (pool1, "Pool")]):
        axes[row, i].imshow(data[i], cmap="viridis")
        axes[row, i].set_title(f"{prefix}1 #{i}", fontsize=9); axes[row, i].axis("off")
plt.suptitle("LeNet 第一层: 卷积→池化 特征图", fontsize=13)
plt.tight_layout(); plt.savefig("02_lenet_features.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 02_lenet_features.png\n")


# ====================================================================
# 第三部分：3x3 vs 5x5 —— VGGNet 的核心洞察
# ====================================================================
print("=" * 70, "\n第三部分：3x3 vs 5x5 — 为什么小卷积核更好\n" + "=" * 70)

# 数值验证：两层 3x3 的感受野 = 一层 5x5
impulse = np.zeros((7, 7)); impulse[3, 3] = 1.0
k3 = np.ones((3, 3))
after_one_3x3 = conv2d(impulse, k3)                    # (5,5)
after_two_3x3 = conv2d(after_one_3x3, k3)              # (3,3)
after_one_5x5 = conv2d(impulse, np.ones((5, 5)))        # (3,3)

print(f"  两层3x3后 形状={after_two_3x3.shape}, 中心值={after_two_3x3[1,1]:.0f}")
print(f"  一层5x5后 形状={after_one_5x5.shape}, 中心值={after_one_5x5[1,1]:.0f}")
print(f"  感受野大小相同!")

# 参数量对比 (假设通道数 C=64)
C = 64
print(f"\n  参数量对比 (C={C}):")
print(f"    一层5x5: {C*C*25:>8,}   vs  两层3x3: {2*C*C*9:>8,}  (节省 {1-2*9/25:.0%})")
print(f"    一层7x7: {C*C*49:>8,}   vs  三层3x3: {3*C*C*9:>8,}  (节省 {1-3*9/49:.0%})")
print(f"  结论: 同样感受野, 更少参数, 更多非线性!")

# 可视化感受野增长
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for idx, (title, data) in enumerate([
    ("输入: 中心脉冲", impulse),
    ("一层3x3后 (感受野=3x3)", after_one_3x3),
    ("两层3x3后 (感受野=5x5)", after_two_3x3),
]):
    im = axes[idx].imshow(data, cmap="YlOrRd", vmin=0)
    axes[idx].set_title(title, fontsize=11)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            axes[idx].text(j, i, f"{data[i,j]:.0f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes[idx], shrink=0.8)
plt.suptitle("VGGNet 洞察: 两层 3x3 = 一层 5x5 的感受野", fontsize=13)
plt.tight_layout(); plt.savefig("02_receptive_field.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 02_receptive_field.png\n")


# ====================================================================
# 第四部分：退化问题演示
# ====================================================================
print("=" * 70, "\n第四部分：退化问题 — 更深不一定更好\n" + "=" * 70)
# 普通深层网络越深，训练误差反而上升——不是过拟合，是优化失败


def train_plain_net(depth, n_iters=300, lr=0.01, dim=32):
    """训练 depth 层普通网络拟合 sin(x)，返回损失列表"""
    np.random.seed(7)
    X, y = np.linspace(-3, 3, 64).reshape(-1, 1), np.sin(np.linspace(-3, 3, 64)).reshape(-1, 1)
    ds = [1] + [dim] * depth + [1]
    W = [np.random.randn(ds[i], ds[i+1]) * np.sqrt(2.0/(ds[i]+ds[i+1])) for i in range(len(ds)-1)]
    losses = []
    for _ in range(n_iters):
        a = [X]; h = X
        for i, w in enumerate(W):
            h = h @ w; (h := np.maximum(0, h)) if i < len(W)-1 else None; a.append(h)
        losses.append(np.mean((h - y) ** 2))
        g = 2.0 * (h - y) / len(X)
        for i in range(len(W)-1, -1, -1):
            dw = a[i].T @ g
            if i > 0: g = (g @ W[i].T) * (a[i] > 0).astype(float)
            W[i] -= lr * np.clip(dw, -1.0, 1.0)
    return losses


depths = [3, 10, 30, 50]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
final = {}
for d in depths:
    ls = simulate_plain_network(d)
    axes[0].plot(ls, label=f"{d} 层", linewidth=1.5)
    final[d] = ls[-1]
    print(f"  深度 {d:>2d} 层: 最终损失 = {ls[-1]:.4f}")

axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("MSE 损失")
axes[0].set_title("退化问题: 普通网络越深, 训练反而更差")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, min(2.0, max(final.values()) * 1.2))

colors = ["#2ecc71" if final[d] < 0.2 else "#e74c3c" for d in depths]
axes[1].bar([f"{d}层" for d in depths], [final[d] for d in depths], color=colors)
axes[1].set_ylabel("最终训练损失"); axes[1].set_title("最终损失 vs 深度 (更深 != 更好!)")
axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig("02_degradation.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 02_degradation.png\n")


# ====================================================================
# 第五部分：残差块实现（前向 + 反向传播）
# ====================================================================
print("=" * 70, "\n第五部分：残差块 — ResNet 的核心组件\n" + "=" * 70)
# y = F(x) + x   ← 跳跃连接
# 反向: dy/dx = dF/dx + 1, 即使 dF/dx≈0 梯度至少还有 1!


class ResidualBlock:
    """残差块: y = ReLU(W2 @ ReLU(W1 @ x) + x), 用全连接层简化实现"""

    def __init__(self, dim):
        scale = np.sqrt(2.0 / dim)
        self.W1 = np.random.randn(dim, dim) * scale
        self.W2 = np.random.randn(dim, dim) * scale
        self.cache = {}

    def forward(self, x):
        """前向: y = ReLU(W2 @ ReLU(W1 @ x) + x)"""
        z1 = x @ self.W1;           a1 = np.maximum(0, z1)
        z2 = a1 @ self.W2;          y = np.maximum(0, z2 + x)  # 残差连接!
        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "y": y}
        return y

    def backward(self, dy):
        """反向传播, 返回 (dx, dW1, dW2)"""
        x, z1, a1, z2, y = (self.cache[k] for k in ["x", "z1", "a1", "z2", "y"])
        dy = dy * (y > 0).astype(float)       # 通过最外层 ReLU
        # 残差连接: 梯度同时流向 F(x) 路径和 skip 路径
        dz2 = dy; dx_skip = dy                 # 两条路径都拿到完整梯度
        dW2 = a1.T @ dz2
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).astype(float)    # 通过 ReLU
        dW1 = x.T @ dz1
        dx = dz1 @ self.W1.T + dx_skip        # 总梯度 = 变换路径 + 跳跃路径
        return dx, dW1, dW2


# 演示 + 梯度检查
block = ResidualBlock(dim=8)
x_demo = np.random.randn(4, 8)
y_demo = block.forward(x_demo)
print(f"  输入 {x_demo.shape} → 输出 {y_demo.shape}")

# 数值梯度检查
x_ck = np.random.randn(2, 8); blk = ResidualBlock(8)
_ = blk.forward(x_ck)
dx_a, _, _ = blk.backward(np.ones_like(x_ck))
eps = 1e-5; dx_n = np.zeros_like(x_ck)
for i in range(x_ck.shape[0]):
    for j in range(x_ck.shape[1]):
        xp, xm = x_ck.copy(), x_ck.copy()
        xp[i, j] += eps; xm[i, j] -= eps
        dx_n[i, j] = (blk.forward(xp).sum() - blk.forward(xm).sum()) / (2 * eps)
err = np.abs(dx_a - dx_n) / (np.abs(dx_a) + np.abs(dx_n) + 1e-8)
print(f"  梯度检查相对误差: {err.mean():.2e} ({'通过' if err.mean() < 1e-4 else '需检查'})\n")


# ====================================================================
# 第六部分：残差连接的梯度流可视化
# ====================================================================
print("=" * 70, "\n第六部分：梯度流可视化 — 跳跃连接为什么有效\n" + "=" * 70)


def gradient_flow(depth, use_residual, dim=16):
    """计算梯度通过 depth 层后各层的范数"""
    np.random.seed(0)
    weights = [np.random.randn(dim, dim) * np.sqrt(2.0 / dim) for _ in range(depth)]
    # 前向
    h = np.random.randn(1, dim); acts = []; pre = []
    for W in weights:
        z = h @ W; pre.append(z)
        a = np.maximum(0, z)
        if use_residual: a = a + h  # 跳跃连接
        h = a; acts.append(h)
    # 反向
    g = np.ones_like(h); norms = [np.linalg.norm(g)]
    for i in range(depth - 1, -1, -1):
        g_skip = g.copy() if use_residual else None
        g = g * (pre[i] > 0).astype(float) @ weights[i].T
        if use_residual: g = g + g_skip
        norms.append(np.linalg.norm(g))
    return list(reversed(norms))


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, depth in enumerate([10, 30, 50]):
    gp = gradient_flow(depth, False); gr = gradient_flow(depth, True)
    axes[idx].semilogy(gp, "r-o", ms=3, label="普通网络", lw=1.5, alpha=0.8)
    axes[idx].semilogy(gr, "b-s", ms=3, label="ResNet", lw=1.5, alpha=0.8)
    axes[idx].set_xlabel("层 (0=输入)"); axes[idx].set_ylabel("梯度范数 (log)")
    axes[idx].set_title(f"深度={depth}层"); axes[idx].legend(fontsize=9); axes[idx].grid(True, alpha=0.3)
plt.suptitle("残差连接防止梯度消失 (普通网络梯度在早期层衰减到~0)", fontsize=12, y=1.03)
plt.tight_layout(); plt.savefig("02_gradient_flow.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 02_gradient_flow.png")
print("  关键: dy/dx = dF/dx + 1, 即使 dF/dx->0 梯度仍 >= 1\n")


# ====================================================================
# 第七部分：简单 ResNet 从零搭建并训练
# ====================================================================
print("=" * 70, "\n第七部分：SimpleResNet — 从零搭建对比训练\n" + "=" * 70)


class SimpleResNet:
    """简单 ResNet: 输入映射 → N个残差块 → 线性输出"""

    def __init__(self, in_dim, hid_dim, out_dim, n_blocks=4):
        s_in = np.sqrt(2.0 / (in_dim + hid_dim))
        self.W_in = np.random.randn(in_dim, hid_dim) * s_in
        self.blocks = [ResidualBlock(hid_dim) for _ in range(n_blocks)]
        s_out = np.sqrt(2.0 / (hid_dim + out_dim))
        self.W_out = np.random.randn(hid_dim, out_dim) * s_out

    def forward(self, x):
        """输入→映射→残差块→输出"""
        self.x = x
        self.h_in = relu(x @ self.W_in)
        h = self.h_in
        for b in self.blocks: h = b.forward(h)
        self.h_last = h
        return h @ self.W_out

    def backward(self, d_out, lr=0.001):
        """反向传播并更新所有参数"""
        dh = d_out @ self.W_out.T
        self.W_out -= lr * np.clip(self.h_last.T @ d_out, -1, 1)
        for b in reversed(self.blocks):
            dh, dW1, dW2 = b.backward(dh)
            b.W1 -= lr * np.clip(dW1, -1, 1)
            b.W2 -= lr * np.clip(dW2, -1, 1)
        dh = dh * (self.h_in > 0).astype(float)
        self.W_in -= lr * np.clip(self.x.T @ dh, -1, 1)


# 训练对比: 普通深层网络 vs ResNet
np.random.seed(42)
X_tr = np.random.randn(128, 4)
y_tr = (np.sin(X_tr[:, 0:1]) + np.cos(X_tr[:, 1:2])) * 0.5
N_ITERS, N_BLOCKS, H_DIM = 500, 8, 16

# 普通深层网络 (等价深度，但无残差连接)
np.random.seed(0)
dims_p = [4] + [H_DIM] * (N_BLOCKS * 2 + 1) + [1]
pw = [np.random.randn(dims_p[i], dims_p[i+1]) * np.sqrt(2.0 / (dims_p[i] + dims_p[i+1]))
      for i in range(len(dims_p) - 1)]
plain_losses = []
for _ in range(N_ITERS):
    a = [X_tr]; h = X_tr
    for i, W in enumerate(pw):
        h = h @ W
        if i < len(pw) - 1: h = np.maximum(0, h)
        a.append(h)
    plain_losses.append(np.mean((h - y_tr) ** 2))
    g = 2.0 * (h - y_tr) / len(X_tr)
    for i in range(len(pw) - 1, -1, -1):
        dW = a[i].T @ g
        if i > 0: g = (g @ pw[i].T) * (a[i] > 0).astype(float)
        pw[i] -= 0.001 * np.clip(dW, -1, 1)

# ResNet
np.random.seed(0)
net = SimpleResNet(4, H_DIM, 1, N_BLOCKS)
res_losses = []
for _ in range(N_ITERS):
    pred = net.forward(X_tr)
    res_losses.append(np.mean((pred - y_tr) ** 2))
    net.backward(2.0 * (pred - y_tr) / len(X_tr), lr=0.001)

print(f"  普通网络 ({len(pw)}层): 最终损失 = {plain_losses[-1]:.4f}")
print(f"  ResNet  ({N_BLOCKS}残差块): 最终损失 = {res_losses[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(plain_losses, label=f"普通网络 ({len(pw)}层)", color="tomato", lw=1.5)
axes[0].plot(res_losses, label=f"ResNet ({N_BLOCKS}残差块)", color="steelblue", lw=1.5)
axes[0].set_xlabel("迭代次数"); axes[0].set_ylabel("MSE 损失")
axes[0].set_title("训练损失: 普通网络 vs ResNet"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 右图: 残差块结构示意图
ax = axes[1]; ax.set_xlim(0, 10); ax.set_ylim(3, 10); ax.set_aspect("equal")
for cx, fc, ec, label, out_label in [
    (2, "#ffcccc", "tomato", "普通块\ny=F(x)", "y=F(x)"),
    (7, "#cce5ff", "steelblue", "残差块\nF(x)", "y=F(x)+x"),
]:
    ax.add_patch(plt.Rectangle((cx-1.5, 6), 3, 3, fc=fc, ec=ec, lw=2))
    ax.text(cx, 7.5, label, ha="center", va="center", fontsize=11)
    ax.annotate("x", xy=(cx, 9.5), fontsize=12, ha="center", fontweight="bold")
    ax.annotate("", xy=(cx, 9), xytext=(cx, 9.8), arrowprops=dict(arrowstyle="->"))
    ax.annotate(out_label, xy=(cx, 5.3), fontsize=10, ha="center", color=ec)
    ax.annotate("", xy=(cx, 5.0), xytext=(cx, 6), arrowprops=dict(arrowstyle="->", color=ec))
# 跳跃连接箭头
ax.annotate("", xy=(9.2, 5.0), xytext=(9.2, 9.5),
            arrowprops=dict(arrowstyle="->", color="green", lw=2, connectionstyle="arc3,rad=-0.3"))
ax.text(9.8, 7.5, "+x", fontsize=13, color="green", fontweight="bold")
ax.set_title("残差块 vs 普通块", fontsize=12); ax.axis("off")
plt.suptitle("ResNet: 跳跃连接让深层网络可训练", fontsize=13, y=1.02)
plt.tight_layout(); plt.savefig("02_resnet_comparison.png", dpi=100, bbox_inches="tight")
plt.show(); print("[已保存] 02_resnet_comparison.png\n")


# ====================================================================
# 第八部分：思考题
# ====================================================================
print("=" * 70, "\n第八部分：思考题\n" + "=" * 70)
print("""
1.【残差连接与恒等映射】
  y = F(x) + x。如果某层"什么都不做"是最优的, F(x) 应该学成什么?
  为什么这比让普通网络直接学 y = x 容易得多?
  提示: 学 F(x)=0 只需所有权重为0, 但学 y=x 需要精确的恒等矩阵。

2.【感受野计算】
  n 层 3x3 卷积的感受野 = (2n+1)x(2n+1)。
  要达到 11x11 感受野需要几层 3x3? 比一层 11x11 节省多少参数?

3.【退化 vs 过拟合】
  两者都表现为"效果差", 但本质不同。如何区分?
  提示: 过拟合=训练好测试差 (记住噪声); 退化=训练本身就差 (优化失败)。

4.【Transformer 中的残差】
  Transformer 每层: output = LayerNorm(x + SubLayer(x))
  去掉残差连接会怎样? 为什么还要加 LayerNorm?
  提示: BERT有12层, GPT-3有96层, 没有残差连接梯度会怎样?

5.【1x1 卷积】
  GoogLeNet 大量使用 1x1 卷积。只看一个像素的卷积有什么用?
  提示: 它对通道维度做了什么? (跨通道信息融合 + 降维减少计算量)
""")

# ====================================================================
# 总结
# ====================================================================
print("=" * 70, "\n总结\n" + "=" * 70)
print("""
  1. 架构演进: LeNet→AlexNet→VGG→ResNet, 核心趋势是"更深"
  2. VGG 洞察: 多层3x3 = 同样感受野 + 更少参数 + 更多非线性
  3. 退化问题: 深层普通网络训练误差反而更高, 是优化失败而非过拟合
  4. ResNet 残差连接: y = F(x) + x → 梯度至少为1, 解决梯度消失
  5. 深远影响: Transformer 每一层都有残差连接, 直接继承自 ResNet

  下一节预告: 第3章 · 第3节 · 经典 CNN 实战
""")
