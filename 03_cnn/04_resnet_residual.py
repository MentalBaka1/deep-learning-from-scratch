"""
==============================================================
第3章 第4节：残差网络（ResNet）—— 梯度高速公路
==============================================================

【为什么需要它？】
理论上，网络越深表达能力越强。
但实际发现：
  - 56层网络 比 20层网络 表现更差！（不是过拟合，是训练更难）
  - 梯度消失：深层的梯度在反向传播中被层层衰减
  - 退化问题（Degradation Problem）：深层网络甚至比浅层还差

2015年，何恺明等提出 ResNet：
  引入"跳跃连接（skip connection）"，让梯度有高速公路直达！

【生活类比】
普通深层网络 = 只能走山路（梯度要爬过每一层）
ResNet = 山路旁还有高速公路（梯度可以直接"跳过"几层）

  普通：y = F(x)          （信息只经过卷积/激活）
  残差：y = F(x) + x      （信息还有一条"直通"路线）

直觉：
  如果某几层什么都不学（F(x)≈0），网络仍然相当于恒等映射 y=x
  让网络学习"残差"F(x) = y - x（比直接学 y 更容易！）

【存在理由】
解决问题：深层网络训练时梯度消失，越深反而越差
核心思想：跳跃连接提供梯度高速公路，使任意深的网络都能有效训练
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 问题重现 —— 深层网络的梯度消失
# ============================================================
print("=" * 50)
print("Part 1: 问题重现 —— 深层网络的梯度幅度")
print("=" * 50)

"""
在没有残差连接的深层网络中，梯度是这样反向传播的：
  dL/dW_1 = dL/dL * 激活_n' * W_n * 激活_{n-1}' * W_{n-1} * ... * 激活_1'

每一层都要乘以：权重矩阵 × 激活函数导数
  - Sigmoid 导数 ≤ 0.25
  - 即使 ReLU，也会有一些死神经元导致梯度为0

让我们模拟不同层数时，第一层的梯度幅度。
"""

def simulate_gradient_flow(n_layers, activation='relu', use_residual=False):
    """
    模拟梯度从输出层反传到第一层的过程
    返回每层的梯度幅度
    """
    # 随机初始化网络（小型，用于模拟）
    np.random.seed(0)
    hidden_size = 64
    layers = []
    for i in range(n_layers):
        W = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        layers.append(W)

    # 随机输入
    x = np.random.randn(1, hidden_size)

    # 前向传播，保存每层激活值
    activations = [x]
    for i, W in enumerate(layers):
        z = activations[-1] @ W
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            a = np.maximum(0, z)
        else:
            a = z

        if use_residual and i > 0:
            a = a + activations[-1]  # 残差连接

        activations.append(a)

    # 反向传播，追踪梯度幅度
    d = np.ones((1, hidden_size))  # 从输出开始
    grad_norms = [np.linalg.norm(d)]

    for i in reversed(range(n_layers)):
        act = activations[i + 1]

        if activation == 'sigmoid':
            d_act = act * (1 - act)
        elif activation == 'relu':
            d_act = (activations[i + 1] > 0).astype(float)
        else:
            d_act = np.ones_like(act)

        d = d * d_act @ layers[i].T

        if use_residual and i > 0:
            d = d + np.ones_like(d)  # 残差的梯度直接流过

        grad_norms.append(np.linalg.norm(d))

    return list(reversed(grad_norms))

# 对比：普通网络 vs 残差网络
n_layers_list = [5, 10, 20]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('梯度流动：普通网络 vs 残差网络', fontsize=14)

for ax, n_layers in zip(axes, n_layers_list):
    norms_sigmoid = simulate_gradient_flow(n_layers, 'sigmoid', use_residual=False)
    norms_relu = simulate_gradient_flow(n_layers, 'relu', use_residual=False)
    norms_residual = simulate_gradient_flow(n_layers, 'relu', use_residual=True)

    layer_idx = list(range(n_layers + 1))
    ax.semilogy(layer_idx, norms_sigmoid, 'r-o', markersize=4, label='Sigmoid（无残差）')
    ax.semilogy(layer_idx, norms_relu, 'b-s', markersize=4, label='ReLU（无残差）')
    ax.semilogy(layer_idx, norms_residual, 'g-^', markersize=4, label='ReLU + 残差连接')

    ax.set_xlabel('从输出到输入的层数')
    ax.set_ylabel('梯度幅度（对数）')
    ax.set_title(f'{n_layers}层网络的梯度流动')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # 从输出（左）到输入（右）

plt.tight_layout()
plt.savefig('03_cnn/resnet_gradient_flow.png', dpi=100, bbox_inches='tight')
print("图片已保存：03_cnn/resnet_gradient_flow.png")
plt.show()

# ============================================================
# Part 2: 残差块实现
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 残差块（Residual Block）实现")
print("=" * 50)

"""
残差块的计算：
  z1 = BN(x @ W1 + b1)
  a1 = ReLU(z1)
  z2 = BN(a1 @ W2 + b2)
  output = ReLU(z2 + x)  ← 这里的 +x 就是跳跃连接！

注意：x 和 z2 的维度必须相同！
如果维度不同（如通道数改变），用 1×1 卷积调整 x 的维度
（称为"投影跳跃连接"）

前向传播时：
  梯度经过 +x 时，会直接"绕过"这两层传递
  → 梯度不会衰减太多！
"""

class ResidualBlock:
    """
    基础残差块（全连接版本，为了简单）
    实际 ResNet 用卷积，原理一样
    """
    def __init__(self, dim):
        # He 初始化
        self.W1 = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.b1 = np.zeros(dim)
        self.W2 = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.b2 = np.zeros(dim)
        self.cache = {}

    def forward(self, x):
        """
        残差块前向：output = ReLU(W2 * ReLU(W1*x + b1) + b2 + x)
        """
        # 第一层
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)

        # 第二层
        z2 = a1 @ self.W2 + self.b2

        # 残差连接：z2 + x（跳跃！）
        output = np.maximum(0, z2 + x)

        self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'output': output}
        return output

    def backward(self, d_out):
        """
        残差块反向：梯度在残差连接处分叉
        """
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']

        # ReLU(z2 + x) 的梯度
        d_relu_out = d_out * ((z2 + x) > 0)

        # 残差连接：梯度分成两路
        # 路1：经过 W2 → a1 → W1 → x（正常反传）
        # 路2：直接传给 x（跳跃连接！这是关键）
        d_z2 = d_relu_out  # 第二层的梯度
        d_x_skip = d_relu_out  # 跳跃连接直接给 x 的梯度

        # W2 的梯度
        self.dW2 = a1.T @ d_z2
        self.db2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ self.W2.T

        # ReLU 1 的梯度
        d_z1 = d_a1 * (z1 > 0)

        # W1 的梯度
        self.dW1 = x.T @ d_z1
        self.db1 = d_z1.sum(axis=0)
        d_x_main = d_z1 @ self.W1.T

        # 两路梯度相加（这就是为什么残差连接能传梯度！）
        d_x = d_x_main + d_x_skip

        return d_x

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

# ============================================================
# Part 3: 数值梯度验证残差块
# ============================================================
print("Part 3: 验证残差块的梯度实现")
print("=" * 50)

def gradient_check_resblock(block, x, eps=1e-5):
    """数值梯度检验残差块的输入梯度"""
    # 前向 + 反向（解析梯度）
    out = block.forward(x)
    d_out = np.ones_like(out)  # 简单损失 L = sum(out)
    d_x_analytical = block.backward(d_out)

    # 数值梯度
    d_x_numerical = np.zeros_like(x)
    for i in range(x.size):
        x_flat = x.ravel().copy()

        x_plus = x_flat.copy(); x_plus[i] += eps
        out_plus = block.forward(x_plus.reshape(x.shape))
        loss_plus = np.sum(out_plus)

        x_minus = x_flat.copy(); x_minus[i] -= eps
        out_minus = block.forward(x_minus.reshape(x.shape))
        loss_minus = np.sum(out_minus)

        d_x_numerical.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)

    rel_error = np.max(np.abs(d_x_analytical - d_x_numerical) /
                       (np.abs(d_x_analytical) + np.abs(d_x_numerical) + 1e-8))

    print(f"  残差块梯度相对误差：{rel_error:.2e}  {'✓ 正确' if rel_error < 1e-4 else '✗ 有bug'}")
    return rel_error

block = ResidualBlock(dim=8)
x_test = np.random.randn(3, 8)
gradient_check_resblock(block, x_test)

# ============================================================
# Part 4: 有无残差连接的训练对比
# ============================================================
print("\nPart 4: 有无残差连接的训练对比（20层网络）")
print("=" * 50)

def make_spiral(n=100, classes=3):
    X, y = [], []
    for c in range(classes):
        t = np.linspace(0, 1, n)
        angle = t * 4 * np.pi + 2*np.pi*c/classes
        X.append(np.column_stack([t*np.cos(angle), t*np.sin(angle)]))
        y.extend([c] * n)
    X = np.vstack(X) + np.random.randn(n*classes, 2) * 0.1
    X = (X - X.mean(0)) / X.std(0)
    return X, np.array(y)

X_train, y_train = make_spiral(n=100, classes=3)

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def train_network(use_residual, n_hidden_layers=10, dim=32, n_epochs=500, lr=0.01):
    """训练有/无残差连接的深层网络"""
    np.random.seed(42)
    n_classes = 3
    n_features = 2

    # 输入层
    W_in = np.random.randn(n_features, dim) * np.sqrt(2.0/n_features)
    b_in = np.zeros(dim)

    # 隐藏层
    blocks = [ResidualBlock(dim) for _ in range(n_hidden_layers)]

    # 输出层
    W_out = np.random.randn(dim, n_classes) * np.sqrt(2.0/dim)
    b_out = np.zeros(n_classes)

    losses = []
    accs = []

    for epoch in range(n_epochs):
        # 前向
        x = np.maximum(0, X_train @ W_in + b_in)
        hidden_acts = [x]

        for block in blocks:
            if use_residual:
                x = block.forward(x)
            else:
                # 无残差：只用普通前向
                z1 = x @ block.W1 + block.b1
                a1 = np.maximum(0, z1)
                z2 = a1 @ block.W2 + block.b2
                x = np.maximum(0, z2)
                block.cache = {'x': hidden_acts[-1], 'z1': z1, 'a1': a1, 'z2': z2, 'output': x}
            hidden_acts.append(x)

        logits = x @ W_out + b_out
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(len(y_train)), y_train] + 1e-8))
        losses.append(loss)

        acc = np.mean(np.argmax(probs, axis=1) == y_train)
        accs.append(acc)

        # 反向
        d = probs.copy()
        d[np.arange(len(y_train)), y_train] -= 1
        d /= len(y_train)

        dW_out = x.T @ d
        db_out = d.sum(0)
        d = d @ W_out.T

        for block in reversed(blocks):
            if use_residual:
                d = block.backward(d)
            else:
                # 无残差反向
                xb = block.cache['x']
                z1, a1, z2 = block.cache['z1'], block.cache['a1'], block.cache['z2']
                d = d * (block.cache['output'] > 0)
                block.dW2 = a1.T @ d
                block.db2 = d.sum(0)
                d = d @ block.W2.T * (z1 > 0)
                block.dW1 = xb.T @ d
                block.db1 = d.sum(0)
                d = d @ block.W1.T

            block.update(lr)

        d_in = d * (X_train @ W_in + b_in > 0)
        W_in -= lr * X_train.T @ d_in
        b_in -= lr * d_in.sum(0)
        W_out -= lr * dW_out
        b_out -= lr * db_out

    return losses, accs

print("训练普通深层网络（10个隐藏层，无残差）...")
losses_plain, accs_plain = train_network(use_residual=False, n_hidden_layers=10)

print("训练残差网络（10个残差块）...")
losses_resnet, accs_resnet = train_network(use_residual=True, n_hidden_layers=10)

print(f"\n最终结果（500个epoch）：")
print(f"  普通网络：loss={losses_plain[-1]:.4f}, acc={accs_plain[-1]:.2%}")
print(f"  残差网络：loss={losses_resnet[-1]:.4f}, acc={accs_resnet[-1]:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.plot(losses_plain, 'r-', linewidth=2, label='普通深层网络（10层）', alpha=0.8)
ax.plot(losses_resnet, 'g-', linewidth=2, label='残差网络（10个残差块）', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('有无残差连接的收敛对比')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(accs_plain, 'r-', linewidth=2, label='普通深层网络', alpha=0.8)
ax.plot(accs_resnet, 'g-', linewidth=2, label='残差网络', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('训练准确率')
ax.set_title('有无残差连接的准确率对比')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_cnn/resnet_comparison.png', dpi=100, bbox_inches='tight')
print("\n图片已保存：03_cnn/resnet_comparison.png")
plt.show()

# ============================================================
# Part 5: 残差连接的数学直觉
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 残差连接的深层理解")
print("=" * 50)

print("""
【为什么残差连接有效？】

1. 梯度高速公路：
   反向传播时，梯度 = ∂L/∂output * ∂output/∂input
   在残差块中：∂output/∂input = ∂F(x)/∂x + 1
   即使 ∂F(x)/∂x ≈ 0（很小），梯度仍至少为 1！
   → 梯度不会消失到0。

2. 学习残差更容易：
   普通网络学习 H(x)（目标函数）
   残差网络学习 F(x) = H(x) - x（残差）
   如果 H(x) ≈ x（恒等映射），则 F(x) ≈ 0（零函数更容易学）

3. 集成模型视角：
   ResNet 可以看作是很多"浅层网络"的集成
   每条不同长度的跳跃路径对应一个不同深度的模型
   → 类似随机森林：多个模型的集成

4. 现实意义：
   ResNet 让 100层+ 的网络成为可能（ResNet-152！）
   在 ImageNet 上首次超过人类水平
   残差思想被广泛采用：DenseNet、Transformer的残差连接等
""")

# ============================================================
# 思考题
# ============================================================
print("=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【维度不匹配的投影连接】
   如果残差块的输入维度（如16）和输出维度（如32）不同，
   直接相加 y = F(x) + x 会失败（维度不匹配）。
   ResNet 的解决方法是"1×1卷积投影"：
     y = F(x) + W_proj @ x
   实现这个 ProjectionResidualBlock，让输入32维→输出64维。
   验证：forward 能运行，backward 的梯度检验通过。

2. 【梯度幅度实验】
   修改 simulate_gradient_flow 函数，记录每层的梯度幅度。
   在 50层网络上：
   - 普通 ReLU 网络：第1层的梯度是第25层的多少倍？
   - 残差 ReLU 网络：同样的比较
   结果说明了什么？

3. 【BottleNeck 结构】
   大型 ResNet（ResNet-50、ResNet-101）用 BottleNeck 残差块：
   1×1 Conv（降维，如 256→64） → 3×3 Conv → 1×1 Conv（升维，64→256）
   参数数量：1×1×256×64 + 3×3×64×64 + 1×1×64×256 = ?
   对比普通残差块：3×3×256×256 + 3×3×256×256 = ?
   BottleNeck 能节省多少参数？
""")
