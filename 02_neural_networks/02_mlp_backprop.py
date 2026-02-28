"""
==============================================================
第2章 第2节：MLP + 手写反向传播 —— 神经网络的核心
==============================================================

【为什么需要它？】
感知机只有一层，解不了 XOR 等非线性问题。
多层感知机（MLP）通过堆叠多层，可以学习任意复杂的函数。
关键：如何训练多层网络？—— 反向传播算法（Backpropagation）

【生活类比】
工厂流水线：原料 → 工序1 → 工序2 → ... → 产品
产品有质量问题（预测错了），要追责：
  检验员先找到最后一道工序的责任（容易算）
  再追查上一道工序的责任（链式法则！）
  一路追到最初的原料来源

这就是反向传播：从输出误差反向追责，找出每个参数的"责任"（梯度）。

【存在理由】
解决问题：多层网络的参数如何高效训练？
核心思想：链式法则 + 计算图，从输出反向计算每个参数的梯度
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 激活函数（带反向传播）
# ============================================================
print("=" * 50)
print("Part 1: 激活函数及其反向传播")
print("=" * 50)

class Sigmoid:
    def forward(self, x):
        """前向：保存输出供反向传播使用"""
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output

    def backward(self, d_out):
        """
        反向：sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = s * (1 - s)
        注意：我们用缓存的 self.output（不需要重新算 sigmoid）
        """
        return d_out * self.output * (1 - self.output)

class ReLU:
    def forward(self, x):
        """前向：保存输入的正负掩码"""
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, d_out):
        """
        反向：relu'(x) = 1 if x > 0 else 0
        梯度只能通过正数区域，负数区域梯度为0
        """
        return d_out * self.mask

class Linear:
    """全连接层：y = xW + b"""

    def __init__(self, in_features, out_features):
        # He 初始化：对 ReLU 很重要
        # 原理：若上层有 n 个输入，权重方差为 2/n，输出方差≈1
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dW = None
        self.db = None

    def forward(self, x):
        """y = xW + b，缓存输入"""
        self.x = x
        return x @ self.W + self.b

    def backward(self, d_out):
        """
        链式法则推导：
          y = xW + b

          dL/dW = x.T @ d_out  （每个样本的 x 和 d_out 做外积，然后平均）
          dL/db = sum(d_out, axis=0)  （b 对每个样本都有贡献，累加）
          dL/dx = d_out @ W.T  （传给上一层的梯度）

        直觉：
          W 的梯度 = "输入x" × "输出误差d_out"
          x 的梯度 = "误差d_out" × "权重W"（反向传播！）
        """
        self.dW = self.x.T @ d_out  # 形状：(in_features, out_features)
        self.db = np.sum(d_out, axis=0)  # 形状：(out_features,)
        return d_out @ self.W.T  # 传给上一层，形状：(batch, in_features)

class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵组合（合并计算，梯度更简洁）

    前向：
      Softmax: p_k = exp(z_k) / Σ exp(z_j)
      Loss: L = -Σ y_true_k * log(p_k)

    反向（推导后非常优雅）：
      dL/dz_k = p_k - y_true_k
      即：预测概率 - 真实概率，就这么简单！
    """

    def forward(self, logits, y_true):
        """
        logits: 原始分数，shape (batch, n_classes)
        y_true: 真实类别整数，shape (batch,)
        """
        # 数值稳定：减去每行最大值（不影响 softmax 结果，但防止溢出）
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.probs = exp_z / exp_z.sum(axis=1, keepdims=True)

        self.y_true = y_true
        n = len(y_true)
        # 只取正确类别的 log(p)
        correct_log_probs = -np.log(self.probs[np.arange(n), y_true] + 1e-8)
        return np.mean(correct_log_probs)

    def backward(self):
        """
        梯度：dL/d_logits = (probs - one_hot(y_true)) / batch_size
        """
        n = len(self.y_true)
        d_logits = self.probs.copy()
        d_logits[np.arange(n), self.y_true] -= 1  # 正确类别的梯度 = p - 1
        d_logits /= n  # 除以 batch_size（均值的梯度）
        return d_logits

# ============================================================
# Part 2: 数值梯度检验 —— 验证反向传播正确性
# ============================================================
print("Part 2: 梯度检验 —— 验证 backward() 正确性")
print("=" * 50)

"""
数值梯度检验：
  ∂L/∂w ≈ [L(w + ε) - L(w - ε)] / (2ε)

如果和解析梯度（backward 计算的）误差 < 1e-5，说明实现正确。
这是实现新层时必做的验证！
"""

def gradient_check(layer, x, loss_fn, eps=1e-5):
    """
    对层的所有参数做数值梯度检验
    """
    print(f"  检验 {layer.__class__.__name__}...")

    # 先做一次前向 + 反向得到解析梯度
    out = layer.forward(x)
    # 用一个简单损失：L = sum(out)（dL/dout = 1）
    analytical_d_out = np.ones_like(out)
    analytical_d_x = layer.backward(analytical_d_out)

    # 数值梯度
    numerical_d_x = np.zeros_like(x)
    for i in range(x.size):
        x_flat = x.ravel()

        x_flat_plus = x.ravel().copy()
        x_flat_plus[i] += eps
        x_plus = x_flat_plus.reshape(x.shape)
        out_plus = layer.forward(x_plus)
        loss_plus = np.sum(out_plus)

        x_flat_minus = x.ravel().copy()
        x_flat_minus[i] -= eps
        x_minus = x_flat_minus.reshape(x.shape)
        out_minus = layer.forward(x_minus)
        loss_minus = np.sum(out_minus)

        numerical_d_x.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)

    layer.forward(x)  # 恢复正常前向状态

    rel_error = np.max(np.abs(analytical_d_x - numerical_d_x) /
                       (np.abs(analytical_d_x) + np.abs(numerical_d_x) + 1e-8))
    status = "✓" if rel_error < 1e-5 else "✗ 可能有bug！"
    print(f"    相对误差：{rel_error:.2e}  {status}")
    return rel_error

# 测试每一层
x_test = np.random.randn(4, 3)
print("各层梯度检验（误差应 < 1e-5）：")
for LayerClass, args in [(Sigmoid, []), (ReLU, []), (Linear, [3, 5])]:
    if args:
        layer = LayerClass(*args)
    else:
        layer = LayerClass()
    gradient_check(layer, x_test, None)

# ============================================================
# Part 3: 组装 MLP
# ============================================================
print("\nPart 3: 组装多层感知机（MLP）")
print("=" * 50)

class MLP:
    """
    多层感知机：任意层数，支持完整的前向+反向传播
    结构：Linear -> ReLU -> Linear -> ReLU -> ... -> Linear -> Softmax
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes: 如 [2, 8, 8, 3] 表示
          输入2维 → 隐层8 → 隐层8 → 输出3类
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # 最后一层不加激活
                self.layers.append(ReLU())

        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x, y=None):
        """前向传播，如果提供 y 则同时计算损失"""
        for layer in self.layers:
            x = layer.forward(x)

        if y is not None:
            loss = self.loss_fn.forward(x, y)
            return x, loss
        return x

    def backward(self):
        """从损失反向传播到第一层"""
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def update(self, lr):
        """更新所有 Linear 层的参数"""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def accuracy(self, x, y):
        return np.mean(self.predict(x) == y)

# ============================================================
# Part 4: 解决 XOR 问题！
# ============================================================
print("Part 4: MLP 解决 XOR 问题")
print("=" * 50)

"""
XOR 是感知机无法解决的，让我们用两层 MLP 征服它！
"""

# XOR 数据
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])  # 0/1 表示

# 用 2→4→2 的网络解决 XOR
mlp_xor = MLP([2, 4, 2])

print("训练 MLP 解决 XOR（2→4→2）：")
losses = []
for epoch in range(3000):
    _, loss = mlp_xor.forward(X_xor, y_xor)
    losses.append(loss)
    mlp_xor.backward()
    mlp_xor.update(lr=0.05)

    if epoch % 500 == 0:
        acc = mlp_xor.accuracy(X_xor, y_xor)
        print(f"  Epoch {epoch:5d}: Loss={loss:.4f}, Acc={acc:.2%}")

print(f"\nMLP 在 XOR 上的最终准确率：{mlp_xor.accuracy(X_xor, y_xor):.2%}")
preds = mlp_xor.predict(X_xor)
for x, pred, true in zip(X_xor, preds, y_xor):
    print(f"  x={x.astype(int)}, 预测={pred}, 真实={true}, {'✓' if pred == true else '✗'}")

# ============================================================
# Part 5: 在真实数据集上训练
# ============================================================
print("\nPart 5: 在螺旋数据集上训练")
print("=" * 50)

# 生成螺旋数据集（经典非线性测试）
def make_spiral(n_per_class=100, n_classes=3):
    X, y = [], []
    for c in range(n_classes):
        t = np.linspace(0, 1, n_per_class)
        r = t
        angle = t * 4 * np.pi + (2 * np.pi * c / n_classes)
        x1 = r * np.sin(angle) + np.random.randn(n_per_class) * 0.1
        x2 = r * np.cos(angle) + np.random.randn(n_per_class) * 0.1
        X.append(np.column_stack([x1, x2]))
        y.extend([c] * n_per_class)
    return np.vstack(X), np.array(y)

X_spiral, y_spiral = make_spiral(n_per_class=100, n_classes=3)

# 标准化
X_mean = X_spiral.mean(axis=0)
X_std = X_spiral.std(axis=0)
X_norm = (X_spiral - X_mean) / X_std

# 训练 MLP（2→64→64→3）
mlp = MLP([2, 64, 64, 3])

losses_spiral = []
print("训练 MLP 在螺旋数据集（2→64→64→3）：")
for epoch in range(3000):
    _, loss = mlp.forward(X_norm, y_spiral)
    losses_spiral.append(loss)
    mlp.backward()
    mlp.update(lr=0.01)

    if epoch % 500 == 0:
        acc = mlp.accuracy(X_norm, y_spiral)
        print(f"  Epoch {epoch:5d}: Loss={loss:.4f}, Acc={acc:.2%}")

final_acc = mlp.accuracy(X_norm, y_spiral)
print(f"\n最终准确率：{final_acc:.2%}")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左：XOR 学习曲线
ax = axes[0]
ax.plot(losses, 'b-', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('XOR 训练曲线')
ax.grid(True, alpha=0.3)

# 中：螺旋数据集学习曲线
ax = axes[1]
ax.plot(losses_spiral, 'r-', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('螺旋数据集训练曲线')
ax.grid(True, alpha=0.3)

# 右：螺旋数据集决策边界
ax = axes[2]
h = 0.05
xx, yy = np.meshgrid(np.arange(-2, 2, h), np.arange(-2, 2, h))
X_grid = np.column_stack([xx.ravel(), yy.ravel()])
Z = mlp.predict(X_grid).reshape(xx.shape)

colors_bg = ['#FFCCCC', '#CCFFCC', '#CCCCFF']
ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5],
           colors=colors_bg, alpha=0.6)
colors_pts = ['red', 'green', 'blue']
for c in range(3):
    mask = y_spiral == c
    ax.scatter(X_norm[mask, 0], X_norm[mask, 1],
              c=colors_pts[c], s=20, alpha=0.8)
ax.set_title(f'MLP 决策边界\n准确率：{final_acc:.2%}')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_neural_networks/mlp_backprop.png', dpi=100, bbox_inches='tight')
print("\n图片已保存：02_neural_networks/mlp_backprop.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【梯度检验】
   为 SoftmaxCrossEntropy 实现梯度检验：
   - 随机生成 logits (batch=3, n_classes=4) 和 y_true
   - 前向计算 loss，反向计算 d_logits
   - 对每个 logits[i,j]，分别用 +ε 和 -ε 计算数值梯度
   - 验证相对误差 < 1e-5

2. 【网络宽度实验】
   在螺旋数据集上，分别试用：
   [2, 8, 3], [2, 32, 3], [2, 128, 3], [2, 32, 32, 3]
   哪个网络收敛最快？哪个最终准确率最高？

3. 【梯度流动】
   在 MLP 的 backward() 之后，检查每一层 Linear.dW 的范数（np.linalg.norm）。
   观察：从输出层到输入层，梯度范数是否在减小？
   如果用 Sigmoid 替换 ReLU，情况会更严重吗？（梯度消失！）
""")
