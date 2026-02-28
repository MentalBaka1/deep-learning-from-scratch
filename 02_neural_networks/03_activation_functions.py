"""
==============================================================
第2章 第3节：激活函数大全 —— 为什么需要非线性？
==============================================================

【为什么需要它？】
如果神经网络所有层都是线性的：
  输出 = W3 * (W2 * (W1 * x + b1) + b2) + b3
       = (W3*W2*W1) * x + 常数
  = 一个线性函数！

多层线性层 = 一层线性层，白搭了！

激活函数引入非线性，让神经网络能学习任意复杂的函数。

【存在理由】
解决问题：多层线性网络等价于单层线性网络（表达能力无提升）
核心思想：在每层之后加非线性变换，打破线性叠加
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)

# ============================================================
# Part 1: 各激活函数及其导数
# ============================================================
print("=" * 50)
print("Part 1: 常见激活函数详解")
print("=" * 50)

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

def gelu(x):
    """GELU：BERT、GPT等现代模型使用"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x):
    """Swish = x * sigmoid(x)，Google提出"""
    return x * sigmoid(x)

# ============================================================
# Part 2: 可视化所有激活函数
# ============================================================
activations = [
    ('Sigmoid\n1/(1+e^{-x})',    sigmoid,    sigmoid_deriv,
     '输出[0,1]\n梯度消失严重\n适合二分类输出'),
    ('Tanh\n(e^x-e^{-x})/(e^x+e^{-x})', tanh, tanh_deriv,
     '输出[-1,1]，零中心\n梯度消失（比Sigmoid好）\n比Sigmoid更常用于隐层'),
    ('ReLU\nmax(0,x)',            relu,       relu_deriv,
     '计算最快\n无梯度消失（正区间）\n"死ReLU"问题（负区间永远0）'),
    ('Leaky ReLU\nmax(0.01x,x)', leaky_relu, leaky_relu_deriv,
     '解决死ReLU\nalpha=0.01让负区间有小梯度\n实用性强'),
    ('ELU\n指数线性单元',         elu,        elu_deriv,
     '负区间平滑（指数）\n输出更接近零均值\n但计算稍慢'),
    ('GELU\n高斯误差线性',        gelu,       None,
     'BERT/GPT3等使用\n平滑版ReLU\n性能略好于ReLU'),
]

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
fig.suptitle('激活函数大全：函数曲线（蓝）和导数曲线（红）', fontsize=14)

for i, (name, func, deriv, desc) in enumerate(activations):
    ax_func = axes[0][i]
    ax_deriv = axes[1][i]

    y = func(x)
    ax_func.plot(x, y, 'b-', linewidth=2.5)
    ax_func.axhline(0, color='gray', linewidth=0.5)
    ax_func.axvline(0, color='gray', linewidth=0.5)
    ax_func.set_title(name, fontsize=10)
    ax_func.set_ylim(-2, 3)
    ax_func.grid(True, alpha=0.3)
    ax_func.text(0.02, 0.02, desc, transform=ax_func.transAxes,
                fontsize=7, va='bottom', color='green',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    if deriv is not None:
        dy = deriv(x)
        ax_deriv.plot(x, dy, 'r-', linewidth=2.5, label="导数")
        max_grad = np.max(dy)
        ax_deriv.set_title(f'导数（最大值≈{max_grad:.2f}）', fontsize=9)
    else:
        # 数值导数
        dy = np.gradient(func(x), x)
        ax_deriv.plot(x, dy, 'r--', linewidth=2.5, label="数值导数")
        ax_deriv.set_title('导数（数值近似）', fontsize=9)

    ax_deriv.axhline(0, color='gray', linewidth=0.5)
    ax_deriv.set_ylim(-0.1, 1.2)
    ax_deriv.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_neural_networks/activation_functions.png', dpi=80, bbox_inches='tight')
print("图片已保存：02_neural_networks/activation_functions.png")
plt.show()

# ============================================================
# Part 3: 为什么需要非线性 —— 实验证明
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 没有非线性激活 = 白搭多层（实验）")
print("=" * 50)

"""
实验：在 XOR 数据集上
  - 网络A：Linear → Linear（无激活，两层线性）
  - 网络B：Linear → ReLU → Linear（有激活）

理论预测：A 学不会 XOR（等价于单层线性），B 可以
"""

def sigmoid_np(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

def train_and_eval(use_activation, n_epochs=5000, lr=0.01):
    """训练一个简单网络（有/无激活函数）"""
    np.random.seed(123)
    W1 = np.random.randn(2, 4) * 0.1
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 2) * 0.1
    b2 = np.zeros(2)
    losses = []

    for _ in range(n_epochs):
        # 前向
        z1 = X_xor @ W1 + b1
        if use_activation:
            a1 = np.maximum(0, z1)   # ReLU
        else:
            a1 = z1                   # 无激活（线性）

        z2 = a1 @ W2 + b2
        exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)

        n = len(y_xor)
        loss = -np.mean(np.log(probs[np.arange(n), y_xor] + 1e-8))
        losses.append(loss)

        # 反向
        d_z2 = probs.copy()
        d_z2[np.arange(n), y_xor] -= 1
        d_z2 /= n

        dW2 = a1.T @ d_z2
        db2 = d_z2.sum(axis=0)
        d_a1 = d_z2 @ W2.T

        if use_activation:
            d_z1 = d_a1 * (z1 > 0)   # ReLU 反向
        else:
            d_z1 = d_a1

        dW1 = X_xor.T @ d_z1
        db1 = d_z1.sum(axis=0)

        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    # 最终准确率
    z1 = X_xor @ W1 + b1
    a1 = np.maximum(0, z1) if use_activation else z1
    z2 = a1 @ W2 + b2
    preds = np.argmax(z2, axis=1)
    acc = np.mean(preds == y_xor)

    return losses, acc

losses_linear, acc_linear = train_and_eval(use_activation=False)
losses_relu, acc_relu = train_and_eval(use_activation=True)

print(f"无激活函数（纯线性）：最终准确率 = {acc_linear:.2%}  （应该 ≤ 75%）")
print(f"有 ReLU 激活函数：  最终准确率 = {acc_relu:.2%}  （应该 = 100%）")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(losses_linear, 'r-', label=f'无激活（纯线性），acc={acc_linear:.0%}')
ax.plot(losses_relu, 'b-', label=f'有 ReLU，acc={acc_relu:.0%}')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('激活函数的作用：有vs无，解XOR的差异')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_neural_networks/activation_vs_linear.png', dpi=100, bbox_inches='tight')
print("图片已保存：02_neural_networks/activation_vs_linear.png")
plt.show()

# ============================================================
# Part 4: 梯度消失问题的可视化
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 梯度消失 —— 为什么深层网络难训练？")
print("=" * 50)

"""
问题：在深层网络中，靠近输入的层梯度极小，几乎不更新。
原因：每层反向传播都要乘以激活函数的导数。
  Sigmoid 的导数最大 0.25，10层就是 0.25^10 ≈ 0.000001！

解决方案：
  1. 用 ReLU（正区间导数=1，不衰减）
  2. 残差连接（ResNet）→ 第3章详细讲
  3. Batch Normalization
"""

# 模拟梯度在不同激活函数下随深度的衰减
n_layers = 20
gradient_magnitude = np.zeros((n_layers, 3))

for layer in range(n_layers):
    # 假设每层激活函数的导数：Sigmoid ≈ 0.25（最大值），Tanh ≈ 1（最大值），ReLU ≈ 1（正区间）
    if layer == 0:
        gradient_magnitude[layer] = [1.0, 1.0, 1.0]
    else:
        gradient_magnitude[layer, 0] = gradient_magnitude[layer-1, 0] * 0.25  # Sigmoid
        gradient_magnitude[layer, 1] = gradient_magnitude[layer-1, 1] * 0.5   # Tanh（平均）
        gradient_magnitude[layer, 2] = gradient_magnitude[layer-1, 2] * 0.9   # ReLU（偶有死神经元）

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.semilogy(gradient_magnitude[:, 0], 'r-o', markersize=4, label='Sigmoid (导数≈0.25)')
ax.semilogy(gradient_magnitude[:, 1], 'g-s', markersize=4, label='Tanh (导数≈0.5)')
ax.semilogy(gradient_magnitude[:, 2], 'b-^', markersize=4, label='ReLU (导数≈0.9)')
ax.axhline(1e-6, color='gray', linestyle='--', alpha=0.5, label='消失阈值')
ax.set_xlabel('从输出向输入反传的层数')
ax.set_ylabel('梯度幅度（对数坐标）')
ax.set_title('梯度消失：不同激活函数的比较')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
# Sigmoid 在几层就"消失"了
for i, (n_l, label) in enumerate([(5, '5层'), (10, '10层'), (20, '20层')]):
    sigmoid_grads = [0.25**l for l in range(n_l)]
    ax.bar(np.arange(n_l) + i*0.25, sigmoid_grads, width=0.25, alpha=0.6, label=label)

ax.set_xlabel('从输出反传到第x层')
ax.set_ylabel('Sigmoid 梯度幅度')
ax.set_title('Sigmoid 的梯度消失：越深越糟糕')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_neural_networks/gradient_vanishing.png', dpi=100, bbox_inches='tight')
print("图片已保存：02_neural_networks/gradient_vanishing.png")
plt.show()

print("\n关键结论：")
print(f"  10层 Sigmoid 网络，梯度在输入层只有 0.25^10 = {0.25**10:.2e}（几乎为0！）")
print(f"  这就是为什么深度学习早期难以成功，直到 ReLU 和 ResNet 出现")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【死ReLU问题】
   当 x < 0 时，ReLU 的梯度为 0。
   如果某个神经元的输入一直是负数，权重永远不更新（"死"了）。
   Leaky ReLU 如何解决这个问题？
   画出 alpha=0.1 的 Leaky ReLU，并计算它在 x=-3 处的梯度。

2. 【GELU vs ReLU】
   GELU(x) = x * Φ(x)，其中 Φ 是标准正态分布的CDF。
   直觉：不是硬截断，而是"概率性"地让信号通过。
   在 -2 < x < 0 的区间，GELU 和 ReLU 的输出有什么区别？
   为什么 Transformer 模型（BERT/GPT）偏好 GELU？

3. 【激活函数的选择指南】
   总结并填写下表（根据本节内容）：
   场景               → 推荐激活函数 → 原因
   二分类输出层       → ?
   多分类输出层       → ?
   RNN/LSTM 内部      → ?
   深度 CNN/MLP 隐层  → ?
   Transformer 内部   → ?
""")
