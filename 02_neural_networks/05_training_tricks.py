"""
====================================================================
第2章 · 第5节 · 训练技巧：BatchNorm、Dropout、权重初始化
====================================================================

【一句话总结】
训练深度网络不只是设计架构——正确的初始化、归一化和正则化技巧
往往是训练成功与失败的分水岭。

【为什么深度学习需要这个？】
- 错误的初始化 → 梯度消失/爆炸 → 训练失败
- BatchNorm 让深度网络训练变得可行（2015年的突破）
- Dropout 是最简单有效的正则化方法
- 这些技巧在 CNN、RNN、Transformer 中都会反复出现
- 理解 BatchNorm 是理解 LayerNorm（Transformer用）的前置

【核心概念】

1. 权重初始化
   - 全零初始化：所有神经元学到完全相同的东西（对称性问题）
   - 随机初始化：打破对称性，但方差要选对
   - Xavier 初始化：Var(w) = 1/n_in，适合 sigmoid/tanh
   - He/Kaiming 初始化：Var(w) = 2/n_in，适合 ReLU
   - 原理：让每层的输出方差保持稳定，不放大也不缩小

2. Batch Normalization
   - 对每个 mini-batch 的每个特征做归一化：x̂ = (x - μ) / σ
   - 再用可学习参数缩放和平移：y = γ·x̂ + β
   - 好处：加速收敛、允许更大学习率、轻微正则化
   - 训练时用 batch 统计量，推理时用全局统计量（running mean/var）

3. Layer Normalization（预告）
   - BatchNorm 对 batch 维度归一化（依赖 batch size）
   - LayerNorm 对特征维度归一化（不依赖 batch size）
   - Transformer 用 LayerNorm，因为序列长度可变

4. Dropout
   - 训练时随机将一部分神经元输出置零（概率 p）
   - 推理时不 drop，但输出乘以 (1-p)（或训练时除以 1-p，即 inverted dropout）
   - 直觉：强迫网络不依赖任何单个神经元，学习更鲁棒的特征
   - 类比：考试时随机不让一些同学参加，逼每个人都学会所有内容

5. 梯度裁剪（Gradient Clipping）
   - 当梯度范数超过阈值时，等比例缩小
   - 防止梯度爆炸（尤其在 RNN 中重要）
   - 也用于 Transformer 训练

【前置知识】
第2章第3-4节
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =====================================================================
# 工具函数：简易全连接网络组件
# =====================================================================

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU 的导数：x>0 时为1，否则为0"""
    return (x > 0).astype(float)


def sigmoid(x):
    """Sigmoid 激活函数（数值稳定版）"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softmax_cross_entropy(logits, y_onehot):
    """Softmax + 交叉熵损失（前向 + 反向）

    参数:
        logits: 模型原始输出，形状 (N, C)
        y_onehot: one-hot 标签，形状 (N, C)
    返回:
        loss: 标量损失值
        dlogits: 损失对 logits 的梯度，形状 (N, C)
    """
    # 数值稳定的 softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
    dlogits = (probs - y_onehot) / N
    return loss, dlogits


def generate_spiral_data(n_per_class=100, n_classes=3, noise=0.3, seed=42):
    """生成螺旋形数据集（经典的非线性分类问题）

    返回:
        X: 特征矩阵 (N, 2)
        y: 标签向量 (N,)
        y_onehot: one-hot 标签 (N, n_classes)
    """
    np.random.seed(seed)
    N = n_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)
    for c in range(n_classes):
        idx = range(n_per_class * c, n_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        theta = np.linspace(c * 4, (c + 1) * 4, n_per_class) + \
                np.random.randn(n_per_class) * noise
        X[idx] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[idx] = c
    y_onehot = np.eye(n_classes)[y]
    return X, y, y_onehot


# =====================================================================
# 演示1：权重初始化实验 —— 不同初始化策略的效果
# =====================================================================
def demo_weight_init():
    """不同初始化策略对深层网络的影响

    核心思想：
    - 全零初始化 → 对称性问题，所有神经元输出完全相同
    - 随机初始化方差太大 → 激活值爆炸
    - 随机初始化方差太小 → 激活值消失
    - Xavier/He 初始化 → 让每层输出方差稳定在合理范围
    """
    print("=" * 60)
    print("演示1：权重初始化 —— 全零 vs 随机 vs Xavier vs He")
    print("=" * 60)

    np.random.seed(42)
    n_layers = 10          # 10层深度网络
    hidden_dim = 256       # 每层256个神经元
    n_samples = 512        # 512个样本
    input_dim = 256

    # 生成随机输入
    X = np.random.randn(n_samples, input_dim)

    # 四种初始化策略
    init_strategies = {
        "全零初始化":       lambda fan_in, fan_out: np.zeros((fan_in, fan_out)),
        "随机(std=1.0)":    lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 1.0,
        "Xavier(std=1/sqrt(n))": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) / np.sqrt(fan_in),
        "He(std=sqrt(2/n))":     lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("不同初始化策略下各层激活值分布（ReLU 激活，10层网络）",
                 fontsize=13, fontweight="bold")

    for ax, (name, init_fn) in zip(axes.flat, init_strategies.items()):
        layer_means = []
        layer_stds = []
        h = X.copy()

        for layer in range(n_layers):
            W = init_fn(hidden_dim, hidden_dim)
            h = relu(h @ W)
            layer_means.append(h.mean())
            layer_stds.append(h.std())

        ax.bar(range(n_layers), layer_stds, color="steelblue", alpha=0.8)
        ax.set_xlabel("层数")
        ax.set_ylabel("激活值标准差")
        ax.set_title(f"{name}\n最终层std={layer_stds[-1]:.6f}")
        ax.set_yscale("symlog", linthresh=1e-10)  # 对称对数尺度显示极小和极大值
        ax.set_xticks(range(n_layers))

        status = "正常" if 0.1 < layer_stds[-1] < 10 else "异常"
        print(f"  {name:<25s} → 最终层std = {layer_stds[-1]:.6e} [{status}]")

    plt.tight_layout()
    plt.savefig("05_init_strategies.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[已保存] 05_init_strategies.png")
    print("结论：He 初始化让 ReLU 网络各层激活值方差最稳定\n")


# =====================================================================
# 演示2：方差传播分析 —— 可视化每层激活值的分布
# =====================================================================
def demo_variance_propagation():
    """逐层跟踪激活值方差，直观展示初始化的重要性

    对比 Xavier 和 He 初始化在 tanh 和 ReLU 网络中的表现。
    正确的搭配：tanh + Xavier，ReLU + He
    """
    print("=" * 60)
    print("演示2：方差传播分析 —— 激活函数与初始化的搭配")
    print("=" * 60)

    np.random.seed(0)
    n_layers = 8
    dim = 512
    n_samples = 1000

    combos = [
        ("tanh + Xavier",  np.tanh, lambda n: np.random.randn(n, n) / np.sqrt(n)),
        ("tanh + He",      np.tanh, lambda n: np.random.randn(n, n) * np.sqrt(2.0 / n)),
        ("ReLU + Xavier",  relu,    lambda n: np.random.randn(n, n) / np.sqrt(n)),
        ("ReLU + He",      relu,    lambda n: np.random.randn(n, n) * np.sqrt(2.0 / n)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("方差传播：激活函数与初始化方法的正确搭配", fontsize=13, fontweight="bold")

    for ax, (name, act_fn, init_fn) in zip(axes.flat, combos):
        h = np.random.randn(n_samples, dim)
        variances = [h.var()]

        for _ in range(n_layers):
            W = init_fn(dim)
            h = act_fn(h @ W)
            variances.append(h.var())

        ax.plot(range(len(variances)), variances, "o-", color="steelblue",
                linewidth=2, markersize=6)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="理想方差=1")
        ax.set_xlabel("层数（0=输入）")
        ax.set_ylabel("激活值方差")
        ax.set_title(f"{name}\n最终方差={variances[-1]:.4f}")
        ax.legend()
        ax.set_ylim(bottom=0, top=max(2.0, max(variances) * 1.1))

        match = "匹配" if 0.3 < variances[-1] < 3.0 else "不匹配"
        print(f"  {name:<20s} → 最终方差 = {variances[-1]:.4f} [{match}]")

    plt.tight_layout()
    plt.savefig("05_variance_propagation.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[已保存] 05_variance_propagation.png")
    print("结论：tanh 配 Xavier、ReLU 配 He 是最佳搭配\n")


# =====================================================================
# 演示3：Batch Normalization 完整实现
# =====================================================================
class BatchNorm:
    """Batch Normalization 层的完整实现

    前向传播（训练模式）：
        1. 计算 mini-batch 的均值 μ 和方差 σ²
        2. 归一化：x̂ = (x - μ) / sqrt(σ² + ε)
        3. 缩放平移：y = γ * x̂ + β
        4. 更新 running_mean 和 running_var（用于推理）

    前向传播（推理模式）：
        使用全局统计量（running_mean/var）代替 batch 统计量

    反向传播：
        计算 dγ, dβ, dx 的梯度（链式法则）
    """

    def __init__(self, n_features, momentum=0.9, eps=1e-5):
        """初始化 BatchNorm 层

        参数:
            n_features: 特征维度
            momentum: 滑动平均系数（用于更新 running 统计量）
            eps: 数值稳定的小常数
        """
        # 可学习参数
        self.gamma = np.ones(n_features)     # 缩放因子，初始为1
        self.beta = np.zeros(n_features)     # 平移因子，初始为0
        # 推理时用的全局统计量
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.momentum = momentum
        self.eps = eps
        # 缓存（反向传播需要）
        self.cache = None

    def forward(self, x, training=True):
        """前向传播

        参数:
            x: 输入数据，形状 (N, D)
            training: 是否为训练模式
        返回:
            out: 归一化后的输出，形状 (N, D)
        """
        if training:
            # 训练模式：用当前 batch 的统计量
            mu = x.mean(axis=0)                          # (D,)
            var = x.var(axis=0)                           # (D,)
            x_hat = (x - mu) / np.sqrt(var + self.eps)   # (N, D)
            out = self.gamma * x_hat + self.beta         # (N, D)

            # 更新全局统计量（滑动平均）
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # 缓存用于反向传播
            self.cache = (x, x_hat, mu, var)
        else:
            # 推理模式：用全局统计量
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_hat + self.beta

        return out

    def backward(self, dout):
        """反向传播 —— 计算 BN 层的梯度

        这是 BN 最复杂的部分。关键：μ 和 σ² 都依赖于 x，
        所以求 dx 时需要考虑这三条路径。

        参数:
            dout: 上游梯度，形状 (N, D)
        返回:
            dx: 对输入的梯度，形状 (N, D)
        """
        x, x_hat, mu, var = self.cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        # 可学习参数的梯度
        self.dgamma = np.sum(dout * x_hat, axis=0)   # (D,)
        self.dbeta = np.sum(dout, axis=0)             # (D,)

        # 对输入 x 的梯度（最核心的推导）
        dx_hat = dout * self.gamma                     # (N, D)
        dvar = np.sum(dx_hat * (x - mu) * (-0.5) * (var + self.eps) ** (-1.5), axis=0)
        dmu = np.sum(dx_hat * (-std_inv), axis=0) + dvar * np.mean(-2.0 * (x - mu), axis=0)
        dx = dx_hat * std_inv + dvar * 2.0 * (x - mu) / N + dmu / N

        return dx


def demo_batchnorm_impl():
    """演示 BatchNorm 的基本行为：归一化 + 训练/推理模式切换"""
    print("=" * 60)
    print("演示3：BatchNorm 实现 —— 训练模式 vs 推理模式")
    print("=" * 60)

    np.random.seed(42)
    # 模拟一个 mini-batch：4个样本，3个特征
    x = np.array([[10.0, 200.0, 3000.0],
                  [20.0, 400.0, 1000.0],
                  [30.0, 100.0, 2000.0],
                  [40.0, 300.0, 4000.0]])

    bn = BatchNorm(n_features=3)

    # 训练模式前向传播
    out_train = bn.forward(x, training=True)
    print(f"输入各特征均值:    {x.mean(axis=0)}")
    print(f"输入各特征标准差:  {x.std(axis=0)}")
    print(f"BN输出各特征均值:  {out_train.mean(axis=0).round(6)}")
    print(f"BN输出各特征标准差:{out_train.std(axis=0).round(4)}")

    # 多次前向传播积累 running 统计量
    for _ in range(100):
        x_batch = np.random.randn(32, 3) * np.array([10, 100, 1000]) + np.array([25, 250, 2500])
        bn.forward(x_batch, training=True)

    # 推理模式
    out_infer = bn.forward(x, training=False)
    print(f"\n推理模式输出均值:  {out_infer.mean(axis=0).round(4)}")
    print(f"推理模式输出标准差:{out_infer.std(axis=0).round(4)}")
    print("注意：推理模式使用 running_mean/var，结果与训练模式略有不同\n")


# =====================================================================
# 演示4：BatchNorm 效果 —— 有无 BN 的训练对比
# =====================================================================
def demo_batchnorm_effect():
    """对比有/无 BatchNorm 的收敛速度

    用一个3层全连接网络在螺旋数据上训练，展示 BN 如何加速收敛。
    """
    print("=" * 60)
    print("演示4：BatchNorm 效果 —— 有 vs 无 BN 的收敛速度")
    print("=" * 60)

    X, y, y_onehot = generate_spiral_data(n_per_class=150, n_classes=3)
    n_classes = 3
    lr = 0.05
    epochs = 200

    def train_network(use_bn, label):
        """训练一个3层网络，可选是否使用 BN"""
        np.random.seed(7)
        d1, d2 = 64, 64

        # He 初始化
        W1 = np.random.randn(2, d1) * np.sqrt(2.0 / 2)
        W2 = np.random.randn(d1, d2) * np.sqrt(2.0 / d1)
        W3 = np.random.randn(d2, n_classes) * np.sqrt(2.0 / d2)

        if use_bn:
            bn1 = BatchNorm(d1)
            bn2 = BatchNorm(d2)

        losses = []
        for epoch in range(epochs):
            # === 前向传播 ===
            z1 = X @ W1
            a1 = bn1.forward(z1, training=True) if use_bn else z1
            h1 = relu(a1)

            z2 = h1 @ W2
            a2 = bn2.forward(z2, training=True) if use_bn else z2
            h2 = relu(a2)

            logits = h2 @ W3
            loss, dlogits = softmax_cross_entropy(logits, y_onehot)
            losses.append(loss)

            # === 反向传播 ===
            dW3 = h2.T @ dlogits
            dh2 = dlogits @ W3.T
            da2 = dh2 * relu_grad(a2)
            da2 = bn2.backward(da2) if use_bn else da2
            dW2 = h1.T @ da2
            dh1 = da2 @ W2.T
            da1 = dh1 * relu_grad(a1)
            da1 = bn1.backward(da1) if use_bn else da1
            dW1 = X.T @ da1

            # === 参数更新 ===
            W1 -= lr * dW1
            W2 -= lr * dW2
            W3 -= lr * dW3
            if use_bn:
                bn1.gamma -= lr * bn1.dgamma
                bn1.beta -= lr * bn1.dbeta
                bn2.gamma -= lr * bn2.dgamma
                bn2.beta -= lr * bn2.dbeta

        print(f"  {label}: 最终损失 = {losses[-1]:.4f}")
        return losses

    losses_no_bn = train_network(use_bn=False, label="无 BN")
    losses_bn    = train_network(use_bn=True,  label="有 BN")

    # 画训练曲线
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(losses_no_bn, label="无 BatchNorm", color="tomato", linewidth=2)
    ax.plot(losses_bn, label="有 BatchNorm", color="steelblue", linewidth=2)
    ax.set_xlabel("训练轮数 (epoch)")
    ax.set_ylabel("交叉熵损失")
    ax.set_title("BatchNorm 加速收敛效果对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("05_batchnorm_effect.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[已保存] 05_batchnorm_effect.png")
    print("观察：有 BN 的网络收敛更快、更稳定\n")


# =====================================================================
# 演示5：Dropout 实现 —— Inverted Dropout（前向 + 反向）
# =====================================================================
class Dropout:
    """Inverted Dropout 实现

    Inverted Dropout 的关键：
    - 训练时：随机置零 + 除以 (1-p) 来补偿
    - 推理时：什么都不做（直接通过）
    这样推理时不需要额外乘系数，更方便部署。

    类比：训练时随机让一些同学缺考（p=0.5 即一半人缺考），
    剩下的同学分数翻倍来补偿。测试时全员参加，分数正常计算。
    """

    def __init__(self, drop_prob=0.5):
        """初始化 Dropout 层

        参数:
            drop_prob: 丢弃概率（0.5 意味着随机丢弃一半神经元）
        """
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, x, training=True):
        """前向传播

        参数:
            x: 输入，形状任意
            training: 是否为训练模式
        返回:
            out: 输出，形状与输入相同
        """
        if training and self.drop_prob > 0:
            # 生成伯努利掩码：每个元素以 (1-p) 的概率保留
            self.mask = (np.random.rand(*x.shape) > self.drop_prob).astype(float)
            # 乘以掩码并除以 (1-p) —— inverted dropout 的核心
            out = x * self.mask / (1.0 - self.drop_prob)
        else:
            # 推理模式：直接通过
            out = x
        return out

    def backward(self, dout):
        """反向传播

        参数:
            dout: 上游梯度
        返回:
            dx: 对输入的梯度（被丢弃的位置梯度也为0）
        """
        # 前向时被置零的位置，反向时梯度也为0
        dx = dout * self.mask / (1.0 - self.drop_prob)
        return dx


def demo_dropout_impl():
    """演示 Dropout 的基本行为"""
    print("=" * 60)
    print("演示5：Dropout 实现 —— 训练 vs 推理模式")
    print("=" * 60)

    np.random.seed(42)
    x = np.ones((1, 10))  # 全1输入，方便观察
    dropout = Dropout(drop_prob=0.5)

    out_train = dropout.forward(x, training=True)
    out_infer = dropout.forward(x, training=False)

    print(f"输入:         {x[0]}")
    print(f"训练模式输出: {out_train[0]}")
    print(f"  掩码:       {dropout.mask[0].astype(int)}")
    print(f"  输出均值:   {out_train.mean():.2f}（期望接近 1.0）")
    print(f"推理模式输出: {out_infer[0]}")

    # 验证：多次训练模式前向传播的期望值等于推理模式输出
    means = []
    for _ in range(10000):
        out = dropout.forward(x, training=True)
        means.append(out.mean())
    print(f"\n训练模式 10000 次平均输出: {np.mean(means):.4f}")
    print("结论：Inverted Dropout 保证训练和推理的期望输出一致\n")


# =====================================================================
# 演示6：Dropout 效果 —— 防止过拟合
# =====================================================================
def demo_dropout_effect():
    """用容易过拟合的小数据集，展示 Dropout 的正则化效果

    对比三种情况：无 Dropout、Dropout=0.3、Dropout=0.5
    观察训练/测试误差的差距（泛化 gap）
    """
    print("=" * 60)
    print("演示6：Dropout 效果 —— 对比有无 Dropout 的泛化能力")
    print("=" * 60)

    # 小数据集 + 大网络 → 容易过拟合
    X_train, y_train, y_train_oh = generate_spiral_data(n_per_class=50, seed=42)
    X_test, y_test, y_test_oh = generate_spiral_data(n_per_class=100, seed=99)
    n_classes = 3
    lr = 0.02
    epochs = 300

    def train_with_dropout(drop_prob, label):
        """训练带 Dropout 的3层网络"""
        np.random.seed(7)
        d1, d2 = 128, 128  # 宽网络，容易过拟合

        W1 = np.random.randn(2, d1) * np.sqrt(2.0 / 2)
        W2 = np.random.randn(d1, d2) * np.sqrt(2.0 / d1)
        W3 = np.random.randn(d2, n_classes) * np.sqrt(2.0 / d2)

        dp1 = Dropout(drop_prob)
        dp2 = Dropout(drop_prob)

        train_losses, test_losses = [], []

        for epoch in range(epochs):
            # --- 训练前向 ---
            h1 = relu(X_train @ W1)
            h1_drop = dp1.forward(h1, training=True)
            h2 = relu(h1_drop @ W2)
            h2_drop = dp2.forward(h2, training=True)
            logits = h2_drop @ W3

            loss, dlogits = softmax_cross_entropy(logits, y_train_oh)
            train_losses.append(loss)

            # --- 测试前向（无 Dropout）---
            h1_t = relu(X_test @ W1)
            h2_t = relu(h1_t @ W2)
            logits_t = h2_t @ W3
            test_loss, _ = softmax_cross_entropy(logits_t, y_test_oh)
            test_losses.append(test_loss)

            # --- 反向传播 ---
            dW3 = h2_drop.T @ dlogits
            dh2_drop = dlogits @ W3.T
            dh2 = dp2.backward(dh2_drop)
            da2 = dh2 * relu_grad(h1_drop @ W2)
            dW2 = h1_drop.T @ da2
            dh1_drop = da2 @ W2.T
            dh1 = dp1.backward(dh1_drop)
            da1 = dh1 * relu_grad(X_train @ W1)
            dW1 = X_train.T @ da1

            W1 -= lr * dW1
            W2 -= lr * dW2
            W3 -= lr * dW3

        gap = train_losses[-1] - test_losses[-1]
        print(f"  {label}: 训练损失={train_losses[-1]:.4f}, "
              f"测试损失={test_losses[-1]:.4f}, 泛化Gap={abs(gap):.4f}")
        return train_losses, test_losses

    results = {}
    for dp, label in [(0.0, "无Dropout"), (0.3, "Dropout=0.3"), (0.5, "Dropout=0.5")]:
        results[label] = train_with_dropout(dp, label)

    # 画训练和测试曲线
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Dropout 对过拟合的抑制效果", fontsize=13, fontweight="bold")
    colors = ["tomato", "orange", "steelblue"]

    for ax, (label, (tr, te)), color in zip(axes, results.items(), colors):
        ax.plot(tr, label="训练损失", color=color, linewidth=2)
        ax.plot(te, label="测试损失", color=color, linewidth=2, linestyle="--")
        ax.set_xlabel("训练轮数")
        ax.set_ylabel("损失")
        ax.set_title(f"{label}\n训练={tr[-1]:.3f} / 测试={te[-1]:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("05_dropout_effect.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[已保存] 05_dropout_effect.png")
    print("观察：Dropout 增大训练损失但降低测试损失 → 抑制过拟合\n")


# =====================================================================
# 演示7：梯度裁剪实现 —— 防止梯度爆炸
# =====================================================================
def clip_grad_norm(grads, max_norm):
    """按范数裁剪梯度

    如果所有参数梯度的总范数超过 max_norm，
    则等比例缩小所有梯度，使总范数恰好等于 max_norm。

    参数:
        grads: 梯度列表 [dW1, dW2, ...]
        max_norm: 允许的最大范数
    返回:
        clipped_grads: 裁剪后的梯度列表
        total_norm: 原始总范数
    """
    # 计算所有梯度的总范数
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))

    # 如果超过阈值，等比例缩小
    clip_coef = max_norm / (total_norm + 1e-8)
    if clip_coef < 1.0:
        clipped_grads = [g * clip_coef for g in grads]
    else:
        clipped_grads = [g.copy() for g in grads]

    return clipped_grads, total_norm


def demo_gradient_clipping():
    """展示梯度裁剪如何防止训练不稳定

    模拟一个梯度爆炸场景：较大的学习率导致训练振荡甚至发散。
    梯度裁剪能有效稳定训练过程。
    """
    print("=" * 60)
    print("演示7：梯度裁剪 —— 防止梯度爆炸")
    print("=" * 60)

    X, y, y_onehot = generate_spiral_data(n_per_class=100, n_classes=3)
    n_classes = 3
    lr = 0.5           # 故意用较大学习率触发不稳定
    epochs = 150

    def train_with_clipping(max_norm, label):
        """训练时可选是否裁剪梯度"""
        np.random.seed(7)
        d1 = 64
        W1 = np.random.randn(2, d1) * np.sqrt(2.0 / 2)
        W2 = np.random.randn(d1, n_classes) * np.sqrt(2.0 / d1)

        losses = []
        grad_norms = []

        for epoch in range(epochs):
            # 前向
            h1 = relu(X @ W1)
            logits = h1 @ W2
            loss, dlogits = softmax_cross_entropy(logits, y_onehot)
            losses.append(loss)

            # 反向
            dW2 = h1.T @ dlogits
            dh1 = dlogits @ W2.T
            da1 = dh1 * relu_grad(X @ W1)
            dW1 = X.T @ da1

            # 记录并裁剪梯度
            grads = [dW1, dW2]
            if max_norm is not None:
                grads, total_norm = clip_grad_norm(grads, max_norm)
            else:
                total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
            grad_norms.append(total_norm)

            # 更新参数
            W1 -= lr * grads[0]
            W2 -= lr * grads[1]

        print(f"  {label}: 最终损失={losses[-1]:.4f}, "
              f"平均梯度范数={np.mean(grad_norms):.4f}")
        return losses, grad_norms

    losses_no_clip, norms_no_clip = train_with_clipping(None, "无裁剪")
    losses_clip, norms_clip       = train_with_clipping(1.0,  "裁剪(max=1.0)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("梯度裁剪对训练稳定性的影响（学习率=0.5）",
                 fontsize=13, fontweight="bold")

    # 左图：损失曲线
    axes[0].plot(losses_no_clip, label="无梯度裁剪", color="tomato", linewidth=2)
    axes[0].plot(losses_clip, label="梯度裁剪(max=1.0)", color="steelblue", linewidth=2)
    axes[0].set_xlabel("训练轮数")
    axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线对比")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：梯度范数
    axes[1].plot(norms_no_clip, label="无裁剪", color="tomato", alpha=0.7, linewidth=1.5)
    axes[1].plot(norms_clip, label="裁剪后", color="steelblue", alpha=0.7, linewidth=1.5)
    axes[1].axhline(y=1.0, color="green", linestyle="--", label="裁剪阈值=1.0")
    axes[1].set_xlabel("训练轮数")
    axes[1].set_ylabel("梯度范数")
    axes[1].set_title("梯度范数变化")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("05_gradient_clipping.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("[已保存] 05_gradient_clipping.png")
    print("观察：梯度裁剪将梯度范数限制在阈值以内，训练更平稳\n")


# =====================================================================
# 思考题
# =====================================================================
def print_questions():
    """本节思考题"""
    questions = """
+================================================================+
|                         本节思考题                              |
+================================================================+
|                                                                |
|  1. 为什么 BatchNorm 在推理时不能用当前输入的统计量？           |
|     如果推理时只有一个样本（batch_size=1），会怎样？            |
|     提示：想想"均值"和"方差"在单个样本上有没有意义。            |
|                                                                |
|  2. LayerNorm 和 BatchNorm 的归一化维度不同。                  |
|     假设输入形状是 (N, D)：                                    |
|       - BatchNorm 沿 N 维度（对每个特征跨样本）归一化           |
|       - LayerNorm 沿 D 维度（对每个样本跨特征）归一化           |
|     为什么 Transformer 选择 LayerNorm 而不是 BatchNorm？        |
|     提示：考虑序列长度可变和分布式训练的场景。                  |
|                                                                |
|  3. Inverted Dropout 训练时除以 (1-p) 而不是推理时乘以 (1-p)。 |
|     这两种方式数学上等价吗？为什么实际中优先用 inverted 版本？  |
|     提示：考虑部署时的计算效率和代码简洁性。                    |
|                                                                |
|  4. 如果同时使用 BatchNorm 和 Dropout，应该怎么安排顺序？      |
|     BN → Dropout 还是 Dropout → BN？会有冲突吗？               |
|     提示：Dropout 改变了激活值的分布，BN 依赖分布的统计量。     |
|                                                                |
|  5. He 初始化的推导假设 ReLU 会丢掉一半的值（负数变0），        |
|     所以方差要乘以 2 来补偿。如果使用 Leaky ReLU               |
|     （负数部分斜率为 0.01 而非 0），初始化方差应该怎么调整？    |
|     提示：计算 Leaky ReLU 输出的方差占输入方差的比例。          |
|                                                                |
+================================================================+
"""
    print(questions)


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  第2章 · 第5节 · 训练技巧：BatchNorm、Dropout、权重初始化")
    print("=" * 60 + "\n")

    demo_weight_init()              # 演示1：初始化实验
    demo_variance_propagation()     # 演示2：方差传播分析
    demo_batchnorm_impl()           # 演示3：BatchNorm 实现
    demo_batchnorm_effect()         # 演示4：BatchNorm 效果
    demo_dropout_impl()             # 演示5：Dropout 实现
    demo_dropout_effect()           # 演示6：Dropout 效果
    demo_gradient_clipping()        # 演示7：梯度裁剪

    print_questions()               # 思考题

    print("=" * 60)
    print("本节总结：")
    print("  1. 权重初始化：He 配 ReLU，Xavier 配 tanh/sigmoid")
    print("  2. BatchNorm：归一化激活值，加速收敛，轻微正则化")
    print("  3. Dropout：随机丢弃神经元，强制学习鲁棒特征")
    print("  4. 梯度裁剪：限制梯度范数，防止梯度爆炸")
    print("  5. 这些技巧可以组合使用，是深度学习工程实践的基石")
    print("=" * 60)
    print("\n下一节预告：第3章 · 卷积神经网络（CNN）")
    print("  → 图像处理的利器，理解卷积、池化和现代 CNN 架构\n")
