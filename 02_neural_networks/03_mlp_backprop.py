"""
====================================================================
第2章 · 第3节 · MLP 与反向传播（本教程最核心的文件！）
====================================================================

【一句话总结】
多层感知机（MLP）+ 反向传播 = 深度学习的基石。理解了这个文件，
你就理解了所有神经网络训练的数学本质。

【为什么这是最重要的文件？】
- 前向传播：数据如何流过网络得到预测
- 损失函数：如何衡量预测的好坏
- 反向传播：如何计算每个权重的梯度
- 参数更新：如何用梯度改进权重
- 这四步就是训练任何神经网络（包括GPT）的完整流程！

【核心概念】

1. 多层感知机（MLP）
   - 结构：输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
   - 每层：线性变换(y=Wx+b) + 激活函数
   - 层与层之间全连接（每个神经元连接下一层所有神经元）

2. 前向传播（Forward Pass）
   - 数据从输入层逐层传递到输出层
   - 每层：h = activation(W·x + b)
   - 最后一层通常用 softmax（分类）或恒等（回归）

3. 损失函数（Loss Function）
   - 分类：交叉熵 L = -Σ y_true · log(y_pred)
   - 回归：MSE L = (1/N)Σ(y_pred - y_true)²
   - 损失是"标量"——一个数字，衡量模型整体表现

4. 反向传播（Backpropagation）
   - 从损失开始，利用链式法则逐层反向计算梯度
   - 每层需要保存前向传播的中间结果（用于计算梯度）
   - ∂L/∂W = ∂L/∂y · ∂y/∂z · ∂z/∂W
   - 关键洞察：反向传播就是链式法则的动态规划实现

5. 梯度检查（Gradient Checking）
   - 用数值梯度验证解析梯度的正确性
   - 数值梯度：(L(w+ε) - L(w-ε)) / 2ε
   - 如果二者相对误差 < 1e-5，说明反向传播实现正确
   - 调试神经网络代码的必备工具

6. 完整训练循环
   for epoch in range(num_epochs):
       # 1. 前向传播：得到预测
       # 2. 计算损失
       # 3. 反向传播：计算梯度
       # 4. 更新参数：w -= lr * grad
   这个循环对CNN、RNN、Transformer、GPT全部适用！

【前置知识】
第0章第4节 - 链式法则，第2章第1-2节
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和全局绘图风格
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════
# 第1部分：层（Layer）的设计 —— 前向传播 + 反向传播的构建单元
# ════════════════════════════════════════════════════════════════════
#
# 设计思想：每一层都是一个独立模块，有两个核心方法：
#   forward(x)       → 计算输出，同时缓存中间结果
#   backward(d_out)   → 根据上游梯度计算本层梯度，返回下游梯度
#
# 为什么要缓存？因为反向传播计算梯度时需要前向传播的中间值！
# 例如：∂(Wx)/∂W = x^T，所以必须保存输入 x。
#

print("=" * 60)
print("第1部分：构建层（Layer）—— 神经网络的积木")
print("=" * 60)


class Linear:
    """
    全连接层（线性层）：z = X @ W + b

    前向传播：
        输入 X 形状 (N, D_in)，输出 z 形状 (N, D_out)
        z = X @ W + b

    反向传播（详细推导）：
        假设损失 L 关于本层输出 z 的梯度为 dz（形状同 z）

        ∂L/∂W = X^T @ dz
        解释：z = X @ W，对 W 求导得 X^T，然后链式法则乘以上游梯度 dz

        ∂L/∂b = sum(dz, axis=0)
        解释：b 被广播加到每个样本上，所以梯度要沿 batch 维度求和

        ∂L/∂X = dz @ W^T
        解释：z = X @ W，对 X 求导得 W^T，然后链式法则乘以 dz
        这个 ∂L/∂X 会传给前一层作为它的 dz（链式法则的精髓！）
    """

    def __init__(self, in_features, out_features):
        """
        参数:
            in_features  : 输入维度 D_in
            out_features : 输出维度 D_out
        """
        # Xavier 初始化：让前向传播的方差稳定，避免信号爆炸或消失
        # 标准差 = sqrt(2 / (D_in + D_out))
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

        # 缓存：前向传播时保存，反向传播时使用
        self.cache = None

        # 梯度：反向传播时计算并存储
        self.dW = None
        self.db = None

    def forward(self, X):
        """
        前向传播：z = X @ W + b

        参数:
            X : 输入，形状 (N, D_in)
        返回:
            z : 输出，形状 (N, D_out)
        """
        self.cache = X  # 缓存输入，反向传播时计算 ∂L/∂W 需要
        z = X @ self.W + self.b  # (N, D_in) @ (D_in, D_out) + (D_out,) → (N, D_out)
        return z

    def backward(self, dz):
        """
        反向传播：计算 ∂L/∂W, ∂L/∂b, ∂L/∂X

        参数:
            dz : 上游梯度，形状 (N, D_out)，即 ∂L/∂z
        返回:
            dX : 下游梯度，形状 (N, D_in)，即 ∂L/∂X，传给前一层
        """
        X = self.cache
        N = X.shape[0]

        # ∂L/∂W = X^T @ dz，形状 (D_in, N) @ (N, D_out) → (D_in, D_out)
        self.dW = X.T @ dz / N   # 除以 N：取 batch 平均

        # ∂L/∂b = Σ dz（沿 batch 维度求和），形状 (D_out,)
        self.db = np.sum(dz, axis=0) / N

        # ∂L/∂X = dz @ W^T，形状 (N, D_out) @ (D_out, D_in) → (N, D_in)
        dX = dz @ self.W.T

        return dX


class ReLU:
    """
    ReLU 激活函数层：a = max(0, z)

    前向传播：
        a = max(0, z)
        大于0的保留，小于0的置为0

    反向传播：
        ∂a/∂z = 1  如果 z > 0
        ∂a/∂z = 0  如果 z ≤ 0
        所以：da_upstream 只在 z > 0 的位置通过，其余置0

    为什么 ReLU 这么流行？
        1. 计算简单（只需比较和赋值）
        2. 缓解梯度消失（正区间梯度恒为1）
        3. 产生稀疏激活（约50%神经元输出为0）
    """

    def __init__(self):
        self.cache = None

    def forward(self, z):
        """前向传播：a = max(0, z)"""
        self.cache = z  # 缓存 z，反向传播时需要判断 z > 0
        return np.maximum(0, z)

    def backward(self, da):
        """
        反向传播：梯度只在 z > 0 的位置通过

        参数:
            da : 上游梯度 ∂L/∂a
        返回:
            dz : ∂L/∂z = da * (z > 0)
        """
        z = self.cache
        dz = da * (z > 0).astype(float)  # z > 0 → 梯度通过；z ≤ 0 → 梯度被"杀死"
        return dz


class Sigmoid:
    """
    Sigmoid 激活函数层：σ(z) = 1 / (1 + exp(-z))

    前向传播：
        a = 1 / (1 + exp(-z))

    反向传播（优美的数学性质！）：
        ∂σ/∂z = σ(z) · (1 - σ(z))
        所以：dz = da * a * (1 - a)
        只需要前向传播的输出 a，不需要重新计算！

    注意：Sigmoid 在深层网络中容易导致梯度消失（因为梯度最大只有0.25）
    """

    def __init__(self):
        self.cache = None

    def forward(self, z):
        """前向传播：a = σ(z)"""
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # clip 防止溢出
        self.cache = a  # 缓存输出 a（不是输入 z！因为梯度公式用的是 a）
        return a

    def backward(self, da):
        """
        反向传播：dz = da * σ(z) * (1 - σ(z))

        参数:
            da : 上游梯度 ∂L/∂a
        返回:
            dz : ∂L/∂z = da * a * (1 - a)
        """
        a = self.cache
        dz = da * a * (1.0 - a)
        return dz


print("  Linear 层 : z = X @ W + b")
print("  ReLU 层   : a = max(0, z)")
print("  Sigmoid 层: a = 1 / (1 + exp(-z))")
print("  每层都实现了 forward() 和 backward() 两个方法")


# ════════════════════════════════════════════════════════════════════
# 第2部分：Softmax + 交叉熵（合并计算，数值稳定）
# ════════════════════════════════════════════════════════════════════
#
# 为什么要合并？
#   单独计算 softmax 再计算 cross-entropy，中间会出现 log(很小的数)
#   导致数值不稳定。合并后可以通过 log-sum-exp 技巧避免这个问题。
#
# 数学推导：
#   softmax: p_i = exp(z_i) / Σ_j exp(z_j)
#   交叉熵: L = -Σ_i y_i · log(p_i)  （y 是 one-hot 标签）
#
#   合并后的反向传播（非常简洁！）：
#   ∂L/∂z_i = p_i - y_i
#   即：梯度 = 预测概率 - 真实标签（one-hot）
#   这个结果的推导需要用到 softmax 的 Jacobian，但最终形式惊人地简单。
#

print("\n" + "=" * 60)
print("第2部分：Softmax 交叉熵损失（合并实现）")
print("=" * 60)


class SoftmaxCrossEntropy:
    """
    Softmax + 交叉熵损失，合并实现以保证数值稳定。

    前向传播：
        1. z_shifted = z - max(z)          ← 减去最大值，防止 exp 溢出
        2. p = exp(z_shifted) / Σexp(z_shifted)  ← softmax 概率
        3. L = -(1/N) Σ_n Σ_c y_nc · log(p_nc)  ← 交叉熵

    反向传播：
        ∂L/∂z = (p - y) / N
        推导过程（以单个样本为例）：
        - softmax: p_i = exp(z_i) / S，其中 S = Σ_j exp(z_j)
        - 交叉熵: L = -Σ_i y_i log(p_i)
        - 当 i = 真实类别 k 时：
          ∂L/∂z_k = -y_k · (1/p_k) · p_k · (1 - p_k) + Σ_{i≠k} -y_i · (1/p_i) · (-p_i · p_k)
                   = -y_k · (1 - p_k) + Σ_{i≠k} y_i · p_k
                   = -y_k + p_k · Σ_i y_i
                   = p_k - y_k   （因为 Σy_i = 1）
        - 所以：∂L/∂z = p - y，惊人地简洁！
    """

    def __init__(self):
        self.cache = None

    def forward(self, z, y_onehot):
        """
        前向传播：计算 softmax 概率和交叉熵损失

        参数:
            z        : logits（未归一化的分数），形状 (N, C)
            y_onehot : one-hot 标签，形状 (N, C)
        返回:
            loss : 标量，平均交叉熵损失
        """
        N = z.shape[0]

        # --- Softmax（数值稳定版）---
        # 减去每行最大值，防止 exp 溢出（不影响概率值，因为分子分母同时缩放）
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # (N, C)

        # --- 交叉熵损失 ---
        # clip 防止 log(0)
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        loss = -np.sum(y_onehot * log_probs) / N

        # 缓存概率和标签，反向传播时使用
        self.cache = (probs, y_onehot, N)

        return loss

    def backward(self):
        """
        反向传播：∂L/∂z = (p - y) / N

        返回:
            dz : 形状 (N, C)，损失对 logits 的梯度
        """
        probs, y_onehot, N = self.cache
        dz = (probs - y_onehot) / N
        return dz


# 快速验证
z_test = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
y_test_oh = np.array([[1, 0, 0], [0, 1, 0]])  # 第一个样本属于类0，第二个属于类1
loss_fn = SoftmaxCrossEntropy()
loss_val = loss_fn.forward(z_test, y_test_oh)
print(f"  测试 logits:\n  {z_test}")
print(f"  测试标签 (one-hot): {y_test_oh}")
print(f"  交叉熵损失: {loss_val:.4f}")
print(f"  反向传播梯度形状: {loss_fn.backward().shape}")


# ════════════════════════════════════════════════════════════════════
# 第3部分：MLP 类 —— 把层堆叠成网络
# ════════════════════════════════════════════════════════════════════
#
# MLP 就是多个层按顺序排列：
#   Linear → ReLU → Linear → ReLU → ... → Linear
# 前向传播：数据依次流过每一层
# 反向传播：梯度从最后一层反向流过每一层
#

print("\n" + "=" * 60)
print("第3部分：MLP 类 —— 搭建多层感知机")
print("=" * 60)


class MLP:
    """
    多层感知机：由多个 Linear + 激活函数层堆叠而成。

    结构示例（2个隐藏层，3分类）：
        输入(2) → Linear(2→64) → ReLU → Linear(64→32) → ReLU → Linear(32→3) → Softmax

    参数:
        layer_dims  : 每层的维度列表，如 [2, 64, 32, 3]
        activation  : 激活函数类型，'relu' 或 'sigmoid'
    """

    def __init__(self, layer_dims, activation="relu"):
        """
        构建网络层。

        参数:
            layer_dims : 列表，如 [2, 64, 32, 3]
                         第一个是输入维度，最后一个是输出类别数
            activation : 'relu' 或 'sigmoid'
        """
        self.layers = []
        self.loss_fn = SoftmaxCrossEntropy()

        # 逐层构建：Linear → Activation → Linear → Activation → ... → Linear
        for i in range(len(layer_dims) - 1):
            # 添加线性层
            self.layers.append(Linear(layer_dims[i], layer_dims[i + 1]))

            # 最后一层（输出层）不加激活函数（softmax 在损失函数中处理）
            if i < len(layer_dims) - 2:
                if activation == "relu":
                    self.layers.append(ReLU())
                elif activation == "sigmoid":
                    self.layers.append(Sigmoid())

        # 打印网络结构
        print(f"  网络结构: {' → '.join(str(d) for d in layer_dims)}")
        print(f"  激活函数: {activation}")
        print(f"  总层数: {len(self.layers)} (含 {len(layer_dims)-1} 个线性层)")

    def forward(self, X):
        """
        前向传播：数据依次流过所有层。

        参数:
            X : 输入数据，形状 (N, D_in)
        返回:
            logits : 最后一层的输出（未经 softmax），形状 (N, C)
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, logits, y_onehot):
        """计算交叉熵损失"""
        return self.loss_fn.forward(logits, y_onehot)

    def backward(self):
        """
        反向传播：梯度从损失层反向流过所有层。

        这就是反向传播的全部——逆序遍历每一层，把梯度传下去！
        """
        # 从损失函数获取初始梯度 ∂L/∂z_last = (softmax_prob - y) / N
        grad = self.loss_fn.backward()

        # 逆序遍历所有层，逐层传递梯度
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, lr):
        """
        参数更新：w -= lr * ∂L/∂w （最简单的 SGD）

        参数:
            lr : 学习率
        """
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

    def predict(self, X):
        """预测类别：取概率最大的类"""
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def get_params(self):
        """获取所有线性层的参数（用于梯度检查）"""
        params = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                params.append(layer.W)
                params.append(layer.b)
        return params

    def get_grads(self):
        """获取所有线性层的梯度（用于梯度检查）"""
        grads = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                grads.append(layer.dW)
                grads.append(layer.db)
        return grads


# ════════════════════════════════════════════════════════════════════
# 第4部分：反向传播的详细推导（两层网络）
# ════════════════════════════════════════════════════════════════════
#
# 以一个具体的两层网络为例，逐步推导每一步的梯度。
# 网络结构：
#   输入 X(N, 2) → Linear1(2→4) → ReLU → Linear2(4→3) → Softmax+CE
#
# 【前向传播路径】
#   z1 = X @ W1 + b1         形状 (N, 4)
#   a1 = ReLU(z1)            形状 (N, 4)
#   z2 = a1 @ W2 + b2        形状 (N, 3)
#   p  = softmax(z2)         形状 (N, 3)
#   L  = cross_entropy(p, y) 标量
#
# 【反向传播路径（从后往前）】
#
#   第一步：∂L/∂z2 = (p - y) / N
#     这是 softmax+交叉熵合并后的梯度，形状 (N, 3)
#
#   第二步：∂L/∂W2 = a1^T @ ∂L/∂z2
#     链式法则：z2 = a1 @ W2 + b2
#     对 W2 求导：∂z2/∂W2 = a1^T
#     所以：∂L/∂W2 = a1^T @ (∂L/∂z2)，形状 (4, 3)
#
#   第三步：∂L/∂b2 = Σ(∂L/∂z2, axis=0)
#     b2 被广播到每个样本，所以梯度沿 batch 维度求和，形状 (3,)
#
#   第四步：∂L/∂a1 = ∂L/∂z2 @ W2^T
#     链式法则：z2 = a1 @ W2，对 a1 求导：∂z2/∂a1 = W2^T
#     所以：∂L/∂a1 = (∂L/∂z2) @ W2^T，形状 (N, 4)
#
#   第五步：∂L/∂z1 = ∂L/∂a1 * (z1 > 0)
#     ReLU 的导数：a1 = max(0, z1)
#     ∂a1/∂z1 = 1 if z1 > 0 else 0
#     所以只在 z1 > 0 的位置让梯度通过，形状 (N, 4)
#
#   第六步：∂L/∂W1 = X^T @ ∂L/∂z1
#     与第二步完全相同的模式！形状 (2, 4)
#
#   第七步：∂L/∂b1 = Σ(∂L/∂z1, axis=0)
#     形状 (4,)
#
# 注意到没有？每一层的反向传播模式完全相同：
#   1. 计算参数梯度（dW, db）
#   2. 计算输入梯度（dX）传给前一层
# 这就是为什么我们可以把它封装成 Layer.backward() 方法！
#

print("\n" + "=" * 60)
print("第4部分：反向传播详细推导（两层网络实例）")
print("=" * 60)

# 创建一个小网络来验证
np.random.seed(0)
demo_net = MLP([2, 4, 3], activation="relu")

# 造一些假数据
X_demo = np.array([[1.0, 0.5], [-0.3, 0.8], [0.7, -0.2]])  # 3个样本，2个特征
y_demo = np.array([0, 1, 2])  # 3个样本，分别属于类 0, 1, 2
y_demo_oh = np.eye(3)[y_demo]  # 转为 one-hot

# 前向传播
logits = demo_net.forward(X_demo)
loss = demo_net.compute_loss(logits, y_demo_oh)
print(f"\n  输入 X 形状: {X_demo.shape}")
print(f"  logits 形状: {logits.shape}")
print(f"  损失值: {loss:.4f}")

# 反向传播
demo_net.backward()
print("\n  反向传播完成！各层梯度形状：")
for i, layer in enumerate(demo_net.layers):
    if isinstance(layer, Linear):
        print(f"    Linear 层: dW{layer.W.shape}, db{layer.b.shape}")


# ════════════════════════════════════════════════════════════════════
# 第5部分：梯度检查 —— 验证反向传播的正确性
# ════════════════════════════════════════════════════════════════════
#
# 数值梯度：用有限差分法近似计算梯度
#   ∂L/∂w_ij ≈ (L(w_ij + ε) - L(w_ij - ε)) / (2ε)
#
# 与解析梯度比较：
#   relative_error = |grad_analytic - grad_numeric| / max(|grad_analytic|, |grad_numeric|)
#   如果 relative_error < 1e-5，说明反向传播实现正确！
#
# 这是调试神经网络代码最重要的工具，没有之一。
# 第一次写反向传播一定要做梯度检查！
#

print("\n" + "=" * 60)
print("第5部分：梯度检查 —— 调试反向传播的必备工具")
print("=" * 60)


def gradient_check(net, X, y_onehot, epsilon=1e-5):
    """
    用数值梯度验证解析梯度（反向传播）的正确性。

    对网络中的每个参数矩阵，逐元素计算数值梯度，
    并与反向传播给出的解析梯度进行比较。

    参数:
        net      : MLP 网络
        X        : 输入数据
        y_onehot : one-hot 标签
        epsilon  : 有限差分的步长

    返回:
        all_passed : 是否所有参数的梯度检查都通过
    """
    # 先做一次前向+反向传播，获取解析梯度
    logits = net.forward(X)
    net.compute_loss(logits, y_onehot)
    net.backward()
    analytic_grads = net.get_grads()

    # 获取所有参数的引用
    params = net.get_params()

    all_passed = True
    param_names = []
    for layer in net.layers:
        if isinstance(layer, Linear):
            param_names.extend(["W", "b"])

    for idx, (param, analytic_grad, name) in enumerate(
        zip(params, analytic_grads, param_names)
    ):
        numeric_grad = np.zeros_like(param)

        # 遍历参数的每个元素
        it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            ix = it.multi_index

            # 保存原始值
            original_value = param[ix]

            # 计算 L(w + ε)
            param[ix] = original_value + epsilon
            logits_plus = net.forward(X)
            loss_plus = net.compute_loss(logits_plus, y_onehot)

            # 计算 L(w - ε)
            param[ix] = original_value - epsilon
            logits_minus = net.forward(X)
            loss_minus = net.compute_loss(logits_minus, y_onehot)

            # 恢复原始值
            param[ix] = original_value

            # 数值梯度 = (L(w+ε) - L(w-ε)) / (2ε)
            numeric_grad[ix] = (loss_plus - loss_minus) / (2.0 * epsilon)

            it.iternext()

        # 计算相对误差
        abs_diff = np.abs(analytic_grad - numeric_grad)
        abs_sum = np.maximum(np.abs(analytic_grad) + np.abs(numeric_grad), 1e-8)
        relative_error = np.max(abs_diff / abs_sum)

        passed = relative_error < 1e-5
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"    参数 {name} (形状 {param.shape}): "
              f"相对误差 = {relative_error:.2e} [{status}]")

    return all_passed


# 运行梯度检查
print("\n  对两层网络执行梯度检查...")
np.random.seed(0)
check_net = MLP([2, 4, 3], activation="relu")
X_check = np.random.randn(5, 2)
y_check = np.array([0, 1, 2, 0, 1])
y_check_oh = np.eye(3)[y_check]

passed = gradient_check(check_net, X_check, y_check_oh)
if passed:
    print("\n  所有梯度检查通过！反向传播实现正确！")
else:
    print("\n  存在梯度检查失败，请检查反向传播实现！")


# ════════════════════════════════════════════════════════════════════
# 第6部分：生成螺旋数据集（经典的非线性分类问题）
# ════════════════════════════════════════════════════════════════════
#
# 螺旋数据集是验证非线性分类器的经典数据集：
# - 三个螺旋臂交织在一起
# - 线性分类器完全无法解决
# - 需要至少一个隐藏层的神经网络才能正确分类
#

print("\n" + "=" * 60)
print("第6部分：生成螺旋数据集")
print("=" * 60)


def generate_spiral_data(n_points_per_class=100, n_classes=3, noise=0.3):
    """
    生成螺旋数据集：n_classes 条螺旋臂，每条 n_points_per_class 个点。

    参数:
        n_points_per_class : 每类的样本数
        n_classes          : 类别数（螺旋臂数）
        noise              : 噪声水平

    返回:
        X : 形状 (N, 2)，N = n_points_per_class * n_classes
        y : 形状 (N,)，类别标签 0, 1, ..., n_classes-1
    """
    N = n_points_per_class * n_classes
    X = np.zeros((N, 2))
    y = np.zeros(N, dtype=int)

    for c in range(n_classes):
        start = c * n_points_per_class
        end = start + n_points_per_class

        # 每条螺旋臂：从圆心向外旋转
        r = np.linspace(0.0, 1.0, n_points_per_class)  # 半径从0到1
        theta = (np.linspace(c * 4, (c + 1) * 4, n_points_per_class)
                 + np.random.randn(n_points_per_class) * noise)

        X[start:end, 0] = r * np.sin(theta)
        X[start:end, 1] = r * np.cos(theta)
        y[start:end] = c

    return X, y


# 生成数据
X_spiral, y_spiral = generate_spiral_data(n_points_per_class=100, n_classes=3)
y_spiral_oh = np.eye(3)[y_spiral]  # 转为 one-hot

print(f"  数据形状: X={X_spiral.shape}, y={y_spiral.shape}")
print(f"  类别分布: {np.bincount(y_spiral)}")

# 可视化螺旋数据
plt.figure(figsize=(7, 7))
colors_map = ["#e74c3c", "#3498db", "#2ecc71"]
for c in range(3):
    mask = y_spiral == c
    plt.scatter(X_spiral[mask, 0], X_spiral[mask, 1],
                c=colors_map[c], s=20, alpha=0.7, label=f"类别 {c}")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("螺旋数据集（3类）\n线性分类器无法解决，需要神经网络！")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()
plt.savefig("03_spiral_data.png", dpi=100, bbox_inches="tight")
plt.show()
print("  [图片已保存] 03_spiral_data.png")


# ════════════════════════════════════════════════════════════════════
# 第7部分：完整训练循环 —— 深度学习的核心模式
# ════════════════════════════════════════════════════════════════════
#
# 这个循环对 CNN、RNN、Transformer、GPT 全部适用！
# 区别只在于模型结构不同，训练框架完全一样。
#
#   for epoch in range(num_epochs):
#       1. logits = model.forward(X)        ← 前向传播
#       2. loss = loss_fn(logits, y)        ← 计算损失
#       3. model.backward()                 ← 反向传播
#       4. model.update_params(lr)          ← 更新参数
#

print("\n" + "=" * 60)
print("第7部分：完整训练循环 —— 训练 MLP 解决螺旋分类")
print("=" * 60)


def train_mlp(X, y_onehot, layer_dims, lr=1.0, n_epochs=1000,
              print_every=100, record_every=50):
    """
    训练 MLP 网络的完整流程。

    参数:
        X          : 训练数据，形状 (N, D)
        y_onehot   : one-hot 标签，形状 (N, C)
        layer_dims : 网络结构，如 [2, 64, 32, 3]
        lr         : 学习率
        n_epochs   : 训练轮数
        print_every: 每隔多少轮打印一次
        record_every: 每隔多少轮记录一次（用于可视化）

    返回:
        net         : 训练好的网络
        loss_hist   : 损失历史
        acc_hist    : 准确率历史
        snapshots   : 决策边界快照列表 [(epoch, net_params), ...]
    """
    net = MLP(layer_dims, activation="relu")
    loss_hist = []
    acc_hist = []
    snapshots = []  # 保存训练过程中网络参数的快照

    y_true = np.argmax(y_onehot, axis=1)

    for epoch in range(n_epochs):
        # ============ 训练四步曲 ============

        # 第1步：前向传播
        logits = net.forward(X)

        # 第2步：计算损失
        loss = net.compute_loss(logits, y_onehot)

        # 第3步：反向传播
        net.backward()

        # 第4步：更新参数
        net.update_params(lr)

        # ============ 记录与监控 ============
        y_pred = np.argmax(logits, axis=1)
        acc = np.mean(y_pred == y_true)
        loss_hist.append(loss)
        acc_hist.append(acc)

        # 保存快照（深拷贝参数）
        if epoch % record_every == 0 or epoch == n_epochs - 1:
            snapshot_params = []
            for layer in net.layers:
                if isinstance(layer, Linear):
                    snapshot_params.append((layer.W.copy(), layer.b.copy()))
            snapshots.append((epoch, snapshot_params))

        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>5d}/{n_epochs} | "
                  f"损失: {loss:.4f} | 准确率: {acc:.2%}")

    return net, loss_hist, acc_hist, snapshots


# 训练！
net, loss_hist, acc_hist, snapshots = train_mlp(
    X_spiral, y_spiral_oh,
    layer_dims=[2, 64, 64, 3],  # 两个隐藏层，各64个神经元
    lr=1.0,
    n_epochs=2000,
    print_every=200,
    record_every=200,
)

print(f"\n  最终损失: {loss_hist[-1]:.4f}")
print(f"  最终准确率: {acc_hist[-1]:.2%}")


# ════════════════════════════════════════════════════════════════════
# 第8部分：训练过程可视化
# ════════════════════════════════════════════════════════════════════
#
# 三幅图：
#   1. 损失曲线 —— 损失是否在持续下降？
#   2. 准确率曲线 —— 模型学到了多少？
#   3. 决策边界演化 —— 网络是如何逐步学会分类的？
#

print("\n" + "=" * 60)
print("第8部分：训练过程可视化")
print("=" * 60)


def plot_decision_boundary(net, X, y, ax, title=""):
    """
    在给定的 ax 上绘制网络的决策边界。

    原理：在输入空间的网格上密集采样，用网络预测每个点的类别，
    然后用不同颜色填充，得到决策边界的可视化。

    参数:
        net   : MLP 网络
        X     : 训练数据
        y     : 标签
        ax    : matplotlib 的 Axes 对象
        title : 子图标题
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # 创建网格
    h = 0.02  # 网格步长
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # 形状 (网格点数, 2)

    # 预测每个网格点的类别
    preds = net.predict(grid_points)
    preds = preds.reshape(xx.shape)

    # 绘制决策区域
    colors_fill = ["#FFCCCC", "#CCE5FF", "#CCFFCC"]
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(colors_fill)
    ax.contourf(xx, yy, preds, alpha=0.4, cmap=cmap_bg, levels=[-0.5, 0.5, 1.5, 2.5])

    # 绘制训练数据
    colors_scatter = ["#e74c3c", "#3498db", "#2ecc71"]
    for c in range(3):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_scatter[c],
                   s=10, alpha=0.6, edgecolors="k", linewidths=0.3)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# --- 图1：损失曲线和准确率曲线 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(loss_hist, color="steelblue", linewidth=1.5, alpha=0.8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("交叉熵损失")
axes[0].set_title("训练损失曲线")
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f"最终损失: {loss_hist[-1]:.4f}",
                 xy=(len(loss_hist) - 1, loss_hist[-1]),
                 xytext=(len(loss_hist) * 0.5, loss_hist[-1] + 0.2),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9)

axes[1].plot(acc_hist, color="coral", linewidth=1.5, alpha=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("准确率")
axes[1].set_title("训练准确率曲线")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)
axes[1].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="100%")
axes[1].annotate(f"最终准确率: {acc_hist[-1]:.2%}",
                 xy=(len(acc_hist) - 1, acc_hist[-1]),
                 xytext=(len(acc_hist) * 0.4, acc_hist[-1] - 0.15),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=9)
axes[1].legend()

plt.tight_layout()
plt.savefig("03_training_curves.png", dpi=100, bbox_inches="tight")
plt.show()
print("  [图片已保存] 03_training_curves.png")


# --- 图2：决策边界演化 ---
# 从快照中恢复网络参数，画出不同训练阶段的决策边界
n_snapshots = len(snapshots)
n_cols = min(n_snapshots, 5)
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
if n_cols == 1:
    axes = [axes]

for idx in range(n_cols):
    # 均匀选取快照
    snap_idx = idx * (n_snapshots - 1) // (n_cols - 1) if n_cols > 1 else 0
    epoch_num, saved_params = snapshots[snap_idx]

    # 创建临时网络并加载保存的参数
    temp_net = MLP([2, 64, 64, 3], activation="relu")
    param_idx = 0
    for layer in temp_net.layers:
        if isinstance(layer, Linear):
            layer.W = saved_params[param_idx][0].copy()
            layer.b = saved_params[param_idx][1].copy()
            param_idx += 1

    plot_decision_boundary(temp_net, X_spiral, y_spiral, axes[idx],
                           title=f"Epoch {epoch_num}")

plt.suptitle("决策边界的演化过程\n从随机猜测到完美分类", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("03_decision_boundary_evolution.png", dpi=100, bbox_inches="tight")
plt.show()
print("  [图片已保存] 03_decision_boundary_evolution.png")


# --- 图3：最终的决策边界（大图）---
fig, ax = plt.subplots(figsize=(8, 8))
plot_decision_boundary(net, X_spiral, y_spiral, ax,
                       title="MLP 最终决策边界\n"
                             f"网络结构: 2→64→64→3 | 准确率: {acc_hist[-1]:.2%}")
plt.tight_layout()
plt.savefig("03_final_boundary.png", dpi=100, bbox_inches="tight")
plt.show()
print("  [图片已保存] 03_final_boundary.png")


# ════════════════════════════════════════════════════════════════════
# 第9部分：深入理解 —— 隐藏层到底在做什么？
# ════════════════════════════════════════════════════════════════════
#
# 隐藏层的作用是将原始特征空间中线性不可分的数据，
# 变换到一个新的空间中使其变得线性可分。
# 这就是"表示学习"（Representation Learning）的核心思想。
#

print("\n" + "=" * 60)
print("第9部分：深入理解 —— 隐藏层的特征变换")
print("=" * 60)

# 提取第一个隐藏层的输出（中间表示）
# 网络结构: layers[0]=Linear1, layers[1]=ReLU, layers[2]=Linear2, layers[3]=ReLU, layers[4]=Linear3
hidden1_out = net.layers[0].forward(X_spiral)  # 线性变换
hidden1_act = net.layers[1].forward(hidden1_out)  # ReLU 激活
hidden2_out = net.layers[2].forward(hidden1_act)  # 第二层线性变换
hidden2_act = net.layers[3].forward(hidden2_out)  # 第二层 ReLU

# 用 PCA 将高维隐藏表示投影到2D进行可视化
def pca_2d(X):
    """简单 PCA：将 X 投影到前两个主成分"""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / X_centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 取最大的两个特征值对应的特征向量
    idx = eigenvalues.argsort()[::-1][:2]
    return X_centered @ eigenvectors[:, idx]


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
colors_scatter = ["#e74c3c", "#3498db", "#2ecc71"]

# 原始输入空间
ax = axes[0]
for c in range(3):
    mask = y_spiral == c
    ax.scatter(X_spiral[mask, 0], X_spiral[mask, 1],
               c=colors_scatter[c], s=15, alpha=0.6, label=f"类别 {c}")
ax.set_title("原始输入空间 (2D)\n三条螺旋臂纠缠在一起", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")

# 第一个隐藏层的特征空间
hidden1_2d = pca_2d(hidden1_act)
ax = axes[1]
for c in range(3):
    mask = y_spiral == c
    ax.scatter(hidden1_2d[mask, 0], hidden1_2d[mask, 1],
               c=colors_scatter[c], s=15, alpha=0.6, label=f"类别 {c}")
ax.set_title("第1隐藏层输出 (PCA→2D)\n开始分离", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

# 第二个隐藏层的特征空间
hidden2_2d = pca_2d(hidden2_act)
ax = axes[2]
for c in range(3):
    mask = y_spiral == c
    ax.scatter(hidden2_2d[mask, 0], hidden2_2d[mask, 1],
               c=colors_scatter[c], s=15, alpha=0.6, label=f"类别 {c}")
ax.set_title("第2隐藏层输出 (PCA→2D)\n接近线性可分！", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

plt.suptitle("隐藏层的作用：将纠缠的数据逐步展开为线性可分的表示", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("03_hidden_representations.png", dpi=100, bbox_inches="tight")
plt.show()
print("  [图片已保存] 03_hidden_representations.png")
print("  观察：随着层数加深，原本纠缠的螺旋臂逐渐被"展开"为可分的簇！")


# ════════════════════════════════════════════════════════════════════
# 总结：本节核心要点
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("总结：本节核心要点")
print("=" * 60)
print("""
  1. 层的设计: 每层实现 forward() 和 backward()，前者缓存中间值，
     后者利用缓存计算梯度 —— 这是所有深度学习框架的核心抽象

  2. 反向传播 = 链式法则的系统化实现：
     从损失开始，逆序遍历每一层，逐层传递梯度
     每层只需关心自己的局部梯度，不需要知道整个网络

  3. 梯度检查是必备调试工具：
     数值梯度 ≈ 解析梯度 → 实现正确
     第一次写反向传播时一定要做！

  4. Softmax + 交叉熵合并计算：
     梯度 = 预测概率 - 真实标签（one-hot），形式惊人地简洁

  5. 完整训练循环：前向传播 → 计算损失 → 反向传播 → 更新参数
     这四步对 CNN、RNN、Transformer、GPT 全部适用！

  6. 隐藏层的作用：将线性不可分的数据变换到新空间使其线性可分
     这就是"表示学习"的核心思想

  下一节预告：第2章 · 第4节 · 优化器与训练技巧
  → 学习更聪明的参数更新策略（Momentum, Adam）和正则化
""")


# ════════════════════════════════════════════════════════════════════
# 思考题
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【梯度消失与激活函数选择】
   将上面代码中的 activation 从 'relu' 改为 'sigmoid'，
   然后训练同样的螺旋数据集。你会发现：
   - 收敛速度有什么变化？
   - 最终准确率能达到一样高吗？
   为什么？提示：计算 sigmoid 导数的最大值是多少（0.25！），
   经过多层之后梯度会怎样？这就是"梯度消失"问题。

2. 【网络宽度 vs 深度】
   比较以下两种网络结构在螺旋数据集上的表现：
   - 宽而浅：[2, 256, 3]（1个隐藏层，256个神经元）
   - 窄而深：[2, 16, 16, 16, 16, 3]（4个隐藏层，各16个神经元）
   哪个参数更多？哪个表现更好？为什么？
   提示：深度能带来更抽象的特征层次，但也更难训练。

3. 【为什么需要激活函数？】
   如果去掉所有激活函数（ReLU），只保留 Linear 层，
   即网络变成 X @ W1 @ W2 @ W3，会发生什么？
   提示：多个矩阵相乘等价于一个矩阵，所以再多层也只是线性变换！
   这就是激活函数存在的根本原因——引入非线性。

4. 【学习率对训练的影响】
   用三个不同的学习率（0.01, 1.0, 10.0）分别训练，
   画出三条损失曲线。你会观察到：
   - lr=0.01：收敛但很慢
   - lr=1.0：稳定下降，效果好
   - lr=10.0：可能震荡甚至发散
   问：有没有办法在训练过程中自动调整学习率？
   （提示：这就是 Adam 优化器要解决的问题）

5. 【从 MLP 到 GPT 的距离】
   本节的 MLP 和 GPT 的训练框架完全相同（前向→损失→反向→更新）。
   那么 GPT 比 MLP 多了什么？请列出 3 个关键差异。
   提示：注意力机制、位置编码、自回归生成。
   我们会在第7章和第8章详细讲解这些。
""")

print("下一节预告: 第2章 · 第4节 · 优化器与训练技巧")
