"""
====================================================================
第7章 · 第3节 · 层归一化与残差连接
====================================================================

【一句话总结】
LayerNorm 稳定训练，残差连接让梯度畅通无阻——
这两个"小技巧"是训练深层 Transformer 的关键。

【为什么深度学习需要这个？】
- 没有 LayerNorm：深层网络的激活值可能爆炸或消失
- 没有残差连接：梯度无法有效回传到浅层
- Pre-Norm vs Post-Norm 的选择影响训练稳定性
- 这两个组件在每一个 Transformer 层都出现两次！

【核心概念】

1. Layer Normalization
   - 对每个样本的特征维度做归一化（不依赖 batch size）
   - LN(x) = γ · (x - μ) / √(σ² + ε) + β
   - μ 和 σ 在特征维度上计算（不是 batch 维度！）
   - γ (scale) 和 β (shift) 是可学习参数

2. LayerNorm vs BatchNorm
   - BatchNorm：对 batch 维度归一化，依赖 batch size
   - LayerNorm：对特征维度归一化，不依赖 batch size
   - Transformer 用 LayerNorm 因为：
     * 序列长度可变
     * 分布式训练中 batch size 可能很小
     * NLP 中 batch 内样本差异大

3. 残差连接（Residual Connection）
   - output = x + SubLayer(x)
   - 来自 ResNet 的思想，被 Transformer 完全继承
   - 作用：提供"梯度高速公路"，即使 SubLayer 梯度消失，梯度仍能通过 x 回传
   - 要求：SubLayer 的输入和输出维度必须相同（这就是为什么 d_model 不变）

4. Pre-Norm vs Post-Norm
   - Post-Norm（原始Transformer）：x + SubLayer(LayerNorm(x)) ← 错
     实际是：LayerNorm(x + SubLayer(x))
   - Pre-Norm（现代做法）：x + SubLayer(LayerNorm(x))
   - Pre-Norm 更稳定，不需要 warmup
   - GPT-2/3、LLaMA、Qwen 都用 Pre-Norm

5. RMSNorm（进一步简化）
   - 只做缩放，不做平移（去掉 μ 和 β）
   - RMSNorm(x) = γ · x / √(mean(x²) + ε)
   - 计算更快，效果差不多
   - LLaMA、Qwen 用 RMSNorm 替代 LayerNorm

【前置知识】
第7章第1-2节，第2章第5节 - BatchNorm
"""

import torch
import torch.nn as nn
import numpy as np

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# =====================================================================
# 演示1：LayerNorm 手写实现 —— 从零开始，对照 nn.LayerNorm
# =====================================================================

class MyLayerNorm(nn.Module):
    """手写 Layer Normalization

    核心公式：LN(x) = γ * (x - μ) / √(σ² + ε) + β

    与 BatchNorm 的关键区别：
    - BatchNorm 对 batch 维度（axis=0）计算 μ 和 σ
    - LayerNorm 对特征维度（最后几个维度）计算 μ 和 σ
    - 因此 LayerNorm 完全不依赖 batch size
    """

    def __init__(self, normalized_shape, eps=1e-5):
        """初始化 LayerNorm

        参数:
            normalized_shape: 要归一化的特征维度（通常是 d_model）
            eps: 防止除零的小常数
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        # 可学习参数：缩放因子 γ 和平移因子 β
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        """前向传播

        参数:
            x: 输入张量，最后几个维度的形状需匹配 normalized_shape
        返回:
            归一化后的张量，形状与输入相同
        """
        # 确定要归一化的维度（最后 len(normalized_shape) 个维度）
        dims = tuple(range(-len(self.normalized_shape), 0))

        # 在特征维度上计算均值和方差
        mean = x.mean(dim=dims, keepdim=True)      # 每个样本独立计算
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        # 归一化 + 仿射变换
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def demo_layernorm_from_scratch():
    """演示手写 LayerNorm 与 PyTorch 官方实现的对比"""
    print("=" * 60)
    print("演示1：LayerNorm 手写实现 —— 对照 nn.LayerNorm")
    print("=" * 60)

    # 模拟 Transformer 输入：(batch=2, seq_len=4, d_model=8)
    x = torch.randn(2, 4, 8)

    # 手写版本
    my_ln = MyLayerNorm(normalized_shape=8)

    # 官方版本
    pt_ln = nn.LayerNorm(normalized_shape=8)

    # 让两者参数一致（都是初始值 γ=1, β=0，本就一致）
    my_out = my_ln(x)
    pt_out = pt_ln(x)

    # 对比结果
    diff = (my_out - pt_out).abs().max().item()
    print(f"输入形状:         {tuple(x.shape)}")
    print(f"手写版输出形状:   {tuple(my_out.shape)}")
    print(f"官方版输出形状:   {tuple(pt_out.shape)}")
    print(f"最大绝对差异:     {diff:.2e}")
    print(f"结果是否一致:     {'是' if diff < 1e-6 else '否'}")

    # 验证归一化效果：每个样本的每个位置，特征维度均值≈0，标准差≈1
    sample_mean = my_out[0, 0, :].mean().item()
    sample_std = my_out[0, 0, :].std(unbiased=False).item()
    print(f"\n归一化验证（第1个样本第1个位置）:")
    print(f"  特征均值: {sample_mean:.6f}（应接近 0）")
    print(f"  特征标准差: {sample_std:.4f}（应接近 1）")
    print()


# =====================================================================
# 演示2：LayerNorm vs BatchNorm —— 归一化维度的差异
# =====================================================================

def demo_ln_vs_bn():
    """对比 LayerNorm 和 BatchNorm 归一化的维度差异

    核心区别一图看懂：
    假设输入形状为 (batch=3, features=4)

    BatchNorm：沿 batch 维度（列方向）归一化
       样本1  [a1, b1, c1, d1]
       样本2  [a2, b2, c2, d2]  →  对每列 [a1,a2,a3] 计算 μ,σ
       样本3  [a3, b3, c3, d3]

    LayerNorm：沿特征维度（行方向）归一化
       样本1  [a1, b1, c1, d1]  →  对每行 [a1,b1,c1,d1] 计算 μ,σ
       样本2  [a2, b2, c2, d2]
       样本3  [a3, b3, c3, d3]
    """
    print("=" * 60)
    print("演示2：LayerNorm vs BatchNorm —— 归一化维度对比")
    print("=" * 60)

    # 使用 2D 输入演示：(batch=4, features=6)
    x_2d = torch.tensor([
        [1.0,  100.0, 1000.0,  2.0,  200.0, 2000.0],
        [3.0,  300.0, 3000.0,  4.0,  400.0, 4000.0],
        [5.0,  500.0, 5000.0,  6.0,  600.0, 6000.0],
        [7.0,  700.0, 7000.0,  8.0,  800.0, 8000.0],
    ])

    bn = nn.BatchNorm1d(6, affine=False)  # 不使用可学习参数，纯归一化
    ln = nn.LayerNorm(6, elementwise_affine=False)

    bn_out = bn(x_2d)
    ln_out = ln(x_2d)

    print(f"输入形状: {tuple(x_2d.shape)} —— (batch=4, features=6)")
    print(f"\nBatchNorm 对每个特征跨样本归一化（沿 batch 维度）:")
    print(f"  第1个特征 [1,3,5,7] 归一化后: {bn_out[:, 0].tolist()}")
    print(f"  → 均值={bn_out[:, 0].mean():.4f}, 标准差={bn_out[:, 0].std(unbiased=False):.4f}")

    print(f"\nLayerNorm 对每个样本跨特征归一化（沿特征维度）:")
    print(f"  第1个样本 归一化后: {ln_out[0].data.numpy().round(4).tolist()}")
    print(f"  → 均值={ln_out[0].mean():.4f}, 标准差={ln_out[0].std(unbiased=False):.4f}")

    # 关键差异：batch_size=1 时的行为
    print("\n--- batch_size=1 时的行为差异 ---")
    x_single = torch.tensor([[10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]])
    ln_single = ln(x_single)
    print(f"LayerNorm (batch=1): 正常工作，输出={ln_single.data.numpy().round(4).tolist()}")
    print("BatchNorm (batch=1): 训练模式下方差为 0，无法归一化！")
    print("→ 这就是 Transformer 选择 LayerNorm 的核心原因之一")
    print()


# =====================================================================
# 演示3：残差连接 —— 有/无残差时的梯度流对比
# =====================================================================

class BlockWithoutResidual(nn.Module):
    """无残差连接的层：output = SubLayer(x)"""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class BlockWithResidual(nn.Module):
    """有残差连接的层：output = x + SubLayer(x)

    即使 SubLayer 的梯度非常小，梯度仍能通过 x 的"高速公路"回传。
    ∂output/∂x = 1 + ∂SubLayer(x)/∂x
    → 至少有恒等项 "1"，梯度不会消失！
    """

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return x + torch.tanh(self.linear(x))


def demo_residual_connection():
    """对比有/无残差连接时梯度流过深层网络的情况"""
    print("=" * 60)
    print("演示3：残差连接 —— 梯度流对比")
    print("=" * 60)

    d_model = 32
    n_layers = 20  # 堆叠 20 层，观察梯度是否消失

    # --- 无残差连接 ---
    torch.manual_seed(42)
    layers_no_res = nn.Sequential(*[BlockWithoutResidual(d_model) for _ in range(n_layers)])
    x_no_res = torch.randn(1, d_model, requires_grad=True)
    out_no_res = layers_no_res(x_no_res)
    loss_no_res = out_no_res.sum()
    loss_no_res.backward()
    grad_norm_no_res = x_no_res.grad.norm().item()

    # --- 有残差连接 ---
    torch.manual_seed(42)
    layers_res = nn.Sequential(*[BlockWithResidual(d_model) for _ in range(n_layers)])
    x_res = torch.randn(1, d_model, requires_grad=True)
    out_res = layers_res(x_res)
    loss_res = out_res.sum()
    loss_res.backward()
    grad_norm_res = x_res.grad.norm().item()

    print(f"堆叠 {n_layers} 层后输入端的梯度范数:")
    print(f"  无残差连接: {grad_norm_no_res:.6e}")
    print(f"  有残差连接: {grad_norm_res:.6e}")
    print(f"  比值: {grad_norm_res / (grad_norm_no_res + 1e-30):.1f} 倍")
    print()

    # 逐层观察梯度范数的变化
    print("逐层梯度范数（从输出到输入方向）:")

    # 重新前向，用 hook 记录每层的梯度
    def collect_grad_norms(layers_module, label):
        """用 backward hook 收集每层梯度范数"""
        grad_norms = []

        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norms.append(grad_output[0].norm().item())

        hooks = []
        for m in layers_module:
            h = m.register_full_backward_hook(hook_fn)
            hooks.append(h)

        x = torch.randn(1, d_model, requires_grad=True)
        out = layers_module(x)
        out.sum().backward()

        for h in hooks:
            h.remove()

        return grad_norms

    torch.manual_seed(42)
    norms_no = collect_grad_norms(
        nn.Sequential(*[BlockWithoutResidual(d_model) for _ in range(n_layers)]),
        "无残差"
    )
    torch.manual_seed(42)
    norms_yes = collect_grad_norms(
        nn.Sequential(*[BlockWithResidual(d_model) for _ in range(n_layers)]),
        "有残差"
    )

    print(f"  {'层':>4s}  {'无残差':>12s}  {'有残差':>12s}")
    print(f"  {'----':>4s}  {'--------':>12s}  {'--------':>12s}")
    for i in range(0, n_layers, 4):
        no_val = norms_no[i] if i < len(norms_no) else 0
        yes_val = norms_yes[i] if i < len(norms_yes) else 0
        print(f"  {n_layers - i:>4d}  {no_val:>12.6e}  {yes_val:>12.6e}")

    print("\n结论：无残差时梯度从输出到输入逐层衰减至接近 0；")
    print("      有残差时梯度保持在合理范围，训练深层网络成为可能。")
    print()


# =====================================================================
# 演示4：Pre-Norm vs Post-Norm —— 训练稳定性对比
# =====================================================================

class PostNormBlock(nn.Module):
    """Post-Norm（原始 Transformer 论文的做法）

    结构：LayerNorm(x + SubLayer(x))
    特点：需要 learning rate warmup 才能稳定训练
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 先残差，后归一化
        return self.norm(x + self.ffn(x))


class PreNormBlock(nn.Module):
    """Pre-Norm（现代 Transformer 的做法）

    结构：x + SubLayer(LayerNorm(x))
    特点：训练更稳定，不需要 warmup
    用于：GPT-2/3、LLaMA、Qwen 等
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 先归一化，后残差
        return x + self.ffn(self.norm(x))


def demo_pre_vs_post_norm():
    """对比 Pre-Norm 和 Post-Norm 在深层网络中的训练稳定性

    实验设计：
    - 堆叠多层 Transformer Block（仅含 FFN，无注意力）
    - 用相同的随机数据训练，观察损失曲线是否发散
    - Pre-Norm 通常更稳定，尤其在层数较多、学习率较大时
    """
    print("=" * 60)
    print("演示4：Pre-Norm vs Post-Norm —— 训练稳定性")
    print("=" * 60)

    d_model, d_ff, n_layers = 64, 128, 8
    n_samples, lr, steps = 128, 1e-3, 200

    # 生成随机回归数据
    torch.manual_seed(42)
    X = torch.randn(n_samples, d_model)
    y = torch.randn(n_samples, d_model)  # 回归目标

    def train_model(block_class, label):
        """训练一个由多个 Block 堆叠而成的网络"""
        torch.manual_seed(42)
        blocks = nn.Sequential(*[block_class(d_model, d_ff) for _ in range(n_layers)])
        optimizer = torch.optim.Adam(blocks.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for step in range(steps):
            pred = blocks(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    losses_post = train_model(PostNormBlock, "Post-Norm")
    losses_pre = train_model(PreNormBlock, "Pre-Norm")

    print(f"配置: {n_layers} 层, d_model={d_model}, lr={lr}")
    print(f"  Post-Norm: 初始损失={losses_post[0]:.4f} → 最终损失={losses_post[-1]:.4f}")
    print(f"  Pre-Norm:  初始损失={losses_pre[0]:.4f} → 最终损失={losses_pre[-1]:.4f}")

    # 用更大学习率再测一次，观察稳定性差异
    lr_big = 1e-2
    print(f"\n提高学习率到 {lr_big}:")

    def train_model_big_lr(block_class, label):
        torch.manual_seed(42)
        blocks = nn.Sequential(*[block_class(d_model, d_ff) for _ in range(n_layers)])
        optimizer = torch.optim.Adam(blocks.parameters(), lr=lr_big)
        criterion = nn.MSELoss()
        losses = []
        for step in range(steps):
            pred = blocks(X)
            loss = criterion(pred, y)
            if torch.isnan(loss):
                losses.append(float('inf'))
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    losses_post_big = train_model_big_lr(PostNormBlock, "Post-Norm")
    losses_pre_big = train_model_big_lr(PreNormBlock, "Pre-Norm")

    post_final = losses_post_big[-1] if losses_post_big[-1] != float('inf') else "发散(NaN)"
    pre_final = losses_pre_big[-1] if losses_pre_big[-1] != float('inf') else "发散(NaN)"
    print(f"  Post-Norm: 最终损失={post_final}")
    print(f"  Pre-Norm:  最终损失={pre_final}")
    print("\n结论：Pre-Norm 对大学习率更鲁棒，训练更稳定。")
    print("      这就是为什么现代大模型（GPT、LLaMA）都用 Pre-Norm。")
    print()


# =====================================================================
# 演示5：RMSNorm 实现 —— 更快的归一化，LLaMA 的选择
# =====================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    与 LayerNorm 相比：
    - 去掉了减均值（re-centering）和偏置项 β
    - 只保留缩放（re-scaling）
    - 公式：RMSNorm(x) = γ * x / √(mean(x²) + ε)
    - 计算量更少，效果几乎一样
    - 被 LLaMA、Qwen 等模型采用
    """

    def __init__(self, d_model, eps=1e-6):
        """初始化 RMSNorm

        参数:
            d_model: 特征维度
            eps: 防止除零的小常数
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """前向传播

        参数:
            x: 输入张量，最后一个维度为 d_model
        返回:
            归一化后的张量
        """
        # 计算 RMS（均方根）
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 缩放
        return self.gamma * (x / rms)


def demo_rmsnorm():
    """对比 RMSNorm 和 LayerNorm 的输出差异与计算效率"""
    print("=" * 60)
    print("演示5：RMSNorm 实现 —— 对比 LayerNorm")
    print("=" * 60)

    d_model = 512
    x = torch.randn(4, 32, d_model)  # (batch=4, seq=32, d_model=512)

    rms_norm = RMSNorm(d_model)
    layer_norm = nn.LayerNorm(d_model)

    rms_out = rms_norm(x)
    ln_out = layer_norm(x)

    print(f"输入形状: {tuple(x.shape)}")
    print(f"输入统计: 均值={x.mean():.4f}, 标准差={x.std():.4f}")
    print(f"\nRMSNorm 输出: 均值={rms_out.mean():.4f}, 标准差={rms_out.std():.4f}")
    print(f"LayerNorm 输出: 均值={ln_out.mean():.4f}, 标准差={ln_out.std():.4f}")

    # 计算效率对比
    import time

    n_iters = 1000
    x_bench = torch.randn(8, 128, d_model)

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = rms_norm(x_bench)
    time_rms = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iters):
        _ = layer_norm(x_bench)
    time_ln = time.perf_counter() - start

    print(f"\n计算效率（{n_iters} 次前向传播）:")
    print(f"  RMSNorm:   {time_rms:.4f}s")
    print(f"  LayerNorm: {time_ln:.4f}s")
    print(f"  RMSNorm 相对速度: {time_ln / time_rms:.2f}x")

    # 关键区别：RMSNorm 没有减均值
    print("\n关键区别：")
    print(f"  LayerNorm 参数量: γ({d_model}) + β({d_model}) = {2 * d_model}")
    print(f"  RMSNorm 参数量:   γ({d_model}) = {d_model}")
    print("  RMSNorm 省略了减均值和偏置，计算更简单但效果相当。")
    print()


# =====================================================================
# 演示6：组合使用 —— LN + 残差在 Transformer 层中的完整协作
# =====================================================================

class TransformerBlockDemo(nn.Module):
    """简化的 Transformer Block（Pre-Norm 风格）

    完整结构：
        x → [LayerNorm → Self-Attention → + 残差]
          → [LayerNorm → FFN            → + 残差]

    本演示用简单线性层代替 Self-Attention，聚焦 LN + 残差的协作。
    """

    def __init__(self, d_model, d_ff, use_norm=True, use_residual=True):
        """初始化 Transformer Block

        参数:
            d_model: 模型维度
            d_ff: FFN 中间层维度
            use_norm: 是否使用 LayerNorm
            use_residual: 是否使用残差连接
        """
        super().__init__()
        self.use_norm = use_norm
        self.use_residual = use_residual

        # 模拟自注意力子层（简化为线性变换）
        self.attn = nn.Linear(d_model, d_model)
        # FFN 子层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        # 两个 LayerNorm（Pre-Norm 风格下每个子层前各一个）
        if use_norm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """前向传播

        参数:
            x: 输入 (batch, seq_len, d_model)
        返回:
            输出 (batch, seq_len, d_model)
        """
        # --- 子层1：Self-Attention（简化） ---
        residual = x
        if self.use_norm:
            x = self.norm1(x)
        x = self.attn(x)
        if self.use_residual:
            x = residual + x
        # --- 子层2：FFN ---
        residual = x
        if self.use_norm:
            x = self.norm2(x)
        x = self.ffn(x)
        if self.use_residual:
            x = residual + x

        return x


def demo_ln_residual_combo():
    """展示 LayerNorm + 残差连接在 Transformer 中的联合效果

    对比四种配置下堆叠多层后的梯度范数和训练稳定性：
    1. 无 Norm 无残差 → 梯度消失/爆炸
    2. 有 Norm 无残差 → 梯度仍然衰减
    3. 无 Norm 有残差 → 激活值逐层增长
    4. 有 Norm 有残差 → 稳定（Transformer 的选择）
    """
    print("=" * 60)
    print("演示6：组合使用 —— LN + 残差在 Transformer 层中的协作")
    print("=" * 60)

    d_model, d_ff, n_layers = 64, 128, 12

    configs = [
        (False, False, "无Norm 无残差"),
        (True,  False, "有Norm 无残差"),
        (False, True,  "无Norm 有残差"),
        (True,  True,  "有Norm 有残差(Transformer)"),
    ]

    print(f"配置: {n_layers} 层, d_model={d_model}")
    print(f"{'配置':<28s} {'输入梯度范数':>14s} {'输出范数':>10s} {'状态':>6s}")
    print("-" * 62)

    for use_norm, use_res, label in configs:
        torch.manual_seed(42)
        blocks = nn.Sequential(
            *[TransformerBlockDemo(d_model, d_ff, use_norm, use_res)
              for _ in range(n_layers)]
        )
        x = torch.randn(2, 8, d_model, requires_grad=True)
        out = blocks(x)

        # 检查输出是否包含 NaN 或 Inf
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"  {label:<28s} {'NaN/Inf':>14s} {'NaN/Inf':>10s} {'崩溃':>6s}")
            continue

        loss = out.sum()
        loss.backward()
        grad_norm = x.grad.norm().item()
        out_norm = out.norm().item()

        if grad_norm < 1e-6:
            status = "消失"
        elif grad_norm > 1e6:
            status = "爆炸"
        else:
            status = "正常"

        print(f"  {label:<28s} {grad_norm:>14.4e} {out_norm:>10.2f} {status:>6s}")

    print("\n结论：只有 LayerNorm + 残差连接 同时使用，")
    print("      才能在深层网络中保持梯度和激活值的稳定。")
    print("      这就是 Transformer 每个子层都用 LN + 残差的原因！")
    print()


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
|  1. LayerNorm 的 γ 和 β 参数有什么作用？                      |
|     如果没有它们（即 γ=1, β=0 固定），会有什么问题？           |
|     提示：归一化后所有特征被压到 N(0,1)，模型的表达能力        |
|     是否受限？γ 和 β 让模型"学会"最合适的分布。                |
|                                                                |
|  2. 为什么残差连接要求 SubLayer 的输入输出维度相同？            |
|     如果维度不同，有什么解决办法？                              |
|     提示：ResNet 中用 1x1 卷积做维度映射；Transformer 中       |
|     通过固定 d_model 来避免这个问题。                           |
|                                                                |
|  3. Pre-Norm 结构中，最后一层输出前是否还需要一个额外的         |
|     LayerNorm？为什么 GPT-2 在最终输出前加了一个 LayerNorm？    |
|     提示：Pre-Norm 中最后一个子层的输出没有经过归一化，         |
|     直接送入预测头可能导致数值不稳定。                          |
|                                                                |
|  4. RMSNorm 去掉了减均值操作，为什么效果没有明显下降？          |
|     提示：研究表明 LayerNorm 的效果主要来自缩放（re-scaling）   |
|     而非平移（re-centering），且减均值会被后续线性层吸收。      |
|                                                                |
|  5. 如果将 Transformer 的层数从 12 增加到 96（如 GPT-3），      |
|     残差连接中反复累加 x + SubLayer(x) 会导致激活值逐渐增大。   |
|     有什么方法可以缓解？                                        |
|     提示：可以在残差连接中加一个缩放系数 α（DeepNorm），        |
|     或在初始化时缩小 SubLayer 权重（如 GPT-2 的做法）。         |
|                                                                |
+================================================================+
"""
    print(questions)


# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  第7章 · 第3节 · 层归一化与残差连接")
    print("=" * 60 + "\n")

    demo_layernorm_from_scratch()   # 演示1：LayerNorm 手写实现
    demo_ln_vs_bn()                 # 演示2：LN vs BN 维度对比
    demo_residual_connection()      # 演示3：残差连接梯度流
    demo_pre_vs_post_norm()         # 演示4：Pre-Norm vs Post-Norm
    demo_rmsnorm()                  # 演示5：RMSNorm 实现
    demo_ln_residual_combo()        # 演示6：组合使用

    print_questions()               # 思考题

    print("=" * 60)
    print("本节总结：")
    print("  1. LayerNorm 对每个样本的特征维度归一化，不依赖 batch size")
    print("  2. 残差连接提供'梯度高速公路'，让深层网络可训练")
    print("  3. Pre-Norm 比 Post-Norm 更稳定，现代大模型的标准选择")
    print("  4. RMSNorm 是 LayerNorm 的简化版，计算更快效果相当")
    print("  5. LN + 残差缺一不可，是 Transformer 每层的标配组件")
    print("=" * 60)
    print("\n下一节预告：第7章 · 第4节 · 位置编码")
    print("  → Transformer 没有循环结构，如何知道 token 的位置？\n")
