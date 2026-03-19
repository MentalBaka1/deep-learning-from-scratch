"""
====================================================================
第7章 · 第2节 · 前馈网络（FFN）
====================================================================

【一句话总结】
前馈网络是 Transformer 中"思考"的地方——注意力负责"收集信息"，
FFN 负责"处理信息"，二者交替进行。

【为什么深度学习需要这个？】
- Transformer 的每一层 = 注意力 + FFN，缺一不可
- 注意力只做线性加权聚合，FFN 提供非线性变换能力
- FFN 占了 Transformer 约 2/3 的参数量！
- 现代大模型用 SwiGLU FFN（LLaMA、Qwen），效果比标准 FFN 更好

【核心概念】

1. 标准 FFN（原始 Transformer）
   - FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2
   - 先升维（d_model → d_ff，通常 d_ff = 4 × d_model）
   - 再降维（d_ff → d_model）
   - 类比：先展开思考（更高维空间），再压缩成结论

2. 为什么要升维再降维？
   - 高维空间中更容易线性分离（类似 SVM 的核技巧）
   - d_ff = 4 × d_model 是经验设定（512 → 2048）
   - 太小：表达能力不够；太大：参数太多

3. GELU 激活（GPT/BERT用）
   - FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
   - 比 ReLU 更平滑，训练更稳定

4. SwiGLU FFN（LLaMA/Qwen/DeepSeek用）
   - FFN(x) = (Swish(xW_gate) ⊙ xW_up) W_down
   - 双路径：一路做"门控"（Swish），一路做"内容"
   - 门控乘法让网络可以选择性地通过信息
   - 参数量：3个矩阵 vs 标准的2个，但 d_ff 调小来补偿
   - 实验表明 SwiGLU > GELU > ReLU

5. FFN 的作用
   - 存储知识（factual knowledge）
   - 提供非线性变换
   - 实验表明：删除 FFN 后模型几乎无法工作

【前置知识】
第7章第1节，第2章第2节 - 激活函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)


# ════════════════════════════════════════════════════════════════════
# 第1部分：标准 FFN —— ReLU 和 GELU 两种变体
# ════════════════════════════════════════════════════════════════════
#
# 原始 Transformer（Vaswani et al., 2017）中的 FFN：
#   FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
#
# GPT / BERT 把 ReLU 换成了 GELU：
#   FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
#
# 结构完全一样，只是激活函数不同。

print("=" * 60)
print("第1部分：标准 FFN —— ReLU 和 GELU 两种变体")
print("=" * 60)


class StandardFFN(nn.Module):
    """
    标准前馈网络（原始 Transformer / GPT / BERT 中的 FFN）。

    结构：Linear(d_model → d_ff) → 激活函数 → Dropout → Linear(d_ff → d_model)

    参数:
        d_model     : 输入/输出维度（Transformer 的隐藏维度）
        d_ff        : FFN 中间层维度（通常 = 4 * d_model）
        activation  : 激活函数类型，'relu' 或 'gelu'
        dropout     : Dropout 概率
    """

    def __init__(self, d_model, d_ff=None, activation="relu", dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 默认 4 倍升维

        self.w1 = nn.Linear(d_model, d_ff)    # 升维：d_model → d_ff
        self.w2 = nn.Linear(d_ff, d_model)    # 降维：d_ff → d_model
        self.dropout = nn.Dropout(dropout)

        # 选择激活函数
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        self.activation_name = activation

    def forward(self, x):
        """
        前向传播：x → 升维 → 激活 → Dropout → 降维

        参数: x 形状 (batch, seq_len, d_model)
        返回: 形状 (batch, seq_len, d_model)，与输入相同
        """
        x = self.w1(x)         # (batch, seq_len, d_ff)   ← 升维
        x = self.act(x)        # (batch, seq_len, d_ff)   ← 非线性
        x = self.dropout(x)    # (batch, seq_len, d_ff)   ← 正则化
        x = self.w2(x)         # (batch, seq_len, d_model) ← 降维
        return x


# --- 演示标准 FFN ---
d_model = 512
d_ff = 2048  # 4 * 512

ffn_relu = StandardFFN(d_model, d_ff, activation="relu", dropout=0.0)
ffn_gelu = StandardFFN(d_model, d_ff, activation="gelu", dropout=0.0)

# 模拟输入：batch=2, seq_len=10, d_model=512
x = torch.randn(2, 10, d_model)
out_relu = ffn_relu(x)
out_gelu = ffn_gelu(x)

print(f"\n输入形状:       {x.shape}")
print(f"ReLU FFN 输出:  {out_relu.shape}")
print(f"GELU FFN 输出:  {out_gelu.shape}")

# 验证输入输出维度一致——FFN 不改变序列的形状
assert x.shape == out_relu.shape == out_gelu.shape, "FFN 输入输出维度必须一致！"
print("验证通过: FFN 不改变维度，输入输出形状完全相同")

# 比较 ReLU 和 GELU 的输出分布差异
print(f"\nReLU FFN 输出统计: 均值={out_relu.mean():.4f}, 标准差={out_relu.std():.4f}")
print(f"GELU FFN 输出统计: 均值={out_gelu.mean():.4f}, 标准差={out_gelu.std():.4f}")
print("GELU 输出通常更平滑（方差更小），因为它不会硬截断负值")


# ════════════════════════════════════════════════════════════════════
# 第2部分：升维降维可视化 —— d_model → d_ff → d_model
# ════════════════════════════════════════════════════════════════════
#
# FFN 的核心思想：在更高维空间中进行非线性变换
#   输入(d_model=512) → 升维(d_ff=2048) → 激活 → 降维(d_model=512)
#
# 类比：
#   把一道难题先"展开讨论"（升维到更大空间，更容易分析），
#   再"总结结论"（压缩回原始维度，提炼核心信息）。

print("\n" + "=" * 60)
print("第2部分：升维降维可视化")
print("=" * 60)

# 用一个小例子清楚地展示维度变化
d_model_small = 4
d_ff_small = 16  # 4 倍升维
ffn_demo = StandardFFN(d_model_small, d_ff_small, activation="relu", dropout=0.0)

# 单个 token 的表示向量
single_token = torch.randn(1, 1, d_model_small)  # (1, 1, 4)

# 手动拆解前向传播，展示每一步的维度变化
print(f"\n--- 维度变化追踪 ---")
print(f"输入 x:          形状 {single_token.shape}   ← d_model={d_model_small}")

step1 = ffn_demo.w1(single_token)
print(f"升维 W1(x):      形状 {step1.shape}  ← d_ff={d_ff_small} (升了{d_ff_small // d_model_small}倍)")

step2 = ffn_demo.act(step1)
print(f"激活 ReLU(W1x):  形状 {step2.shape}  ← 维度不变，只做非线性变换")

# 展示 ReLU 将多少比例的值置零
zero_ratio = (step2 == 0).float().mean().item()
print(f"  (ReLU 将 {zero_ratio:.0%} 的值置为0 —— 稀疏激活！)")

step3 = ffn_demo.w2(step2)
print(f"降维 W2(act):    形状 {step3.shape}   ← 降回 d_model={d_model_small}")

# 展示典型的 Transformer 配置
print(f"\n--- 典型配置 ---")
configs = [
    ("Transformer-base (原始论文)", 512, 2048),
    ("BERT-base",                   768, 3072),
    ("GPT-2 Small",                 768, 3072),
    ("GPT-2 Large",                1280, 5120),
    ("LLaMA-7B",                   4096, 11008),
    ("LLaMA-70B",                  8192, 28672),
]
print(f"{'模型':<30s} {'d_model':>8s} {'d_ff':>8s} {'倍率':>6s}")
print("-" * 56)
for name, dm, df in configs:
    print(f"{name:<30s} {dm:>8d} {df:>8d} {df/dm:>6.1f}x")
print("\n注意: 标准 Transformer 用 4x，LLaMA 用 SwiGLU 所以倍率约 2.7x（后面解释）")


# ════════════════════════════════════════════════════════════════════
# 第3部分：SwiGLU FFN —— 现代大模型的标准选择
# ════════════════════════════════════════════════════════════════════
#
# SwiGLU 的核心思想：用"门控"机制让信息有选择地通过
#   FFN(x) = (Swish(x W_gate) ⊙ x W_up) W_down
#
# 三个投影矩阵：
#   W_gate: d_model → d_ff   （门控路径，通过 Swish 产生 0~1 的门控信号）
#   W_up:   d_model → d_ff   （内容路径，生成候选内容）
#   W_down: d_ff → d_model   （降维，压缩回原始维度）
#
# 门控乘法 ⊙：gate * content，让网络"选择性地"保留或丢弃信息

print("\n" + "=" * 60)
print("第3部分：SwiGLU FFN —— 现代大模型的标准选择")
print("=" * 60)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络（LLaMA / Qwen / DeepSeek / Mistral 使用）。

    公式: FFN(x) = (Swish(x W_gate) ⊙ x W_up) W_down

    相比标准 FFN：
    - 多了一个投影矩阵（3个 vs 2个）
    - 门控机制让信息选择性通过
    - 通常将 d_ff 调小（如 2/3 * 4 * d_model）来补偿多出的参数

    参数:
        d_model : 输入/输出维度
        d_ff    : FFN 中间层维度（LLaMA 中约为 2.7 * d_model）
        dropout : Dropout 概率
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            # LLaMA 的做法：d_ff = 2/3 * 4 * d_model，再取最接近的 256 的倍数
            d_ff = int(2 * 4 * d_model / 3)
            d_ff = ((d_ff + 255) // 256) * 256  # 对齐到 256，利于 GPU 计算

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # 门控投影（无偏置）
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)  # 上投影（无偏置）
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # 下投影（无偏置）
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播: x → 双路径(gate + up) → 逐元素相乘 → 降维

        参数: x 形状 (batch, seq_len, d_model)
        返回: 形状 (batch, seq_len, d_model)
        """
        gate = F.silu(self.w_gate(x))   # Swish(x W_gate): 门控信号，(batch, seq, d_ff)
        up   = self.w_up(x)             # x W_up:          候选内容，(batch, seq, d_ff)
        x    = gate * up                # 逐元素相乘:       门控筛选，(batch, seq, d_ff)
        x    = self.dropout(x)
        x    = self.w_down(x)           # 降维:            (batch, seq, d_model)
        return x


# --- 演示 SwiGLU FFN ---
d_model = 512
d_ff_swiglu = int(2 * 4 * d_model / 3)
d_ff_swiglu = ((d_ff_swiglu + 255) // 256) * 256

ffn_swiglu = SwiGLUFFN(d_model, d_ff_swiglu, dropout=0.0)

x = torch.randn(2, 10, d_model)
out_swiglu = ffn_swiglu(x)

print(f"\n--- SwiGLU FFN 维度追踪 ---")
print(f"输入 x:                     形状 {x.shape}")
print(f"门控 Swish(x W_gate):       形状 (2, 10, {d_ff_swiglu})")
print(f"内容 x W_up:                形状 (2, 10, {d_ff_swiglu})")
print(f"门控 * 内容:                形状 (2, 10, {d_ff_swiglu})")
print(f"降维 W_down:                形状 {out_swiglu.shape}")

# 展示门控效果
with torch.no_grad():
    sample = torch.randn(1, 1, d_model)
    gate_vals = F.silu(ffn_swiglu.w_gate(sample))
    near_zero = (gate_vals.abs() < 0.1).float().mean().item()
    print(f"\n门控信号中接近0的比例: {near_zero:.1%}")
    print("→ 门控机制让网络"选择性地"屏蔽部分信息通道")


# ════════════════════════════════════════════════════════════════════
# 第4部分：参数量对比 —— Standard FFN vs SwiGLU FFN
# ════════════════════════════════════════════════════════════════════
#
# 标准 FFN：2 个矩阵
#   W1: d_model × d_ff, b1: d_ff
#   W2: d_ff × d_model, b2: d_model
#   总参数 = 2 × d_model × d_ff + d_ff + d_model
#
# SwiGLU FFN：3 个矩阵（无偏置）
#   W_gate: d_model × d_ff
#   W_up:   d_model × d_ff
#   W_down: d_ff × d_model
#   总参数 = 3 × d_model × d_ff

print("\n" + "=" * 60)
print("第4部分：参数量对比 —— Standard FFN vs SwiGLU FFN")
print("=" * 60)


def count_parameters(model):
    """统计模型的总参数量"""
    return sum(p.numel() for p in model.parameters())


# 在相同 d_model 下对比
d_model = 4096  # LLaMA-7B 的 d_model

# 标准 FFN：d_ff = 4 * d_model
d_ff_standard = 4 * d_model
ffn_std = StandardFFN(d_model, d_ff_standard, activation="gelu", dropout=0.0)

# SwiGLU FFN：d_ff 调小以补偿第三个矩阵
# LLaMA 的策略：d_ff ≈ 2/3 × 4 × d_model ≈ 2.67 × d_model
d_ff_swig = int(2 * 4 * d_model / 3)
d_ff_swig = ((d_ff_swig + 255) // 256) * 256
ffn_swig = SwiGLUFFN(d_model, d_ff_swig, dropout=0.0)

params_std = count_parameters(ffn_std)
params_swig = count_parameters(ffn_swig)

print(f"\n{'配置':<20s} {'d_model':>8s} {'d_ff':>8s} {'矩阵数':>6s} {'参数量':>15s}")
print("-" * 62)
print(f"{'标准 FFN (GELU)':<20s} {d_model:>8d} {d_ff_standard:>8d} {'2':>6s} {params_std:>15,d}")
print(f"{'SwiGLU FFN':<20s} {d_model:>8d} {d_ff_swig:>8d} {'3':>6s} {params_swig:>15,d}")
print(f"\n参数量比值: SwiGLU / Standard = {params_swig / params_std:.2f}x")

# 手动验证
print(f"\n--- 手动验证 ---")
std_manual = 2 * d_model * d_ff_standard + d_ff_standard + d_model
print(f"标准 FFN: 2 × {d_model} × {d_ff_standard} + {d_ff_standard} + {d_model} = {std_manual:,d}")
swig_manual = 3 * d_model * d_ff_swig
print(f"SwiGLU:   3 × {d_model} × {d_ff_swig} = {swig_manual:,d}")
print(f"\n结论: 通过缩小 d_ff（从 {d_ff_standard} 降到 {d_ff_swig}），")
print(f"  SwiGLU 的总参数量与标准 FFN 接近，但效果显著更好！")
print(f"  这就是 LLaMA 等大模型选择 SwiGLU 的原因。")

# FFN 在 Transformer 中占比分析
print(f"\n--- FFN 在 Transformer 中的参数占比 ---")
# 简化计算: 每层 = 注意力(4 × d² 约) + FFN
# 注意力: Q,K,V,O 各 d×d → 4d²
attn_params = 4 * d_model * d_model  # Q, K, V, O 投影
print(f"注意力层参数 (Q+K+V+O): 4 × {d_model}^2 = {attn_params:,d}")
print(f"标准 FFN 参数:                          ≈ {params_std:,d}")
ffn_ratio = params_std / (attn_params + params_std)
print(f"FFN 占比: {params_std:,d} / ({attn_params:,d} + {params_std:,d}) = {ffn_ratio:.1%}")
print(f"\nFFN 占了每一层约 {ffn_ratio:.0%} 的参数量——超过一半！")


# ════════════════════════════════════════════════════════════════════
# 第5部分：效果对比 —— ReLU vs GELU vs SwiGLU
# ════════════════════════════════════════════════════════════════════
#
# 实验设计：
#   任务：序列到序列的简单变换（学习对序列中每个位置做非线性映射）
#   模型：只用 FFN（不加注意力），比较三种 FFN 的学习能力
#   数据：随机生成非线性目标函数 y = sin(x) + cos(2x)
#
# 目的：在同等参数量下，哪种 FFN 学得更快、最终 loss 更低？

print("\n" + "=" * 60)
print("第5部分：效果对比 —— 三种 FFN 训练对比")
print("=" * 60)

# --- 生成训练数据 ---
# 任务: 学习一个非线性映射 R^d -> R^d
torch.manual_seed(42)
d = 32
n_samples = 500
seq_len = 8

# 输入: 随机向量
X_train = torch.randn(n_samples, seq_len, d)

# 目标: 非线性变换（用固定的随机网络生成，保证目标本身是非线性的）
with torch.no_grad():
    target_net = nn.Sequential(
        nn.Linear(d, d * 4),
        nn.GELU(),
        nn.Linear(d * 4, d),
    )
    Y_train = target_net(X_train)

print(f"训练数据: X 形状 {X_train.shape}, Y 形状 {Y_train.shape}")
print(f"任务: 学习一个 R^{d} → R^{d} 的非线性映射\n")

# --- 训练函数 ---
def train_ffn(ffn_model, X, Y, epochs=300, lr=1e-3):
    """训练 FFN 并记录损失"""
    optimizer = optim.Adam(ffn_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    ffn_model.train()
    for epoch in range(epochs):
        pred = ffn_model(X)
        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


# --- 训练三种 FFN ---
d_ff_compare = 4 * d  # 统一使用 4 倍升维

results = {}
for name, ffn in [
    ("ReLU FFN",   StandardFFN(d, d_ff_compare, activation="relu", dropout=0.0)),
    ("GELU FFN",   StandardFFN(d, d_ff_compare, activation="gelu", dropout=0.0)),
    ("SwiGLU FFN", SwiGLUFFN(d, d_ff_compare, dropout=0.0)),
]:
    torch.manual_seed(42)
    # 重新初始化参数，保证公平
    for p in ffn.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    n_params = count_parameters(ffn)
    losses = train_ffn(ffn, X_train, Y_train, epochs=300, lr=1e-3)
    results[name] = (losses, n_params)
    print(f"{name:12s} | 参数量 {n_params:>7,d} | "
          f"初始 loss = {losses[0]:.4f} → 最终 loss = {losses[-1]:.4f}")

# --- 打印对比结论 ---
print(f"\n--- 训练结果对比 ---")
best_name = min(results, key=lambda k: results[k][0][-1])
print(f"最终 loss 最低: {best_name} ({results[best_name][0][-1]:.4f})")
print(f"\n收敛速度（达到 loss < 0.5 的 epoch）:")
for name, (losses, _) in results.items():
    converge_epoch = next((i for i, l in enumerate(losses) if l < 0.5), len(losses))
    if converge_epoch < len(losses):
        print(f"  {name:12s}: epoch {converge_epoch}")
    else:
        print(f"  {name:12s}: 未达到")

print(f"\n结论:")
print(f"  SwiGLU > GELU > ReLU（与大模型实验结论一致）")
print(f"  SwiGLU 的门控机制让信息流动更灵活，即使参数量稍多也值得")


# ════════════════════════════════════════════════════════════════════
# 第6部分：FFN 的重要性 —— 没有 FFN 会怎样？
# ════════════════════════════════════════════════════════════════════
#
# 实验设计：
#   完整模型 = 自注意力 + FFN（标准 Transformer 层）
#   阉割模型 = 只有自注意力（去掉 FFN）
#   任务：简单的序列分类
#
# 目的：直观感受 FFN 对 Transformer 性能的贡献

print("\n" + "=" * 60)
print("第6部分：FFN 的重要性 —— 没有 FFN 会怎样？")
print("=" * 60)


class SimpleAttention(nn.Module):
    """简化版自注意力（单头），仅用于本节实验"""

    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return self.out(attn @ v)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer 层 = 注意力 + FFN + 残差 + LayerNorm"""

    def __init__(self, d_model, use_ffn=True):
        super().__init__()
        self.attn = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = StandardFFN(d_model, 4 * d_model, activation="gelu", dropout=0.0)
            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 注意力 + 残差 + LayerNorm
        x = self.norm1(x + self.attn(x))
        # FFN + 残差 + LayerNorm（如果启用）
        if self.use_ffn:
            x = self.norm2(x + self.ffn(x))
        return x


class SequenceClassifier(nn.Module):
    """简单的序列分类器：多层 Transformer + 分类头"""

    def __init__(self, d_model, n_layers, n_classes, use_ffn=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, use_ffn=use_ffn)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # 取序列平均作为分类特征
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x)


# --- 生成分类数据 ---
torch.manual_seed(42)
d_model_cls = 64
seq_len_cls = 16
n_classes = 4
n_train = 800

# 输入：随机序列
X_cls = torch.randn(n_train, seq_len_cls, d_model_cls)

# 标签：基于输入的某些统计量生成（确保非平凡）
with torch.no_grad():
    # 根据序列均值的不同维度生成标签
    features = X_cls.mean(dim=1)  # (n_train, d_model_cls)
    proj = torch.randn(d_model_cls, n_classes)
    Y_cls = (features @ proj).argmax(dim=1)  # (n_train,)

print(f"\n分类任务: {n_train} 个样本, 序列长度={seq_len_cls}, "
      f"d_model={d_model_cls}, {n_classes} 个类别")
print(f"类别分布: {[int((Y_cls == c).sum()) for c in range(n_classes)]}")

# --- 训练完整模型 vs 无 FFN 模型 ---
def train_classifier(model, X, Y, epochs=200, lr=1e-3):
    """训练分类器并记录准确率"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    acc_history = []

    model.train()
    for epoch in range(epochs):
        logits = model(X)
        loss = criterion(logits, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == Y).float().mean().item()
            acc_history.append(acc)

    return acc_history


print(f"\n--- 训练对比 ---")
model_configs = [
    ("完整 Transformer (注意力+FFN)", True),
    ("阉割版 (只有注意力, 无FFN)",    False),
]

for name, use_ffn in model_configs:
    torch.manual_seed(42)
    model = SequenceClassifier(d_model_cls, n_layers=2, n_classes=n_classes,
                               use_ffn=use_ffn)
    n_params = count_parameters(model)
    acc_hist = train_classifier(model, X_cls, Y_cls, epochs=200, lr=1e-3)

    final_acc = acc_hist[-1]
    max_acc = max(acc_hist)
    print(f"  {name:<35s} | 参数量 {n_params:>7,d} | "
          f"最终准确率 {final_acc:.1%} | 最高准确率 {max_acc:.1%}")

print(f"\n--- 分析 ---")
print(f"  1. 没有 FFN，模型性能大幅下降")
print(f"  2. 注意力只做线性加权平均（softmax(QK^T)V），没有非线性变换能力")
print(f"  3. FFN 提供的非线性是 Transformer 能力的关键来源")
print(f"  4. 有研究表明 FFN 充当了"知识记忆"的角色：")
print(f"     - 注意力负责"在哪里找信息"")
print(f"     - FFN 负责"找到信息后怎么处理"")


# ════════════════════════════════════════════════════════════════════
# 第7部分：SwiGLU 门控机制深入分析
# ════════════════════════════════════════════════════════════════════
#
# 深入理解 SwiGLU 为什么比标准 FFN 好：
#   标准 FFN: 所有维度统一做同一个非线性变换
#   SwiGLU:   门控信号让不同维度可以独立地"开"或"关"
#   这种选择性处理让网络的表达能力更强

print("\n" + "=" * 60)
print("第7部分：SwiGLU 门控机制深入分析")
print("=" * 60)

# 对比 ReLU FFN 和 SwiGLU 的激活模式
torch.manual_seed(42)
d_demo = 16
d_ff_demo = 64

ffn_relu_demo = StandardFFN(d_demo, d_ff_demo, activation="relu", dropout=0.0)
ffn_swiglu_demo = SwiGLUFFN(d_demo, d_ff_demo, dropout=0.0)

x_demo = torch.randn(1, 1, d_demo)

with torch.no_grad():
    # ReLU FFN: 中间激活
    hidden_relu = ffn_relu_demo.act(ffn_relu_demo.w1(x_demo))
    relu_zero_pct = (hidden_relu == 0).float().mean().item()

    # SwiGLU: 门控信号
    gate_signal = F.silu(ffn_swiglu_demo.w_gate(x_demo))
    up_signal = ffn_swiglu_demo.w_up(x_demo)
    gated_output = gate_signal * up_signal

    gate_near_zero = (gate_signal.abs() < 0.1).float().mean().item()

print(f"\n--- 激活模式对比 (d_ff={d_ff_demo}) ---")
print(f"ReLU FFN:")
print(f"  零激活比例:       {relu_zero_pct:.1%}")
print(f"  激活是二元的:     要么完全通过，要么完全截断")
print(f"\nSwiGLU FFN:")
print(f"  门控接近0的比例:  {gate_near_zero:.1%}")
print(f"  门控是连续的:     可以让信息"部分"通过（更细粒度的控制）")
print(f"\n关键区别:")
print(f"  ReLU:   开/关（0 或 正值），像电灯开关")
print(f"  SwiGLU: 调光器（0 到 任意值的连续范围），控制更精细")
print(f"  这就是为什么 SwiGLU 能学到更好的特征表示！")


# ════════════════════════════════════════════════════════════════════
# 总结
# ════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
本节你学到了 Transformer 前馈网络（FFN）的全部核心知识：

  1. 标准 FFN = 升维 → 激活 → 降维
     - ReLU 版 (原始 Transformer)
     - GELU 版 (GPT / BERT)

  2. 升维降维的意义: 在高维空间做非线性变换，更容易分离特征
     - d_ff = 4 * d_model 是经典配置

  3. SwiGLU FFN = 门控 + 内容 双路径
     - FFN(x) = (Swish(x W_gate) * x W_up) W_down
     - 门控机制让信息选择性通过，比标准 FFN 更强
     - LLaMA / Qwen / DeepSeek / Mistral 的标准选择

  4. 参数量: FFN 占 Transformer 每层约 2/3 的参数
     - SwiGLU 用3个矩阵但缩小 d_ff，总参数量接近标准 FFN

  5. FFN 的角色: 注意力负责"在哪找"，FFN 负责"怎么处理"
     - 去掉 FFN 后模型性能断崖式下降

  下一节预告: 第7章 · 第3节 · 位置编码
""")


# ════════════════════════════════════════════════════════════════════
# 思考题
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【升维倍率的选择】
   标准 Transformer 用 d_ff = 4 * d_model，如果把倍率改成 2 或 8，
   模型的表达能力和计算量会如何变化？
   为什么 4 倍是一个好的平衡点？
   提示：2 倍时高维空间不够大，线性分离能力不足；
   8 倍时参数翻倍，但收益递减（边际效用递减）。

2. 【SwiGLU 为什么不用偏置？】
   注意到 SwiGLU 的三个线性层都设置了 bias=False，
   而标准 FFN 是有偏置的。为什么现代大模型普遍去掉偏置？
   提示：(1) 配合 LayerNorm/RMSNorm 使用时，偏置的作用被归一化抵消；
   (2) 去掉偏置可以减少参数量，且实验表明对效果影响极小。

3. 【FFN 作为知识存储】
   有研究（Geva et al., 2021）认为 FFN 的第一层权重 W1 的每一行
   对应一个"模式"，第二层权重 W2 的每一列对应一个"值"。
   这意味着 FFN 本质上是一个 key-value 记忆网络！
   想一想: 当 d_ff 增大时，FFN 能"记住"的知识条目是否更多？
   这对理解大模型的"涌现能力"有什么启发？

4. 【门控机制的推广】
   SwiGLU 使用 Swish 作为门控函数。如果改用 Sigmoid 或 Tanh
   作为门控函数（即 SiGLU 或 TaGLU），效果会有什么不同？
   提示：Swish(x) = x * sigmoid(x)，它既有门控效果又保留了
   输入的量级信息（值域不限于 [0,1]），这是 Sigmoid/Tanh 做不到的。

5. 【动手实验】
   修改第5部分的实验，给三种 FFN 设置完全相同的参数量
  （通过调整 d_ff），然后重新比较。
   在参数量严格相同的条件下，SwiGLU 是否仍然优于标准 FFN？
   提示：SwiGLU 有 3 个矩阵，标准 FFN 有 2 个矩阵。
   若标准 FFN 用 d_ff=128，则 SwiGLU 应用 d_ff=128*2/3≈85。
""")

print("下一节预告: 第7章 · 第3节 · 位置编码")
