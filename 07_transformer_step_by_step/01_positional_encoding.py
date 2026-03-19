"""
====================================================================
第7章 · 第1节 · 位置编码
====================================================================

【一句话总结】
Transformer 没有循环结构，无法感知"顺序"——位置编码为每个位置
注入唯一的位置信号，让模型知道"谁在前谁在后"。

【为什么深度学习需要这个？】
- Attention 机制是"集合操作"，对输入顺序完全不敏感
- 没有位置编码："我爱你" 和 "你爱我" 会得到完全相同的输出！
- 位置编码是 Transformer 能处理序列的关键补丁
- 现代大模型（LLaMA、Qwen）用 RoPE 替代了原始正弦编码

【核心概念】

1. 为什么需要位置信息？
   - RNN 天然有顺序（逐步处理）
   - Transformer 一次看到所有位置（并行），但丢失了顺序
   - 解决：在输入嵌入上加一个"位置信号"

2. 正弦/余弦位置编码（原始 Transformer）
   - PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
   - PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
   - 不同维度的频率不同：低维度变化快，高维度变化慢
   - 好处：可以推广到训练时未见过的长度
   - 好处：PE(pos+k) 可以表示为 PE(pos) 的线性变换（相对位置）

3. 可学习位置编码
   - 每个位置一个可训练的向量（nn.Embedding）
   - GPT-2 用的就是这种
   - 简单有效，但无法推广到超过训练长度的序列

4. RoPE（旋转位置编码）— 现代大模型标配
   - 将位置信息编码为旋转角度
   - 对 Q 和 K 的向量做旋转变换
   - 巧妙之处：两个位置的点积只依赖于相对距离
   - LLaMA、Qwen、DeepSeek 都用 RoPE
   - 数学：将 d 维向量视为 d/2 个二维平面上的向量，每个平面旋转不同角度

5. 位置编码 vs 位置嵌入
   - 正弦编码：固定的，不参与训练
   - 可学习嵌入：参与训练，但有最大长度限制
   - RoPE：不加到嵌入上，而是作用在 Q 和 K 上

【前置知识】
第6章 - 注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# 设置中文字体显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
np.random.seed(42)


# ====================================================================
# 第1部分：为什么需要位置编码
# ====================================================================
def part1_why_position_matters():
    """
    演示：没有位置编码时，Attention 对输入顺序完全不敏感。

    核心观察：
    - "我 爱 你" 和 "你 爱 我" 用同一套词嵌入，只是顺序不同
    - 没有位置编码 → Attention 的输出在打乱顺序后完全一样
    - 这说明 Attention 本身是"集合操作"，无法区分顺序
    """
    print("=" * 60)
    print("第1部分：为什么需要位置编码")
    print("=" * 60)

    d_model = 8  # 嵌入维度（小维度方便展示）

    # --- 模拟词嵌入：为每个词分配一个固定向量 ---
    torch.manual_seed(42)
    word_embeddings = {
        "我": torch.randn(d_model),
        "爱": torch.randn(d_model),
        "你": torch.randn(d_model),
    }

    # 两个句子：顺序不同，含义完全不同
    sentence_a = ["我", "爱", "你"]  # 我爱你
    sentence_b = ["你", "爱", "我"]  # 你爱我

    # 构造输入矩阵：(seq_len, d_model)
    X_a = torch.stack([word_embeddings[w] for w in sentence_a])
    X_b = torch.stack([word_embeddings[w] for w in sentence_b])

    print(f"\n句子A: {''.join(sentence_a)}")
    print(f"句子B: {''.join(sentence_b)}")
    print(f"嵌入维度: {d_model}")

    # --- 简单的自注意力（不含位置编码） ---
    # Attention(X) = softmax(X @ X^T / sqrt(d)) @ X
    def simple_attention(X):
        """不含可学习参数的简化自注意力，纯粹展示顺序不变性。"""
        d = X.shape[-1]
        scores = X @ X.T / math.sqrt(d)       # (seq_len, seq_len)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ X                    # (seq_len, d_model)
        return output, weights

    out_a, w_a = simple_attention(X_a)
    out_b, w_b = simple_attention(X_b)

    # --- 关键验证：打乱顺序后，输出是否相同？ ---
    # 句子B 相当于句子A 的第 [2, 1, 0] 重排（你爱我 → 我爱你的倒序）
    # 如果 Attention 真的对顺序不敏感：
    #   out_b 应该等于 out_a 的相应重排
    perm = [2, 1, 0]  # "你爱我" 中 "你"=A的第2个, "爱"=A的第1个, "我"=A的第0个
    out_a_permuted = out_a[perm]

    diff = torch.abs(out_b - out_a_permuted).max().item()
    print(f"\n--- 验证 Attention 的顺序不变性 ---")
    print(f"  句子A 的 Attention 输出（第0行 = '我' 的输出）:")
    print(f"    {out_a[0].tolist()[:4]}...")
    print(f"  句子B 的 Attention 输出（第2行 = '我' 的输出）:")
    print(f"    {out_b[2].tolist()[:4]}...")
    print(f"\n  max|out_B - permute(out_A)| = {diff:.10f}")
    print(f"  结论：差异为 {'零' if diff < 1e-6 else '非零'}！")
    print(f"  → Attention 完全无法区分 '我爱你' 和 '你爱我'！")
    print(f"  → 每个词的输出只取决于集合中有哪些词，与顺序无关。")

    # --- 加上位置编码后的对比 ---
    print(f"\n--- 加上位置编码后 ---")
    # 简单的位置编码：每个位置加一个不同的向量
    PE = torch.zeros(3, d_model)
    for pos in range(3):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / 10000 ** (i / d_model))
            PE[pos, i + 1] = math.cos(pos / 10000 ** (i / d_model))

    X_a_pe = X_a + PE   # 嵌入 + 位置编码
    X_b_pe = X_b + PE

    out_a_pe, _ = simple_attention(X_a_pe)
    out_b_pe, _ = simple_attention(X_b_pe)

    out_a_pe_permuted = out_a_pe[perm]
    diff_pe = torch.abs(out_b_pe - out_a_pe_permuted).max().item()
    print(f"  加上 PE 后 max|out_B - permute(out_A)| = {diff_pe:.6f}")
    print(f"  结论：差异 {'显著' if diff_pe > 0.01 else '仍然很小'}！")
    print(f"  → 位置编码让 Attention 能区分不同顺序了。")


# ====================================================================
# 第2部分：正弦位置编码
# ====================================================================
def part2_sinusoidal_pe():
    """
    实现原始 Transformer 的正弦/余弦位置编码，并用热力图可视化。

    公式：
        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    直觉：
    - 每个维度对应一个不同频率的正弦/余弦波
    - 低维度（i小）频率高 → 位置变化快 → 区分相邻位置
    - 高维度（i大）频率低 → 位置变化慢 → 编码大范围位置关系
    """
    print("\n" + "=" * 60)
    print("第2部分：正弦位置编码")
    print("=" * 60)

    class SinusoidalPositionalEncoding(nn.Module):
        """
        正弦/余弦位置编码（原始 Transformer，Vaswani et al. 2017）。

        特点：
        - 固定的，不参与训练
        - 可以推广到任意长度（训练时未见过的长度也能用）
        - PE(pos+k) 可以表示为 PE(pos) 的线性变换

        参数：
            d_model  : 嵌入维度
            max_len  : 最大序列长度（预计算用）
        """

        def __init__(self, d_model, max_len=5000):
            super().__init__()
            # 预计算整个位置编码矩阵
            pe = torch.zeros(max_len, d_model)          # (max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
            # 分母项：10000^{2i/d_model} = exp(2i * ln(10000) / d_model)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )  # (d_model/2,)

            pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos

            # 注册为 buffer（不参与训练，但随模型保存/加载）
            self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

        def forward(self, x):
            """
            输入：x，形状 (batch, seq_len, d_model)
            输出：x + PE[:seq_len]
            """
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]

    # --- 实例化并可视化 ---
    d_model = 64
    max_len = 100
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)
    pe_matrix = pe_module.pe.squeeze(0).numpy()  # (max_len, d_model)

    print(f"\n位置编码矩阵形状: ({max_len}, {d_model})")
    print(f"PE[0, :8] = {pe_matrix[0, :8].round(4)}")
    print(f"PE[1, :8] = {pe_matrix[1, :8].round(4)}")
    print(f"  → 位置 0 和位置 1 的编码明显不同")

    # --- 热力图可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：完整热力图
    ax = axes[0]
    im = ax.imshow(pe_matrix[:50, :], aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title("正弦位置编码热力图 (前50个位置)", fontsize=12, fontweight="bold")
    ax.set_xlabel("维度索引 i")
    ax.set_ylabel("位置 pos")
    plt.colorbar(im, ax=ax)

    # 右图：选取几个维度，展示不同频率
    ax = axes[1]
    dims_to_show = [0, 1, 10, 11, 30, 31, 60, 61]
    positions = np.arange(max_len)
    for dim in dims_to_show:
        kind = "sin" if dim % 2 == 0 else "cos"
        freq_idx = dim // 2
        ax.plot(positions, pe_matrix[:, dim],
                label=f"dim={dim} ({kind}, i={freq_idx})", alpha=0.8)
    ax.set_title("不同维度的位置编码波形", fontsize=12, fontweight="bold")
    ax.set_xlabel("位置 pos")
    ax.set_ylabel("编码值")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_01_part2_sinusoidal_pe.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 07_01_part2_sinusoidal_pe.png]")

    print("\n关键观察：")
    print("  - 热力图中左侧（低维度）条纹密 → 高频，变化快")
    print("  - 右侧（高维度）条纹稀 → 低频，变化慢")
    print("  - 每个位置的编码向量都是唯一的（不同条纹的组合）")

    return SinusoidalPositionalEncoding


# ====================================================================
# 第3部分：频率分析
# ====================================================================
def part3_frequency_analysis():
    """
    深入分析不同维度的频率特性。

    关键洞察：
    - 维度 2i 的波长 = 2π × 10000^{2i/d_model}
    - i=0 时波长 ≈ 6.28（区分相邻位置）
    - i=d/2-1 时波长 ≈ 62832（覆盖极长范围）
    - 类似傅里叶级数：用多个频率合成出唯一的位置"指纹"
    """
    print("\n" + "=" * 60)
    print("第3部分：频率分析")
    print("=" * 60)

    d_model = 64

    # --- 计算每个维度对的频率和波长 ---
    print(f"\n各维度对的频率和波长（d_model={d_model}）：")
    print(f"  {'维度对':>8s}  {'频率 ω':>14s}  {'波长 λ':>14s}  {'含义':>20s}")
    print(f"  {'-' * 62}")

    freq_indices = list(range(0, d_model, 2))
    wavelengths = []
    for i_pair, i in enumerate(freq_indices):
        freq = 1.0 / (10000 ** (i / d_model))
        wavelength = 2 * math.pi / freq
        wavelengths.append(wavelength)
        if i_pair < 5 or i_pair >= len(freq_indices) - 3:
            meaning = "高频，区分近距离" if wavelength < 100 else "低频，覆盖远距离"
            print(f"  ({i:2d},{i + 1:2d})   {freq:>14.8f}  {wavelength:>14.1f}  {meaning}")
        elif i_pair == 5:
            print(f"  {'...':>8s}  {'...':>14s}  {'...':>14s}")

    # --- 可视化：位置编码的"指纹"性质 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # 左图：不同位置的编码向量的余弦相似度
    ax = axes[0]
    max_pos = 50
    pe = torch.zeros(max_pos, d_model)
    position = torch.arange(0, max_pos).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 计算余弦相似度矩阵
    pe_norm = pe / pe.norm(dim=1, keepdim=True)
    sim_matrix = (pe_norm @ pe_norm.T).numpy()

    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("位置编码的余弦相似度", fontsize=12, fontweight="bold")
    ax.set_xlabel("位置 j")
    ax.set_ylabel("位置 i")
    plt.colorbar(im, ax=ax)

    # 右图：固定一个位置，看与其他位置的相似度
    ax = axes[1]
    ref_positions = [0, 10, 25]
    for ref_pos in ref_positions:
        sims = sim_matrix[ref_pos, :]
        ax.plot(range(max_pos), sims, linewidth=2,
                label=f"参考位置 = {ref_pos}")
    ax.set_title("与参考位置的余弦相似度", fontsize=12, fontweight="bold")
    ax.set_xlabel("位置")
    ax.set_ylabel("余弦相似度")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_01_part3_frequency_analysis.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 07_01_part3_frequency_analysis.png]")

    print("\n关键观察：")
    print("  - 对角线上相似度最高（自身与自身）")
    print("  - 距离越远，相似度越低 → 位置编码能区分远近")
    print("  - 相似度曲线平滑递减 → 编码中隐含了距离信息")


# ====================================================================
# 第4部分：相对位置性质
# ====================================================================
def part4_relative_position_property():
    """
    验证正弦编码的关键性质：PE(pos+k) 可以表示为 PE(pos) 的线性变换。

    数学推导：
        sin(ω(pos+k)) = sin(ωpos)cos(ωk) + cos(ωpos)sin(ωk)
        cos(ω(pos+k)) = cos(ωpos)cos(ωk) - sin(ωpos)sin(ωk)

    写成矩阵形式（对每个维度对 2i, 2i+1）：
        [PE(pos+k, 2i)  ]   [cos(ωk)  sin(ωk)] [PE(pos, 2i)  ]
        [PE(pos+k, 2i+1)] = [-sin(ωk) cos(ωk)] [PE(pos, 2i+1)]

    这是一个旋转矩阵！偏移 k 等价于旋转角度 ωk。
    """
    print("\n" + "=" * 60)
    print("第4部分：相对位置性质")
    print("=" * 60)

    d_model = 16
    max_len = 100

    # 生成位置编码
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # --- 验证线性变换性质 ---
    print(f"\n验证：PE(pos+k) = T_k @ PE(pos)（对每个维度对）")
    k = 5  # 偏移量
    pos = 10  # 参考位置

    print(f"  偏移 k = {k}, 参考位置 pos = {pos}")
    print(f"\n  对每个维度对 (2i, 2i+1)：")

    max_error = 0.0
    for i in range(d_model // 2):
        omega_k = k * div_term[i].item()
        # 构造旋转矩阵
        T_k = torch.tensor([
            [math.cos(omega_k), math.sin(omega_k)],
            [-math.sin(omega_k), math.cos(omega_k)]
        ])
        # PE(pos) 的第 (2i, 2i+1) 维
        pe_pos = pe[pos, [2 * i, 2 * i + 1]]
        # 通过旋转矩阵计算 PE(pos+k)
        pe_pos_k_predicted = T_k @ pe_pos
        # 实际的 PE(pos+k)
        pe_pos_k_actual = pe[pos + k, [2 * i, 2 * i + 1]]

        error = torch.abs(pe_pos_k_predicted - pe_pos_k_actual).max().item()
        max_error = max(max_error, error)

        if i < 3:  # 只打印前3对
            print(f"    维度对 ({2*i}, {2*i+1}): "
                  f"预测={pe_pos_k_predicted.tolist()}, "
                  f"实际={pe_pos_k_actual.tolist()}, "
                  f"误差={error:.2e}")

    print(f"    ...")
    print(f"  所有维度对的最大误差: {max_error:.2e}")
    print(f"  → PE(pos+k) 确实是 PE(pos) 的线性（旋转）变换！")

    # --- 验证不同 pos 和 k 的组合 ---
    print(f"\n  验证多个 (pos, k) 组合：")
    for pos in [0, 20, 50]:
        for k in [1, 3, 10]:
            if pos + k >= max_len:
                continue
            max_err = 0.0
            for i in range(d_model // 2):
                omega_k = k * div_term[i].item()
                T_k = torch.tensor([
                    [math.cos(omega_k), math.sin(omega_k)],
                    [-math.sin(omega_k), math.cos(omega_k)]
                ])
                pe_pos = pe[pos, [2 * i, 2 * i + 1]]
                predicted = T_k @ pe_pos
                actual = pe[pos + k, [2 * i, 2 * i + 1]]
                max_err = max(max_err, torch.abs(predicted - actual).max().item())
            print(f"    pos={pos:2d}, k={k:2d}: 最大误差 = {max_err:.2e}  PASS")

    print(f"\n  意义：模型可以通过学习线性变换来捕捉相对位置关系！")
    print(f"  → 不需要显式编码 |pos_i - pos_j|，点积中已隐含相对距离信息。")


# ====================================================================
# 第5部分：可学习位置编码
# ====================================================================
def part5_learnable_pe(SinusoidalPE):
    """
    可学习位置编码：每个位置一个可训练的向量（nn.Embedding）。

    对比：
    - GPT-2 使用可学习位置编码（最大 1024 个位置）
    - 原始 Transformer 使用固定正弦编码
    - 论文实验显示两者效果几乎一样
    - 可学习编码的缺点：无法推广到训练时未见过的长度
    """
    print("\n" + "=" * 60)
    print("第5部分：可学习位置编码")
    print("=" * 60)

    class LearnablePositionalEncoding(nn.Module):
        """
        可学习位置编码（GPT-2 风格）。

        每个位置有一个独立的可训练向量。
        本质上就是一个 nn.Embedding，位置索引作为输入。

        参数：
            d_model : 嵌入维度
            max_len : 最大序列长度
        """

        def __init__(self, d_model, max_len=512):
            super().__init__()
            self.position_embedding = nn.Embedding(max_len, d_model)

        def forward(self, x):
            """
            输入：x，形状 (batch, seq_len, d_model)
            输出：x + position_embedding
            """
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device)  # (seq_len,)
            pos_emb = self.position_embedding(positions)         # (seq_len, d_model)
            return x + pos_emb

    # --- 实例化并对比 ---
    d_model = 64
    max_len = 100

    learnable_pe = LearnablePositionalEncoding(d_model, max_len)
    sinusoidal_pe = SinusoidalPE(d_model, max_len)

    # 可学习编码的参数量
    n_params = sum(p.numel() for p in learnable_pe.parameters())
    print(f"\n可学习位置编码参数量: {n_params}")
    print(f"  = max_len × d_model = {max_len} × {d_model} = {max_len * d_model}")
    print(f"正弦位置编码参数量: 0（固定公式，无需训练）")

    # --- 对比两种编码的结构 ---
    print(f"\n--- 对比两种编码 ---")

    # 获取可学习编码（初始化是随机的）
    with torch.no_grad():
        learn_matrix = learnable_pe.position_embedding.weight.numpy()
    sin_matrix = sinusoidal_pe.pe.squeeze(0)[:max_len].numpy()

    # 分别计算余弦相似度矩阵
    def cosine_sim_matrix(M):
        """计算行向量之间的余弦相似度矩阵。"""
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M_normed = M / (norms + 1e-8)
        return M_normed @ M_normed.T

    sim_sin = cosine_sim_matrix(sin_matrix)
    sim_learn = cosine_sim_matrix(learn_matrix)

    # --- 可视化对比 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    im = ax.imshow(sim_sin[:50, :50], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("正弦编码的位置相似度", fontsize=12, fontweight="bold")
    ax.set_xlabel("位置 j")
    ax.set_ylabel("位置 i")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    im = ax.imshow(sim_learn[:50, :50], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("可学习编码的位置相似度（随机初始化）", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("位置 j")
    ax.set_ylabel("位置 i")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("07_01_part5_learnable_pe.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 07_01_part5_learnable_pe.png]")

    print("\n对比总结：")
    print("  正弦编码：")
    print("    - 相似度矩阵有明显的距离模式（近的更相似）")
    print("    - 不需要训练，可以推广到任意长度")
    print("  可学习编码：")
    print("    - 初始化时没有距离模式（随机），需要通过训练学习")
    print("    - 训练后通常也能学到类似的距离模式")
    print("    - 受限于 max_len，超出范围无法外推")


# ====================================================================
# 第6部分：RoPE 实现
# ====================================================================
def part6_rope():
    """
    RoPE（Rotary Position Embedding）—— 旋转位置编码。

    核心思想：
    1. 将 d 维向量视为 d/2 个二维向量
    2. 每个二维平面上，按位置 pos 旋转 θ_i × pos 角度
    3. θ_i = 1 / 10000^{2i/d}（和正弦编码的频率一样）
    4. 只作用在 Q 和 K 上，不加到嵌入上

    为什么旋转有效？
    - 旋转后 Q(pos1) · K(pos2) 只依赖 pos1 - pos2
    - 即内积只包含相对位置信息（而非绝对位置）
    - 这正是注意力机制真正需要的！

    数学：
        对于向量 [x_0, x_1]（一个二维平面）：
        RoPE(x, pos) = [x_0 cos(θ·pos) - x_1 sin(θ·pos),
                         x_0 sin(θ·pos) + x_1 cos(θ·pos)]
    """
    print("\n" + "=" * 60)
    print("第6部分：RoPE 实现")
    print("=" * 60)

    def precompute_freqs(dim, max_len=512, base=10000.0):
        """
        预计算 RoPE 的频率和角度。

        参数：
            dim     : 向量维度（必须为偶数）
            max_len : 最大序列长度
            base    : 频率基数，默认 10000

        返回：
            cos_cached : (max_len, dim/2) 的余弦缓存
            sin_cached : (max_len, dim/2) 的正弦缓存
        """
        # θ_i = 1 / base^{2i/dim}, i = 0, 1, ..., dim/2 - 1
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        # 位置 × 频率 → 角度矩阵
        positions = torch.arange(max_len).float()   # (max_len,)
        angles = torch.outer(positions, freqs)       # (max_len, dim/2)
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        return cos_cached, sin_cached

    def apply_rope(x, cos_cache, sin_cache):
        """
        对输入向量应用 RoPE 旋转。

        参数：
            x         : 输入，形状 (..., seq_len, dim)
            cos_cache : (max_len, dim/2) 余弦缓存
            sin_cache : (max_len, dim/2) 正弦缓存

        返回：
            旋转后的向量，形状同 x

        实现技巧：
            把 [x_0, x_1, x_2, x_3, ...] 分成两半
            前半部分 x_even = [x_0, x_2, ...]，后半部分 x_odd = [x_1, x_3, ...]
            旋转后拼回去
        """
        seq_len = x.shape[-2]
        dim = x.shape[-1]
        half_dim = dim // 2

        # 取出当前序列长度对应的 cos/sin
        cos_val = cos_cache[:seq_len]   # (seq_len, dim/2)
        sin_val = sin_cache[:seq_len]   # (seq_len, dim/2)

        # 将 x 拆成两半（相邻维度配对）
        x_even = x[..., :half_dim]      # (..., seq_len, dim/2)
        x_odd = x[..., half_dim:]       # (..., seq_len, dim/2)

        # 应用旋转：
        # x_even_new = x_even * cos - x_odd * sin
        # x_odd_new  = x_even * sin + x_odd * cos
        x_even_new = x_even * cos_val - x_odd * sin_val
        x_odd_new = x_even * sin_val + x_odd * cos_val

        return torch.cat([x_even_new, x_odd_new], dim=-1)

    # --- 演示 RoPE ---
    dim = 8
    max_len = 64
    cos_cache, sin_cache = precompute_freqs(dim, max_len)

    print(f"\nRoPE 参数：")
    print(f"  向量维度: {dim}")
    print(f"  二维平面数: {dim // 2}")
    print(f"  最大序列长度: {max_len}")

    # 构造一个简单的输入
    torch.manual_seed(42)
    x = torch.randn(1, 4, dim)  # (batch=1, seq_len=4, dim=8)
    x_rotated = apply_rope(x, cos_cache, sin_cache)

    print(f"\n原始向量 x[0, 0]: {x[0, 0].tolist()}")
    print(f"旋转后   x[0, 0]: {x_rotated[0, 0].tolist()}")
    print(f"  → 位置 0 的旋转角度为 0，所以 cos=1, sin=0")

    # 验证位置 0 不变
    diff_pos0 = torch.abs(x[0, 0, :dim // 2] - x_rotated[0, 0, :dim // 2]).max().item()
    print(f"  → 前半部分不变: max_diff = {diff_pos0:.6f}")

    print(f"\n原始向量 x[0, 1]: {x[0, 1].tolist()}")
    print(f"旋转后   x[0, 1]: {x_rotated[0, 1].tolist()}")
    print(f"  → 位置 1 被旋转了不同角度")

    # 验证旋转保范性质（旋转不改变向量长度）
    print(f"\n--- 验证旋转保范性 ---")
    for pos in range(4):
        norm_before = x[0, pos].norm().item()
        norm_after = x_rotated[0, pos].norm().item()
        print(f"  位置 {pos}: ||x||={norm_before:.6f}, "
              f"||RoPE(x)||={norm_after:.6f}, "
              f"差异={abs(norm_before - norm_after):.2e}")
    print(f"  → 旋转不改变向量长度（保范变换）！")

    return precompute_freqs, apply_rope


# ====================================================================
# 第7部分：RoPE 的相对位置性质
# ====================================================================
def part7_rope_relative_position(precompute_freqs, apply_rope):
    """
    验证 RoPE 的核心性质：Q(pos1) · K(pos2) 只依赖于 pos1 - pos2。

    推导：
        设 q = RoPE(q_raw, pos1), k = RoPE(k_raw, pos2)
        q · k = q_raw^T R(pos1)^T R(pos2) k_raw
              = q_raw^T R(pos2 - pos1) k_raw

    因为旋转矩阵的性质：R(a)^T R(b) = R(b - a)
    所以内积只依赖于位置差 pos2 - pos1！
    """
    print("\n" + "=" * 60)
    print("第7部分：RoPE 的相对位置性质")
    print("=" * 60)

    dim = 16
    max_len = 100
    cos_cache, sin_cache = precompute_freqs(dim, max_len)

    # 固定的 q_raw 和 k_raw 向量
    torch.manual_seed(123)
    q_raw = torch.randn(dim)
    k_raw = torch.randn(dim)

    print(f"\n固定 q_raw 和 k_raw 向量，改变它们的绝对位置：")
    print(f"  q_raw = {q_raw[:4].tolist()}...")
    print(f"  k_raw = {k_raw[:4].tolist()}...")

    # --- 测试：相同相对距离，不同绝对位置 ---
    print(f"\n--- 验证：相同的 Δpos → 相同的内积 ---")
    relative_dist = 5
    print(f"  固定相对距离 Δpos = {relative_dist}")

    dot_products = []
    for pos_q in range(0, 50, 10):
        pos_k = pos_q + relative_dist
        # 对 q_raw 应用位置 pos_q 的旋转
        q_input = q_raw.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)
        k_input = k_raw.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)

        # 构造只包含目标位置的 cos/sin
        cos_q = cos_cache[pos_q:pos_q + 1]
        sin_q = sin_cache[pos_q:pos_q + 1]
        cos_k = cos_cache[pos_k:pos_k + 1]
        sin_k = sin_cache[pos_k:pos_k + 1]

        q_rotated = apply_rope(q_input, cos_q, sin_q).squeeze()
        k_rotated = apply_rope(k_input, cos_k, sin_k).squeeze()

        dot = (q_rotated @ k_rotated).item()
        dot_products.append(dot)
        print(f"  pos_q={pos_q:2d}, pos_k={pos_k:2d} (Δ={relative_dist}): "
              f"Q·K = {dot:.8f}")

    max_variation = max(dot_products) - min(dot_products)
    print(f"\n  内积变化范围: {max_variation:.2e}")
    print(f"  → 相同相对距离，不同绝对位置 → 内积{'几乎不变' if max_variation < 1e-5 else '有差异'}！")

    # --- 测试：不同相对距离 ---
    print(f"\n--- 不同相对距离的内积变化 ---")
    pos_q_fixed = 10
    distances = list(range(0, 30, 3))
    dots_by_dist = []

    for delta in distances:
        pos_k = pos_q_fixed + delta
        q_input = q_raw.unsqueeze(0).unsqueeze(0)
        k_input = k_raw.unsqueeze(0).unsqueeze(0)

        cos_q = cos_cache[pos_q_fixed:pos_q_fixed + 1]
        sin_q = sin_cache[pos_q_fixed:pos_q_fixed + 1]
        cos_k = cos_cache[pos_k:pos_k + 1]
        sin_k = sin_cache[pos_k:pos_k + 1]

        q_rotated = apply_rope(q_input, cos_q, sin_q).squeeze()
        k_rotated = apply_rope(k_input, cos_k, sin_k).squeeze()

        dot = (q_rotated @ k_rotated).item()
        dots_by_dist.append(dot)

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：内积随相对距离的变化
    ax = axes[0]
    ax.plot(distances, dots_by_dist, "bo-", linewidth=2, markersize=6)
    ax.set_title("RoPE: Q·K 随相对距离变化", fontsize=12, fontweight="bold")
    ax.set_xlabel("相对距离 Δpos")
    ax.set_ylabel("Q(pos_q) · K(pos_k)")
    ax.grid(True, alpha=0.3)

    # 右图：验证绝对位置无关性——多组绝对位置，相同相对距离
    ax = axes[1]
    for delta in [1, 5, 10, 20]:
        dots = []
        positions = list(range(0, 60))
        for pq in positions:
            pk = pq + delta
            if pk >= max_len:
                break
            q_input = q_raw.unsqueeze(0).unsqueeze(0)
            k_input = k_raw.unsqueeze(0).unsqueeze(0)
            cos_q = cos_cache[pq:pq + 1]
            sin_q = sin_cache[pq:pq + 1]
            cos_k = cos_cache[pk:pk + 1]
            sin_k = sin_cache[pk:pk + 1]
            qr = apply_rope(q_input, cos_q, sin_q).squeeze()
            kr = apply_rope(k_input, cos_k, sin_k).squeeze()
            dots.append((qr @ kr).item())
        ax.plot(positions[:len(dots)], dots, linewidth=1.5,
                label=f"Δpos={delta}")

    ax.set_title("相同Δpos下，内积 vs 绝对位置", fontsize=12, fontweight="bold")
    ax.set_xlabel("q 的绝对位置")
    ax.set_ylabel("Q · K")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("07_01_part7_rope_relative.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n[图已保存: 07_01_part7_rope_relative.png]")

    print("\n核心结论：")
    print("  1. 相同相对距离 → Q·K 内积恒定（与绝对位置无关）")
    print("  2. 不同相对距离 → 内积不同，编码了距离信息")
    print("  3. 这正是注意力机制需要的：关注'谁离我近/远'")
    print("  4. RoPE 无需额外参数，优雅地将相对位置信息注入注意力计算")


# ====================================================================
# 第8部分：思考题
# ====================================================================
def part8_exercises():
    """
    思考题：检验你对位置编码的理解。
    """
    print("\n" + "=" * 60)
    print("第8部分：思考题")
    print("=" * 60)

    questions = [
        {
            "q": "正弦位置编码中，为什么偶数维度用 sin、奇数维度用 cos？\n"
                 "   如果全部用 sin 会怎样？",
            "hint": "提示：考虑位置 0 处的编码向量，以及 sin 和 cos 的值域。",
            "answer": (
                "如果全用 sin，位置 0 的编码 = 全零向量！\n"
                "    所有维度 sin(0 / ...) = 0，位置 0 没有任何位置信息。\n"
                "    sin 和 cos 搭配使用保证了：\n"
                "    1) 位置 0 也有非零编码（cos(0) = 1）\n"
                "    2) 构成旋转矩阵的基础（sin/cos 配对 → 二维旋转）\n"
                "    3) 相对位置性质依赖于三角恒等式 sin/cos 的配合"
            )
        },
        {
            "q": "RoPE 为什么不像正弦编码那样加到嵌入上，而是作用在 Q 和 K 上？",
            "hint": "提示：考虑位置信息应该影响注意力权重还是值（Value）的内容。",
            "answer": (
                "位置信息的核心作用是影响'谁该注意谁'（注意力权重），\n"
                "    即 Q·K 的内积。而 Value 代表的是内容信息。\n"
                "    如果加到嵌入上：位置信息会同时影响 Q、K、V 和前馈网络，\n"
                "    这会在内容表示中混入位置噪声。\n"
                "    RoPE 只作用在 Q 和 K 上，让位置信息精准地影响注意力权重，\n"
                "    而 Value 保持纯粹的内容表示。"
            )
        },
        {
            "q": "可学习位置编码最大长度为 1024，推理时遇到长度 2000 的序列怎么办？\n"
                 "   有哪些解决方案？",
            "hint": "提示：想想插值和外推的区别。",
            "answer": (
                "三种常见解决方案：\n"
                "    1) 位置插值（Position Interpolation）：\n"
                "       将 0~2000 映射到 0~1024，即 pos' = pos × 1024/2000\n"
                "       缺点：压缩了分辨率，需要微调\n"
                "    2) NTK-aware 插值：\n"
                "       修改 RoPE 的 base 参数（如从 10000 扩大到 40000），\n"
                "       等价于降低高频分量的频率，保留低频分量\n"
                "    3) ALiBi（Attention with Linear Biases）：\n"
                "       完全不用位置编码，而是给注意力分数加距离惩罚\n"
                "       天然支持任意长度外推"
            )
        },
        {
            "q": "如果不加任何位置编码，Transformer 的输出会有什么特点？\n"
                 "   在什么任务上可能不需要位置编码？",
            "hint": "提示：哪些任务对顺序不敏感？",
            "answer": (
                "不加位置编码时，Transformer 退化为集合函数（Set Function）：\n"
                "    输出与输入的排列顺序无关（置换等变性）。\n"
                "    适合不需要顺序的任务：\n"
                "    1) 集合分类：判断一组物品是否满足某个条件\n"
                "    2) 点云处理：3D 点的集合没有固定顺序\n"
                "    3) 图神经网络：邻居节点没有固定顺序\n"
                "    Deep Sets 和 Set Transformer 就是利用了这种无序性。"
            )
        },
        {
            "q": "RoPE 中，为什么 base 选 10000？改成 100 或 1000000 会怎样？",
            "hint": "提示：base 决定了频率的范围。想想最低频率和最高频率。",
            "answer": (
                "base 决定了最低频率的周期：\n"
                "    最高频（i=0）：ω = 1，周期 = 2π ≈ 6.28（与 base 无关）\n"
                "    最低频（i=d/2-1）：ω = 1/base，周期 = 2π × base\n"
                "    base=100：最低频周期 ≈ 628，只能覆盖约 628 个位置\n"
                "    base=10000：最低频周期 ≈ 62832，可覆盖数万个位置\n"
                "    base=1000000：周期 ≈ 6283185，可覆盖数百万位置，\n"
                "      但不同位置的编码差异太小，难以区分\n"
                "    10000 是一个在区分度和覆盖范围之间的平衡点。\n"
                "    长上下文模型（如 128K）会增大 base（YaRN、NTK 方法）。"
            )
        },
    ]

    for i, item in enumerate(questions, 1):
        print(f"\n思考题 {i}：{item['q']}")
        print(f"  {item['hint']}")
        print(f"\n  参考答案：{item['answer']}")


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   第7章 · 第1节 · 位置编码                               |")
    print("|   让 Transformer 感知顺序的关键技术                      |")
    print("+" + "=" * 58 + "+")

    part1_why_position_matters()                          # 为什么需要位置编码
    SinusoidalPE = part2_sinusoidal_pe()                  # 正弦位置编码
    part3_frequency_analysis()                            # 频率分析
    part4_relative_position_property()                    # 相对位置性质
    part5_learnable_pe(SinusoidalPE)                      # 可学习位置编码
    precompute_freqs, apply_rope = part6_rope()           # RoPE 实现
    part7_rope_relative_position(precompute_freqs,        # RoPE 相对位置性质
                                  apply_rope)
    part8_exercises()                                     # 思考题

    print("\n" + "=" * 60)
    print("本节总结")
    print("=" * 60)
    print("""
    1. Attention 是集合操作，对顺序完全不敏感——必须加位置编码
    2. 正弦编码：固定公式，不同维度对应不同频率，可外推到任意长度
    3. 正弦编码的关键性质：PE(pos+k) = 旋转矩阵 × PE(pos)
    4. 可学习编码：简单有效（nn.Embedding），但有最大长度限制
    5. RoPE：旋转位置编码，只作用在 Q/K 上，内积只依赖相对距离
    6. RoPE 是现代大模型的标配（LLaMA、Qwen、DeepSeek）

    下一节预告：第7章 · 第2节 · 多头注意力
    → Transformer 的核心组件：为什么要"多头"？每个头关注什么？
    """)
