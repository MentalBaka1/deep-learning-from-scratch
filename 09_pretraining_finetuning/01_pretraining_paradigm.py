"""
第9章·第1节·预训练范式与Scaling Laws

核心概念:
    - 自监督学习: 从无标注数据中自动构造监督信号
    - MLM (Masked Language Model, BERT): 随机遮盖token，预测被遮盖的token
    - CLM (Causal Language Model, GPT): 自回归方式，预测下一个token
    - PrefixLM: 前缀部分双向注意力 + 后续部分自回归
    - Scaling Laws: 损失与计算量/数据量/参数量之间的幂律关系
    - Chinchilla最优配比: 计算预算固定时，模型大小与数据量的最优比例

依赖: pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ============================================================
# 第一部分: MLM (Masked Language Model) 任务演示
# ============================================================

class MLMDemo(nn.Module):
    """
    掩码语言模型演示 (BERT风格)

    核心思想:
        1. 随机选择15%的token进行处理
        2. 其中80%替换为[MASK], 10%替换为随机token, 10%保持不变
        3. 模型预测被选中位置的原始token
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

        # Transformer编码器 (双向注意力)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # MLM预测头
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播，输出每个位置的词汇表概率分布"""
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.encoder(x)  # 双向注意力，每个token可以看到所有token
        logits = self.mlm_head(x)
        return logits


def apply_mlm_mask(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int = 1,
    mask_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对输入应用MLM掩码策略

    规则 (遵循BERT原始论文):
        - 15%的token被选中
        - 选中的token中: 80%替换为[MASK], 10%随机替换, 10%保持不变

    返回:
        masked_ids: 掩码后的输入
        labels: 被掩码位置的真实token (-100表示不计算损失)
        mask_positions: 被选中的位置
    """
    labels = input_ids.clone()
    masked_ids = input_ids.clone()

    # 生成掩码概率矩阵
    prob_matrix = torch.full(input_ids.shape, mask_prob)
    mask_positions = torch.bernoulli(prob_matrix).bool()

    # 未被选中的位置标签设为-100 (CrossEntropy会忽略)
    labels[~mask_positions] = -100

    # 80%替换为[MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & mask_positions
    masked_ids[replace_mask] = mask_token_id

    # 10%替换为随机token
    random_mask = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
        & mask_positions
        & ~replace_mask
    )
    random_tokens = torch.randint(2, vocab_size, input_ids.shape)
    masked_ids[random_mask] = random_tokens[random_mask]

    # 剩余10%保持不变 (不需要额外操作)

    return masked_ids, labels, mask_positions


def demo_mlm():
    """演示MLM训练一个batch"""
    print("=" * 60)
    print("MLM (掩码语言模型) 演示")
    print("=" * 60)

    vocab_size = 100
    mask_token_id = 1  # [MASK]的ID
    batch_size, seq_len = 2, 16

    # 构造模拟输入 (token ID从2开始，0=PAD, 1=MASK)
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    print(f"原始输入: {input_ids[0].tolist()}")

    # 应用MLM掩码
    masked_ids, labels, mask_pos = apply_mlm_mask(input_ids, vocab_size, mask_token_id)
    print(f"掩码输入: {masked_ids[0].tolist()}")
    print(f"掩码位置: {mask_pos[0].nonzero().squeeze(-1).tolist()}")

    # 模型前向传播
    model = MLMDemo(vocab_size, d_model=64, n_heads=4, n_layers=2)
    logits = model(masked_ids)

    # 计算损失 (只在被掩码位置计算)
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
    print(f"MLM损失: {loss.item():.4f}")
    print()


# ============================================================
# 第二部分: CLM (Causal Language Model) 任务演示
# ============================================================

class CLMDemo(nn.Module):
    """
    因果语言模型演示 (GPT风格)

    核心思想:
        - 自回归: 只能看到当前位置及之前的token
        - 通过因果掩码(上三角矩阵)实现单向注意力
        - 预测下一个token: P(x_t | x_1, x_2, ..., x_{t-1})
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 因果掩码: 上三角为True表示不可见
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )

        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.decoder(x, mask=causal_mask)
        logits = self.lm_head(x)
        return logits


def demo_clm():
    """演示CLM训练一个batch"""
    print("=" * 60)
    print("CLM (因果语言模型) 演示")
    print("=" * 60)

    vocab_size = 100
    batch_size, seq_len = 2, 16

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入序列: {input_ids[0].tolist()}")

    model = CLMDemo(vocab_size, d_model=64, n_heads=4, n_layers=2)
    logits = model(input_ids)

    # CLM损失: 用位置t的输出预测位置t+1的token
    # logits[:, :-1] 对应位置 0..T-2 的预测
    # input_ids[:, 1:] 对应位置 1..T-1 的真实token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    print(f"CLM损失: {loss.item():.4f}")
    print(f"随机猜测的理论损失: {np.log(vocab_size):.4f} (ln({vocab_size}))")
    print()


# ============================================================
# 第三部分: PrefixLM 简要说明
# ============================================================

def demo_prefix_lm():
    """
    PrefixLM (前缀语言模型) 概念说明

    PrefixLM结合了MLM和CLM的特点:
        - 前缀部分: 使用双向注意力 (类似BERT)
        - 生成部分: 使用单向注意力 (类似GPT)

    典型代表: T5, UniLM, GLM

    注意力掩码示意 (prefix_len=3, total_len=6):
        位置  0  1  2  3  4  5
      0 [  1  1  1  0  0  0 ]  ← 前缀部分:
      1 [  1  1  1  0  0  0 ]    可以看到所有前缀token
      2 [  1  1  1  0  0  0 ]
      3 [  1  1  1  1  0  0 ]  ← 生成部分:
      4 [  1  1  1  1  1  0 ]    可以看到前缀 + 已生成的token
      5 [  1  1  1  1  1  1 ]
    """
    print("=" * 60)
    print("PrefixLM (前缀语言模型) 注意力掩码")
    print("=" * 60)

    total_len, prefix_len = 6, 3

    # 构造PrefixLM的注意力掩码
    # 1 = 可见, 0 = 不可见
    attn_mask = torch.zeros(total_len, total_len)
    attn_mask[:, :prefix_len] = 1  # 所有位置都能看到前缀
    for i in range(prefix_len, total_len):
        attn_mask[i, prefix_len : i + 1] = 1  # 生成部分的因果掩码

    print("注意力掩码 (1=可见, 0=不可见):")
    for i in range(total_len):
        role = "前缀" if i < prefix_len else "生成"
        row_str = " ".join([str(int(attn_mask[i, j].item())) for j in range(total_len)])
        print(f"  位置{i} ({role}): [{row_str}]")
    print()


# ============================================================
# 第四部分: Scaling Laws 可视化
# ============================================================

def demo_scaling_laws():
    """
    Scaling Laws 可视化

    核心发现 (Kaplan et al., 2020):
        L(N) = (N_c / N)^α_N  — 损失与参数量的幂律关系
        L(D) = (D_c / D)^α_D  — 损失与数据量的幂律关系
        L(C) = (C_c / C)^α_C  — 损失与计算量的幂律关系

    Chinchilla (Hoffmann et al., 2022):
        最优配比: N_opt ∝ C^0.5,  D_opt ∝ C^0.5
        即: 参数量和数据量应同步扩大
        经验法则: 每个参数约需20个token的训练数据
    """
    print("=" * 60)
    print("Scaling Laws 可视化")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --- 图1: Loss vs 参数量 N ---
    N = np.logspace(6, 11, 50)  # 1M到100B参数
    N_c, alpha_N = 8.8e13, 0.076
    L_N = (N_c / N) ** alpha_N
    axes[0].loglog(N, L_N, "b-", linewidth=2)
    axes[0].set_xlabel("参数量 N", fontsize=11)
    axes[0].set_ylabel("测试损失 L(N)", fontsize=11)
    axes[0].set_title("损失 vs 参数量", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # --- 图2: Loss vs 数据量 D ---
    D = np.logspace(8, 13, 50)  # 100M到10T tokens
    D_c, alpha_D = 5.4e13, 0.095
    L_D = (D_c / D) ** alpha_D
    axes[1].loglog(D, L_D, "r-", linewidth=2)
    axes[1].set_xlabel("数据量 D (tokens)", fontsize=11)
    axes[1].set_ylabel("测试损失 L(D)", fontsize=11)
    axes[1].set_title("损失 vs 数据量", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # --- 图3: Loss vs 计算量 C ---
    C = np.logspace(15, 24, 50)  # FLOPs
    C_c, alpha_C = 3.1e8, 0.050
    L_C = (C_c / C) ** alpha_C
    axes[2].loglog(C, L_C, "g-", linewidth=2)
    axes[2].set_xlabel("计算量 C (FLOPs)", fontsize=11)
    axes[2].set_ylabel("测试损失 L(C)", fontsize=11)
    axes[2].set_title("损失 vs 计算量", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Scaling Laws: 损失与资源的幂律关系", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("scaling_laws.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图表已保存为 scaling_laws.png")

    # Chinchilla最优配比
    print("\n--- Chinchilla 最优配比 ---")
    print("核心结论: 在固定计算预算下，参数量和数据量应同步扩大")
    print()
    compute_budgets = [1e18, 1e20, 1e22, 1e24]
    print(f"{'计算预算(FLOPs)':>18} | {'最优参数量':>14} | {'最优数据量(tokens)':>18} | {'训练token/参数':>14}")
    print("-" * 72)
    for C_budget in compute_budgets:
        # Chinchilla: N_opt ≈ 0.0592 * C^0.50,  D_opt ≈ 0.2825 * C^0.50
        # 简化: C ≈ 6ND → N_opt ≈ sqrt(C/120), D_opt ≈ 20*N_opt
        N_opt = (C_budget / 120) ** 0.5
        D_opt = 20 * N_opt
        ratio = D_opt / N_opt
        print(f"{C_budget:>18.1e} | {N_opt:>14.2e} | {D_opt:>18.2e} | {ratio:>14.1f}")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_mlm()
    demo_clm()
    demo_prefix_lm()
    demo_scaling_laws()

    # ===========================================================
    # 思考题
    # ===========================================================
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    print("""
1. MLM和CLM各有什么优缺点？为什么BERT适合理解任务而GPT适合生成任务？

2. MLM中为什么不把所有选中的token都替换为[MASK]，而是采用80/10/10的策略？
   提示: 考虑预训练与微调之间的分布差异。

3. Scaling Laws告诉我们：增大模型、增加数据、增加算力都能降低损失。
   如果你只有有限预算，应该优先扩大哪个维度？为什么？

4. Chinchilla论文指出LLaMA-65B只用了1.4T tokens训练可能不够充分。
   按照Chinchilla最优配比，65B参数的模型应该用多少tokens训练？

5. PrefixLM相比纯CLM在哪些任务上更有优势？举例说明。
""")
