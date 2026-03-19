"""
第9章·第3节·LoRA原理与实现

核心概念:
    - Full Fine-tuning的问题: 每个任务都需存储完整模型副本，参数量巨大
    - LoRA (Low-Rank Adaptation): W = W0 + B @ A
      * W0: 冻结的原始权重 (d_out × d_in)
      * A:  低秩矩阵 (r × d_in)，用高斯初始化
      * B:  低秩矩阵 (d_out × r)，用零初始化
      * r << min(d_in, d_out)，大幅减少可训练参数
    - QLoRA: 将W0量化为4-bit，进一步节省显存
    - Adapter: 在Transformer层中插入小型瓶颈模块

依赖: pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional

# ============================================================
# 第一部分: LoRA Linear 层实现
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层

    原理:
        h = (W0 + (α/r) · B @ A) · x
        - W0: 冻结的预训练权重
        - B @ A: 低秩更新矩阵 (rank = r)
        - α: 缩放因子，控制LoRA更新的强度
        - 初始化: A ~ N(0, σ²), B = 0 → 初始时 B@A = 0，不改变原始行为

    参数:
        original_linear: 要替换的原始线性层
        rank: 低秩分解的秩 r
        alpha: 缩放因子 α
        dropout: LoRA dropout概率
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放系数

        # 冻结原始权重 W0
        self.weight = original_linear.weight  # (out, in)
        self.weight.requires_grad = False

        # 处理偏置
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        # LoRA低秩矩阵 (只有这两个参与训练)
        # A: (rank, in_features) — 用Kaiming均匀初始化
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: (out_features, rank) — 用零初始化 (保证初始输出不变)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # LoRA dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播:
            output = x @ W0^T + bias + (x @ A^T @ B^T) * scaling
        """
        # 原始线性变换 (冻结)
        base_output = F.linear(x, self.weight, self.bias)

        # LoRA增量 (可训练)
        lora_input = self.lora_dropout(x)
        lora_output = lora_input @ self.lora_A.T @ self.lora_B.T  # (batch, out)
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.2f}"
        )


# ============================================================
# 第二部分: 将LoRA应用到模型
# ============================================================

class SmallTransformerBlock(nn.Module):
    """简化的Transformer块，用于演示LoRA的应用"""

    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        # 自注意力的QKV投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        # FFN
        self.ffn_up = nn.Linear(d_model, d_model * 4)
        self.ffn_down = nn.Linear(d_model * 4, d_model)
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 简化版: 不做真正的多头注意力，只做投影
        h = self.norm1(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        attn_out = self.o_proj(v)  # 简化
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.ffn_down(F.gelu(self.ffn_up(h)))
        return x


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 8,
    alpha: float = 16.0,
) -> nn.Module:
    """
    将模型中的指定线性层替换为LoRA版本

    参数:
        model: 要应用LoRA的模型
        target_modules: 要替换的模块名称列表 (如 ['q_proj', 'v_proj'])
        rank: LoRA秩
        alpha: LoRA缩放因子
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # 默认只对Q和V应用LoRA

    for name, module in model.named_modules():
        # 检查是否是目标模块
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # 找到父模块
                parts = name.rsplit(".", 1)
                parent = model
                if len(parts) > 1:
                    for p in parts[0].split("."):
                        parent = getattr(parent, p)
                attr_name = parts[-1] if len(parts) > 1 else parts[0]

                # 替换为LoRA版本
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
                print(f"  已替换: {name} → LoRA (rank={rank})")

    # 冻结所有非LoRA参数
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    return model


# ============================================================
# 第三部分: 参数量对比
# ============================================================

def demo_parameter_comparison():
    """对比Full Fine-tuning和LoRA在不同rank下的参数量"""
    print("=" * 60)
    print("参数量对比: Full Fine-tuning vs LoRA")
    print("=" * 60)

    d_model = 128
    model = SmallTransformerBlock(d_model=d_model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n原始模型总参数量: {total_params:,}")
    print()

    ranks = [1, 2, 4, 8, 16, 32]
    lora_params_list = []

    print(f"{'Rank':>6} | {'LoRA参数量':>12} | {'占总参数%':>10} | {'节省比例':>10}")
    print("-" * 50)

    for r in ranks:
        # 计算LoRA参数量 (只对q_proj和v_proj)
        # 每个LoRA层: A(r × d_in) + B(d_out × r) = r*(d_in + d_out)
        lora_per_layer = r * (d_model + d_model)  # Q和V的输入输出维度都是d_model
        n_target_layers = 2  # q_proj, v_proj
        lora_total = lora_per_layer * n_target_layers

        ratio = lora_total / total_params * 100
        saving = (1 - lora_total / total_params) * 100
        lora_params_list.append(lora_total)

        print(f"{r:>6} | {lora_total:>12,} | {ratio:>9.2f}% | {saving:>9.2f}%")

    # 可视化
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(ranks)), lora_params_list, color="steelblue", alpha=0.8)
    plt.axhline(y=total_params, color="red", linestyle="--", label=f"Full FT: {total_params:,}")
    plt.xticks(range(len(ranks)), [f"r={r}" for r in ranks])
    plt.ylabel("可训练参数量")
    plt.title("LoRA参数量 vs Full Fine-tuning\n(对q_proj和v_proj应用LoRA)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig("lora_param_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n图表已保存为 lora_param_comparison.png")
    print()


# ============================================================
# 第四部分: LoRA训练验证
# ============================================================

def demo_lora_training():
    """简单训练demo: 验证LoRA可以有效学习"""
    print("=" * 60)
    print("LoRA训练验证")
    print("=" * 60)

    torch.manual_seed(42)
    d_model = 64
    seq_len = 16
    n_classes = 5

    # 构建模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = SmallTransformerBlock(d_model=d_model, n_heads=4)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x):
            h = self.block(x)
            return self.classifier(h.mean(dim=1))  # 平均池化后分类

    # 生成数据
    X = torch.randn(200, seq_len, d_model)
    y = torch.randint(0, n_classes, (200,))

    # --- 全量微调基线 ---
    model_full = SimpleModel()
    opt_full = torch.optim.Adam(model_full.parameters(), lr=1e-3)
    losses_full = []
    for epoch in range(60):
        loss = F.cross_entropy(model_full(X), y)
        opt_full.zero_grad()
        loss.backward()
        opt_full.step()
        losses_full.append(loss.item())

    # --- LoRA微调 ---
    model_lora = SimpleModel()
    # 加载相同的初始权重 (模拟预训练权重)
    model_lora.load_state_dict(SimpleModel().state_dict())

    print("\n应用LoRA:")
    apply_lora_to_model(
        model_lora.block, target_modules=["q_proj", "v_proj"], rank=4, alpha=8.0
    )
    # 分类头也需要训练
    for param in model_lora.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_lora.parameters())
    print(f"\n可训练参数: {trainable:,} / {total:,} ({trainable / total:.1%})")

    opt_lora = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_lora.parameters()), lr=1e-3
    )
    losses_lora = []
    for epoch in range(60):
        loss = F.cross_entropy(model_lora(X), y)
        opt_lora.zero_grad()
        loss.backward()
        opt_lora.step()
        losses_lora.append(loss.item())

    # 绘图对比
    plt.figure(figsize=(8, 5))
    plt.plot(losses_full, label="Full Fine-tuning", alpha=0.8)
    plt.plot(losses_lora, label="LoRA (rank=4)", alpha=0.8)
    plt.xlabel("训练轮次")
    plt.ylabel("损失")
    plt.title("LoRA vs Full Fine-tuning 训练曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lora_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n最终损失 — Full FT: {losses_full[-1]:.4f}, LoRA: {losses_lora[-1]:.4f}")
    print("图表已保存为 lora_training.png")
    print()


# ============================================================
# 第五部分: QLoRA 与 Adapter 概念
# ============================================================

def explain_qlora_and_adapter():
    """
    QLoRA 和 Adapter 概念讲解
    """
    print("=" * 60)
    print("QLoRA 与 Adapter 概念")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────┐
│  QLoRA (Quantized LoRA, Dettmers et al. 2023)           │
│                                                          │
│  核心改进:                                               │
│    1. 将W0量化为4-bit NormalFloat (NF4)                  │
│    2. 双重量化: 对量化常数再做量化，节省0.5bit/param     │
│    3. 分页优化器: 使用统一内存管理GPU内存峰值             │
│    4. LoRA部分仍为16-bit (BF16)，保证训练精度             │
│                                                          │
│  效果: 在单张48GB GPU上微调65B模型！                      │
│        性能几乎无损                                       │
│                                                          │
│  显存对比 (以LLaMA-7B为例):                              │
│    Full FT (FP16):     ~28 GB                            │
│    LoRA (FP16):        ~16 GB                            │
│    QLoRA (NF4+FP16):   ~6 GB                             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  Adapter (Houlsby et al. 2019)                           │
│                                                          │
│  做法: 在Transformer每层中插入小型瓶颈模块               │
│                                                          │
│    原始层输出 x                                           │
│        │                                                  │
│        ├──→ Down: Linear(d_model, r)                     │
│        │       │                                          │
│        │    Activation (GELU)                             │
│        │       │                                          │
│        │    Up: Linear(r, d_model)                        │
│        │       │                                          │
│        └───(+)─┘  (残差连接)                              │
│        │                                                  │
│    Adapter输出                                            │
│                                                          │
│  与LoRA对比:                                              │
│    - Adapter: 增加推理延迟 (串行结构)                     │
│    - LoRA:    可合并回原权重，无推理延迟 (W = W0 + BA)    │
│    - 因此LoRA在实践中更受欢迎                             │
└──────────────────────────────────────────────────────────┘
""")

    # Adapter简单实现
    class Adapter(nn.Module):
        """Adapter瓶颈模块"""
        def __init__(self, d_model: int, bottleneck: int = 16):
            super().__init__()
            self.down = nn.Linear(d_model, bottleneck)
            self.up = nn.Linear(bottleneck, d_model)
            nn.init.zeros_(self.up.weight)  # 初始化为零，保证初始时输出不变
            nn.init.zeros_(self.up.bias)

        def forward(self, x):
            return x + self.up(F.gelu(self.down(x)))

    # 演示Adapter参数量
    d = 768
    adapter = Adapter(d, bottleneck=64)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter参数量 (d={d}, bottleneck=64): {adapter_params:,}")
    print(f"  相比全连接层 ({d}×{d}={d*d:,}): {adapter_params / (d * d):.1%}")
    print()


# ============================================================
# 第六部分: LoRA合并权重演示
# ============================================================

def demo_lora_merge():
    """演示LoRA权重合并: 推理时将BA合并到W0中，消除额外延迟"""
    print("=" * 60)
    print("LoRA权重合并演示")
    print("=" * 60)

    torch.manual_seed(0)

    # 创建原始层并应用LoRA
    original = nn.Linear(64, 64)
    lora_layer = LoRALinear(original, rank=4, alpha=8.0)

    # 模拟训练: 手动修改B让LoRA有非零贡献
    with torch.no_grad():
        lora_layer.lora_B.normal_(0, 0.1)

    # 测试输入
    x = torch.randn(2, 64)

    # 方式1: 使用LoRA层推理
    out_lora = lora_layer(x)

    # 方式2: 合并权重后用普通线性层推理
    with torch.no_grad():
        merged_weight = (
            lora_layer.weight
            + lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
        )
    merged_linear = nn.Linear(64, 64, bias=lora_layer.bias is not None)
    merged_linear.weight = nn.Parameter(merged_weight)
    if lora_layer.bias is not None:
        merged_linear.bias = nn.Parameter(lora_layer.bias.clone())
    out_merged = merged_linear(x)

    # 验证两种方式输出一致
    diff = (out_lora - out_merged).abs().max().item()
    print(f"  LoRA输出与合并后输出的最大差异: {diff:.2e}")
    print(f"  结论: {'一致' if diff < 1e-5 else '不一致'} (阈值1e-5)")
    print("  → 合并后可以丢弃LoRA矩阵，推理速度与原始模型相同！")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_parameter_comparison()
    demo_lora_training()
    explain_qlora_and_adapter()
    demo_lora_merge()

    # ===========================================================
    # 思考题
    # ===========================================================
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    print("""
1. LoRA为什么用零初始化B矩阵？如果B和A都随机初始化会有什么问题？

2. LoRA的rank r如何选择？rank过大或过小分别有什么后果？
   提示: 想想参数量与表达能力的权衡。

3. 为什么LoRA通常只应用在注意力层的Q和V投影上？
   如果同时应用到K、O、FFN，效果会更好吗？

4. LoRA的一大优势是可以在推理时将BA合并到W0中。
   请问Adapter能做到这一点吗？为什么？

5. QLoRA用NF4量化W0到4-bit，但LoRA矩阵保持16-bit。
   为什么不把LoRA矩阵也量化？思考训练过程中的数值精度需求。
""")
