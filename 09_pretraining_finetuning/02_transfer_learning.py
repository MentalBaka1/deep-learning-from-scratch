"""
第9章·第2节·迁移学习

核心概念:
    - 为什么预训练有效: 低层学通用特征，高层学任务特征
    - 特征提取 (Feature Extraction): 冻结预训练模型，只训练分类头
    - 全量微调 (Full Fine-tuning): 所有参数都参与梯度更新
    - 灾难性遗忘 (Catastrophic Forgetting): 微调后模型丧失预训练能力
    - 学习率策略: 分层学习率、warmup、较小学习率微调

依赖: pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================
# 第一部分: 预训练→微调流程概览
# ============================================================

"""
预训练→微调流程 (文字流程图):

┌─────────────────────────────────────────────────────┐
│  第一阶段: 预训练 (Pre-training)                      │
│                                                      │
│  大规模无标注语料 ──→ 自监督任务 (MLM/CLM)             │
│       (TB级别)        ──→ 预训练模型权重               │
│                                                      │
│  目标: 学习通用的语言/视觉表征                         │
│  特点: 数据量大、计算成本高、只需训练一次               │
└───────────────────────┬─────────────────────────────┘
                        │ 权重迁移
                        ▼
┌─────────────────────────────────────────────────────┐
│  第二阶段: 微调 (Fine-tuning)                         │
│                                                      │
│  任务特定标注数据 ──→ 加载预训练权重                    │
│     (少量/中等)       ──→ 在下游任务上继续训练          │
│                                                      │
│  方式A: 特征提取 — 冻结backbone，只训练分类头          │
│  方式B: 全量微调 — 所有层都更新，用较小学习率          │
│  方式C: 参数高效微调 — 只训练少量新增参数 (LoRA等)     │
└─────────────────────────────────────────────────────┘
"""


# ============================================================
# 第二部分: 模拟预训练模型
# ============================================================

class SimpleBackbone(nn.Module):
    """模拟一个预训练backbone (3层MLP作为简化)"""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, feature_dim: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DownstreamModel(nn.Module):
    """
    下游任务模型: backbone + 分类头

    参数:
        backbone: 预训练的特征提取器
        num_classes: 下游任务类别数
        freeze_backbone: 是否冻结backbone (特征提取模式)
    """

    def __init__(self, backbone: SimpleBackbone, num_classes: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = deepcopy(backbone)  # 深拷贝，不影响原始模型
        self.classifier = nn.Linear(32, num_classes)

        if freeze_backbone:
            # 特征提取模式: 冻结backbone的所有参数
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ============================================================
# 第三部分: Feature Extraction vs Fine-tuning 对比
# ============================================================

def generate_data(n_samples: int = 500, input_dim: int = 20, n_classes: int = 5):
    """生成模拟数据"""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def train_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                lr: float = 1e-3, epochs: int = 50) -> list:
    """训练模型并返回损失曲线"""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    losses = []

    for epoch in range(epochs):
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def demo_feature_extraction_vs_finetuning():
    """对比特征提取和全量微调"""
    print("=" * 60)
    print("特征提取 vs 全量微调 对比")
    print("=" * 60)

    # 模拟预训练: 先在一个"预训练任务"上训练backbone
    pretrained_backbone = SimpleBackbone()
    pretrain_model = nn.Sequential(pretrained_backbone, nn.Linear(32, 10))
    X_pretrain, y_pretrain = generate_data(1000, n_classes=10)

    optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = F.cross_entropy(pretrain_model(X_pretrain), y_pretrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"预训练完成，最终损失: {loss.item():.4f}")

    # 下游任务数据
    X_down, y_down = generate_data(200, n_classes=3)

    # 方式1: 随机初始化 (无预训练)
    random_model = DownstreamModel(SimpleBackbone(), num_classes=3)
    losses_random = train_model(random_model, X_down, y_down)

    # 方式2: 特征提取 (冻结backbone)
    fe_model = DownstreamModel(pretrained_backbone, num_classes=3, freeze_backbone=True)
    losses_fe = train_model(fe_model, X_down, y_down)

    # 方式3: 全量微调
    ft_model = DownstreamModel(pretrained_backbone, num_classes=3, freeze_backbone=False)
    losses_ft = train_model(ft_model, X_down, y_down, lr=1e-4)  # 微调用更小的学习率

    # 统计可训练参数量
    def count_trainable(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n可训练参数量:")
    print(f"  随机初始化:  {count_trainable(random_model):>6}")
    print(f"  特征提取:    {count_trainable(fe_model):>6}  (只有分类头)")
    print(f"  全量微调:    {count_trainable(ft_model):>6}  (所有参数)")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(losses_random, label="随机初始化 (无预训练)", alpha=0.8)
    plt.plot(losses_fe, label="特征提取 (冻结backbone)", alpha=0.8)
    plt.plot(losses_ft, label="全量微调 (小学习率)", alpha=0.8)
    plt.xlabel("训练轮次")
    plt.ylabel("损失")
    plt.title("迁移学习: 不同微调策略的收敛对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("transfer_learning_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n图表已保存为 transfer_learning_comparison.png")
    print()


# ============================================================
# 第四部分: 学习率对微调的影响
# ============================================================

def demo_lr_impact():
    """不同学习率对微调效果的影响"""
    print("=" * 60)
    print("学习率对微调的影响")
    print("=" * 60)

    # 预训练backbone
    backbone = SimpleBackbone()
    pretrain_model = nn.Sequential(backbone, nn.Linear(32, 10))
    X_pt, y_pt = generate_data(1000, n_classes=10)
    opt = torch.optim.Adam(pretrain_model.parameters(), lr=1e-3)
    for _ in range(100):
        loss = F.cross_entropy(pretrain_model(X_pt), y_pt)
        opt.zero_grad()
        loss.backward()
        opt.step()

    X_down, y_down = generate_data(200, n_classes=3)

    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    plt.figure(figsize=(8, 5))

    for lr in learning_rates:
        model = DownstreamModel(backbone, num_classes=3, freeze_backbone=False)
        losses = train_model(model, X_down, y_down, lr=lr, epochs=80)
        plt.plot(losses, label=f"lr={lr:.0e}", alpha=0.8)
        print(f"  lr={lr:.0e}  最终损失: {losses[-1]:.4f}")

    plt.xlabel("训练轮次")
    plt.ylabel("损失")
    plt.title("微调学习率对比\n(太大→破坏预训练特征, 太小→收敛慢)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lr_impact_finetuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n图表已保存为 lr_impact_finetuning.png")
    print()


# ============================================================
# 第五部分: 灾难性遗忘演示
# ============================================================

def demo_catastrophic_forgetting():
    """
    灾难性遗忘: 微调后模型在原始任务上性能下降

    对策:
        1. 使用较小学习率微调
        2. 正则化 (L2, EWC等)
        3. 多任务联合训练
        4. 参数高效微调 (LoRA等, 见下一节)
    """
    print("=" * 60)
    print("灾难性遗忘演示")
    print("=" * 60)

    # 任务A: 预训练任务
    backbone = SimpleBackbone()
    model_A = nn.Sequential(backbone, nn.Linear(32, 10))
    X_A, y_A = generate_data(500, n_classes=10)

    opt = torch.optim.Adam(model_A.parameters(), lr=1e-3)
    for _ in range(150):
        loss = F.cross_entropy(model_A(X_A), y_A)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 记录任务A的性能
    with torch.no_grad():
        acc_A_before = (model_A(X_A).argmax(dim=1) == y_A).float().mean().item()
    print(f"微调前任务A准确率: {acc_A_before:.2%}")

    # 任务B: 在同一个backbone上微调 (全量微调，较大学习率)
    model_B = nn.Sequential(backbone, nn.Linear(32, 3))
    X_B, y_B = generate_data(200, n_classes=3)

    # 用较大学习率微调 → 加剧灾难性遗忘
    opt_B = torch.optim.Adam(model_B.parameters(), lr=1e-2)
    for _ in range(100):
        loss_B = F.cross_entropy(model_B(X_B), y_B)
        opt_B.zero_grad()
        loss_B.backward()
        opt_B.step()

    # 再次评估任务A (backbone已被改变)
    with torch.no_grad():
        acc_A_after = (model_A(X_A).argmax(dim=1) == y_A).float().mean().item()
    print(f"微调后任务A准确率: {acc_A_after:.2%}")
    print(f"准确率下降: {acc_A_before - acc_A_after:.2%}")
    print("→ 这就是灾难性遗忘: 在任务B上微调后，任务A的性能显著下降")
    print()


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    demo_feature_extraction_vs_finetuning()
    demo_lr_impact()
    demo_catastrophic_forgetting()

    # ===========================================================
    # 思考题
    # ===========================================================
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    print("""
1. 为什么预训练有效？从"特征层次"的角度解释:
   低层网络学到了什么？高层网络学到了什么？

2. 特征提取和全量微调分别在什么场景下更合适？
   提示: 考虑下游数据量、下游任务与预训练任务的相似度。

3. 分层学习率 (layerwise learning rate decay) 的直觉是什么？
   为什么低层应该用更小的学习率？

4. 如何缓解灾难性遗忘？列举至少3种方法并说明原理。

5. 在NLP中，BERT微调时通常推荐学习率2e-5到5e-5。
   为什么不能用预训练时的学习率(如1e-4)直接微调？
""")
