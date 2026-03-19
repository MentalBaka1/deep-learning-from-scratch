"""
第10章·第4节·对齐技术全景
============================
核心: Constitutional AI, GRPO(DeepSeek), ORPO/SimPO, 安全对齐, 对齐税

关键概念:
- Constitutional AI (Anthropic): 用 AI 反馈代替人类反馈, 自我批评与修正
- GRPO (DeepSeek): 组相对策略优化, 用组内排名代替价值模型
- ORPO: 无需参考模型的偏好优化
- SimPO: 用序列平均 log 概率作为隐式奖励
- 对齐税: 对齐可能降低模型的基础能力

运行: python 04_alignment_overview.py
"""

import torch
import torch.nn.functional as F

# ============================================================
# 第一部分: 对齐方法时间线
# ============================================================

def print_timeline():
    print("=" * 60)
    print("对齐技术发展时间线")
    print("=" * 60)
    print("""
    2017  RLHF 概念提出 (Christiano et al.)
      │   从人类偏好中学习奖励函数
    2022  InstructGPT / ChatGPT (OpenAI)
      │   RLHF 大规模应用: SFT → RM → PPO
    2022  Constitutional AI (Anthropic)
      │   RLAIF: AI 反馈代替人类反馈, 自我批评→修正→训练
    2023  DPO (Stanford)
      │   去掉 RL, 直接从偏好数据优化
    2023  ORPO (韩国团队)
      │   去掉参考模型, SFT + 偏好优化一步完成
    2024  SimPO (UMD)
      │   长度归一化 log 概率作为隐式奖励
    2024  GRPO (DeepSeek)
      │   组相对策略优化, 去掉价值模型
      │   DeepSeek-R1 的核心训练方法
    2024+ KTO, SPPO, Self-Play, WARM ...
    """)


# ============================================================
# 第二部分: GRPO 简化 Demo (DeepSeek)
# ============================================================
# GRPO 核心:
#   1. 同一 prompt 采样 G 个回答
#   2. RM 打分 → 组内标准化得到优势 (代替价值模型!)
#   3. PPO clip 更新策略

def grpo_loss(log_probs_new, log_probs_old, rewards,
              clip_eps=0.2, beta_kl=0.01, ref_log_probs=None):
    """
    GRPO 损失函数 (简化版)
    参数:
        log_probs_new/old: 新/旧策略 log 概率, (group_size,)
        rewards: RM 分数, (group_size,)
        clip_eps: 裁剪范围
        beta_kl: KL 惩罚系数
        ref_log_probs: 参考模型 log 概率 (可选)
    核心: 用组内标准化代替价值模型估计优势
    """
    # 组内标准化 → 优势 (GRPO 核心创新)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # PPO clip 目标
    ratio = torch.exp(log_probs_new - log_probs_old)
    obj_unclipped = ratio * advantages
    obj_clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(obj_unclipped, obj_clipped).mean()

    # KL 惩罚
    kl_loss = 0.0
    if ref_log_probs is not None:
        kl_loss = beta_kl * (log_probs_new - ref_log_probs).mean()

    return policy_loss + kl_loss


def grpo_demo():
    print("=" * 60)
    print("GRPO 简化演示 (DeepSeek 风格)")
    print("=" * 60)
    print("""
    GRPO 流程:
    1. 给定 prompt, 采样 G 个回答
    2. RM 打分每个回答
    3. 组内标准化 → 优势 (无需价值模型!)
    4. PPO clip 更新策略
    """)

    group_size = 8
    log_probs_old = torch.randn(group_size) * 0.5
    rewards = torch.tensor([1.2, 0.8, -0.5, 2.1, 0.3, -1.0, 1.5, 0.1])

    print(f"组大小 G = {group_size}")
    print(f"奖励: {rewards.tolist()}\n")

    # 组内标准化
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    print("组内标准化后:")
    for i, (r, a) in enumerate(zip(rewards, advantages)):
        label = "好回答" if a > 0 else "差回答"
        print(f"  回答{i}: 奖励={r:>6.2f}, 优势={a:>6.3f}  {label}")

    log_probs_new = log_probs_old.clone().requires_grad_(True)
    loss = grpo_loss(log_probs_new, log_probs_old.detach(), rewards,
                     ref_log_probs=log_probs_old.detach())
    print(f"\nGRPO 损失: {loss.item():.4f}")
    print("\nGRPO vs PPO 对比:")
    print("  ┌────────┬──────────────────────┬──────────────────────┐")
    print("  │        │ PPO                  │ GRPO                 │")
    print("  ├────────┼──────────────────────┼──────────────────────┤")
    print("  │ 优势   │ 价值模型 (额外训练)  │ 组内标准化 (无需训练)│")
    print("  │ 内存   │ 需加载价值模型       │ 省去价值模型         │")
    print("  │ 采样   │ 单次采样             │ 批量采样 G 个        │")
    print("  │ 代表   │ InstructGPT          │ DeepSeek-R1          │")
    print("  └────────┴──────────────────────┴──────────────────────┘")
    print()


# ============================================================
# 第三部分: 方法对比表
# ============================================================

def print_method_comparison():
    print("=" * 60)
    print("对齐方法对比")
    print("=" * 60)
    print("""
    ┌───────────┬──────────┬────────┬────────┬────────────────┐
    │   方法     │ 需要RM?  │ 需要RL?│ 参考模型│ 核心优势       │
    ├───────────┼──────────┼────────┼────────┼────────────────┤
    │ RLHF/PPO  │ 是       │ 是     │ 是     │ 灵活, 在线探索  │
    │ DPO       │ 否       │ 否     │ 是     │ 简单稳定        │
    │ GRPO      │ 是(打分) │ 是     │ 是     │ 无需价值模型    │
    │ ORPO      │ 否       │ 否     │ 否     │ SFT+对齐一步   │
    │ SimPO     │ 否       │ 否     │ 否     │ 无参考模型     │
    │ KTO       │ 否       │ 否     │ 是     │ 只需好/坏标签  │
    │ CAI/RLAIF │ AI生成   │ 是     │ 是     │ 减少人类标注   │
    └───────────┴──────────┴────────┴────────┴────────────────┘
    """)


# ============================================================
# 第四部分: 安全对齐与对齐税
# ============================================================

def print_safety_alignment():
    print("=" * 60)
    print("安全对齐与对齐税")
    print("=" * 60)
    print("""
    一、安全对齐目标 (3H 原则)
      Helpful  (有用): 准确回答, 完成任务
      Harmless (无害): 不生成有害/歧视/暴力内容
      Honest   (诚实): 不编造事实, 承认不确定性

    二、Constitutional AI (Anthropic)
      核心: 一套"宪法"原则指导 AI 自我改进
      流程: 生成回答 → AI自我批评 → AI自我修正 → 用修正对训练
      优势: 减少人类标注依赖, 可扩展

    三、对齐税 (Alignment Tax)
      定义: 对齐训练可能降低基础能力
      表现:
      - 过度拒绝: 拒绝合理但敏感的问题
      - 能力退化: 数学/编码能力下降
      - 多样性降低: 回答趋于保守和模板化
      缓解:
      - 合理 KL 约束, 防止过度对齐
      - 高质量偏好数据, 减少噪声
      - 多目标优化: 平衡安全性和有用性
      - 红队测试: 持续发现和修复问题
    """)


# ============================================================
# 第五部分: 各方法优缺点总结
# ============================================================

def print_pros_cons():
    print("=" * 60)
    print("各方法优缺点总结")
    print("=" * 60)
    print("""
    RLHF (PPO)
      优: 在线探索, 奖励模型可复用, 大规模验证
      缺: 复杂不稳定, 内存大, reward hacking

    DPO
      优: 简单稳定, 内存友好, 无需 RL 知识
      缺: 离线数据可能过时, 缺乏探索, 可能过拟合

    GRPO (DeepSeek)
      优: 去掉价值模型省内存, 推理任务效果好
      缺: 需较大组采样数, 仍需奖励模型

    ORPO
      优: SFT 和对齐合一, 不需参考模型, 极简
      缺: 理论基础较弱, 效果不如 DPO 稳定

    SimPO
      优: 无参考模型, 长度归一化处理偏差
      缺: 隐式奖励可能不够准确

    Constitutional AI
      优: 减少人类标注, 可扩展, 原则化
      缺: AI 判断可能有偏差, 宪法设计困难
    """)


# ============================================================
# 第六部分: 思考题
# ============================================================

def print_questions():
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    questions = [
        "1. GRPO 用组内排名代替价值模型。\n"
        "   组大小 G 如何影响优势估计? G 太小会怎样?",
        "2. Constitutional AI 依赖 AI 自身判断改进回答。\n"
        "   若 AI 有系统性偏差, 会导致什么后果?",
        "3. 如何量化对齐税?\n"
        "   设计实验衡量对齐训练对基础能力的影响。",
        "4. DPO/ORPO/SimPO 都在简化 RLHF。\n"
        "   简化到什么程度会损失效果? 极限在哪?",
        "5. DeepSeek-R1 用 GRPO 训练推理能力, 出现 'aha moment'。\n"
        "   为何 GRPO 适合推理? 与 DPO 相比有何优势?",
    ]
    for q in questions:
        print(f"\n{q}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print_timeline()
    grpo_demo()
    print_method_comparison()
    print_safety_alignment()
    print_pros_cons()
    print_questions()
