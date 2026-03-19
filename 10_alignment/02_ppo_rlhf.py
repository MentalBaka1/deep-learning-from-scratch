"""
第10章·第2节·PPO与RLHF
========================
核心: 策略梯度基础, PPO(clip objective), RLHF四步流程(预训练→SFT→RM→PPO), KL散度约束

关键概念:
- REINFORCE: 最基础的策略梯度, ∇J = E[R · ∇log π(a|s)]
- PPO: 通过 clip 限制策略更新幅度, 保证训练稳定
- RLHF 流程: 预训练 → SFT → RM → PPO
- KL 散度约束: 防止策略偏离参考模型太远, 避免 reward hacking

运行: python 02_ppo_rlhf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy

# ============================================================
# 第一部分: REINFORCE 策略梯度基础
# ============================================================
# REINFORCE: 采样动作 → 计算回报 → 更新: ∇J ≈ R · ∇log π(a|s)
# 直觉: 高回报动作 → 增大其概率

class SimplePolicy(nn.Module):
    """简单策略网络: 给定状态, 输出动作概率分布"""
    def __init__(self, state_dim=8, hidden_dim=32, action_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state):
        """采样动作并返回 log 概率"""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_log_prob(self, state, action):
        """计算给定 (state, action) 对的 log 概率"""
        logits = self.forward(state)
        return Categorical(logits=logits).log_prob(action)


def reinforce_demo():
    """
    REINFORCE 演示
    任务: 学习选择动作 7 (奖励最高)
    """
    print("=" * 60)
    print("REINFORCE 策略梯度演示")
    print("=" * 60)
    print("目标: 学习选择动作 7 (奖励最高)\n")

    policy = SimplePolicy(state_dim=8, action_dim=10)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    state = torch.randn(8)  # 固定状态

    for episode in range(200):
        action, log_prob = policy.get_action(state)
        reward = -abs(action.item() - 7)  # 最优动作 = 7
        loss = -reward * log_prob          # REINFORCE: -R · log π(a|s)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode + 1) % 50 == 0:
            with torch.no_grad():
                probs = F.softmax(policy(state), dim=-1)
                best = probs.argmax().item()
                print(f"Episode {episode+1:>3} | 最可能动作: {best} | "
                      f"概率: {probs[best]:.4f} | 奖励: {reward}")
    print()


# ============================================================
# 第二部分: PPO Clipped Objective
# ============================================================
# 核心: 限制每次更新幅度
# 比率 r(θ) = π_new(a|s) / π_old(a|s)
# PPO: L = min(r·A, clip(r, 1-ε, 1+ε)·A)
# A>0 (好动作): r 不超过 1+ε → 不过度增大概率
# A<0 (坏动作): r 不低于 1-ε → 不过度减小概率

def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, clip_eps=0.2):
    """
    PPO 裁剪目标
    参数: log_probs_new/old (batch,), advantages (batch,), clip_eps ε
    返回: PPO 损失 (取负, 梯度下降最大化目标)
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    obj_unclipped = ratio * advantages
    obj_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(obj_unclipped, obj_clipped).mean()


def demonstrate_ppo_clipping():
    """演示 PPO 裁剪机制"""
    print("=" * 60)
    print("PPO 裁剪机制演示")
    print("=" * 60)

    clip_eps = 0.2
    ratios = torch.linspace(0.5, 1.5, 11)

    for adv_val, label in [(1.0, "A > 0 (好动作)"), (-1.0, "A < 0 (坏动作)")]:
        print(f"\n优势 {label}:")
        print(f"{'ratio':>8} | {'无裁剪':>8} | {'裁剪后':>8} | {'PPO':>8}")
        print("-" * 45)
        for r in ratios:
            unc = r.item() * adv_val
            clp = torch.clamp(r, 1 - clip_eps, 1 + clip_eps).item() * adv_val
            print(f"{r.item():>8.2f} | {unc:>8.3f} | {clp:>8.3f} | {min(unc,clp):>8.3f}")

    print("\n要点: PPO 裁剪阻止策略变化过大, 保证训练稳定\n")


# ============================================================
# 第三部分: RLHF 完整流程图
# ============================================================

def print_rlhf_pipeline():
    """展示 RLHF 的四步流程"""
    print("=" * 60)
    print("RLHF 完整流程")
    print("=" * 60)
    print("""
    ┌──────────────────────────────────────────────────────┐
    │  第1步: 预训练 (Pretraining)                         │
    │  大规模文本 → 下一 token 预测 → 基座模型              │
    ├──────────────────────────────────────────────────────┤
    │  第2步: 监督微调 SFT                                  │
    │  高质量 (指令, 回答) 对 → 学习遵循指令                 │
    │  结果: SFT 模型 (也作为 PPO 的参考模型 π_ref)         │
    ├──────────────────────────────────────────────────────┤
    │  第3步: 奖励模型训练 RM                               │
    │  人类偏好 (chosen vs rejected) → Bradley-Terry 训练    │
    │  结果: RM(x, y) → 标量分数                            │
    ├──────────────────────────────────────────────────────┤
    │  第4步: PPO 强化学习微调                              │
    │  max E[RM(x,y)] - β·KL(π||π_ref)                    │
    │  KL 惩罚防止 reward hacking → RLHF 对齐模型          │
    └──────────────────────────────────────────────────────┘
    """)


# ============================================================
# 第四部分: KL 散度约束
# ============================================================
# KL(π || π_ref) = E_π[log(π/π_ref)]
# 作用: 1) 防止 reward hacking  2) 保持语言能力

def compute_kl_divergence(logits_policy, logits_reference):
    """
    计算策略与参考模型间的 KL 散度
    参数: logits_policy, logits_reference shape: (batch, vocab_size)
    返回: 标量 KL 散度
    """
    log_p = F.log_softmax(logits_policy, dim=-1)
    p = F.softmax(logits_policy, dim=-1)
    log_q = F.log_softmax(logits_reference, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def demonstrate_kl_penalty():
    """演示 KL 惩罚在 RLHF 中的作用"""
    print("=" * 60)
    print("KL 散度约束演示")
    print("=" * 60)

    vocab_size = 10
    logits_ref = torch.zeros(4, vocab_size)  # 参考: 均匀分布

    print("策略偏离参考模型的不同程度:")
    print(f"{'温度':>6} | {'KL散度':>8} | 解释")
    print("-" * 45)
    for temp in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        logits_p = torch.randn(4, vocab_size) / temp
        kl = compute_kl_divergence(logits_p, logits_ref)
        desc = ("接近参考" if kl < 0.5 else
                "适度偏离" if kl < 2.0 else
                "严重偏离 (reward hacking 风险)")
        print(f"{temp:>6.2f} | {kl.item():>8.4f} | {desc}")

    print("\nRLHF 总奖励: R_total = R_model(x,y) - β · KL(π||π_ref)")
    print("\n不同 β 值的效果 (假设 R_model=5.0, KL=2.0):")
    print(f"{'β':>6} | {'R_total':>10} | 效果")
    print("-" * 45)
    for beta in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
        r = 5.0 - beta * 2.0
        eff = ("几乎不约束" if beta < 0.05 else
               "适度约束" if beta < 0.5 else "强约束, 模型改变小")
        print(f"{beta:>6.2f} | {r:>10.4f} | {eff}")
    print()


# ============================================================
# 第五部分: 简化版 PPO-RLHF 训练循环
# ============================================================

def simplified_ppo_rlhf():
    """模拟 RLHF 第4步: 用 PPO 优化策略模型"""
    print("=" * 60)
    print("简化版 PPO-RLHF 训练")
    print("=" * 60)

    state_dim, action_dim = 8, 10
    beta_kl, clip_eps = 0.1, 0.2
    batch_size = 16

    # 策略模型 (从 SFT 初始化)
    policy = SimplePolicy(state_dim, 32, action_dim)
    # 参考模型 (冻结的 SFT)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # 模拟奖励模型: 偏好动作 7
    def reward_model(actions):
        return -torch.abs(actions.float() - 7.0)

    print(f"β={beta_kl}, ε={clip_eps}\n")

    for step in range(100):
        states = torch.randn(batch_size, state_dim)

        # 采样阶段
        with torch.no_grad():
            old_logits = policy(states)
            dist_old = Categorical(logits=old_logits)
            actions = dist_old.sample()
            log_probs_old = dist_old.log_prob(actions)
            ref_logits = ref_policy(states)

        rm_reward = reward_model(actions)
        with torch.no_grad():
            kl = compute_kl_divergence(old_logits, ref_logits)
            total_reward = rm_reward - beta_kl * kl
        advantages = total_reward - total_reward.mean()

        # PPO 多轮更新
        for _ in range(4):
            log_probs_new = policy.get_log_prob(states, actions)
            loss = ppo_clipped_loss(log_probs_new, log_probs_old, advantages, clip_eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (step + 1) % 20 == 0:
            with torch.no_grad():
                probs = F.softmax(policy(states[0:1]), dim=-1)
                best = probs.argmax(dim=-1).item()
                print(f"Step {step+1:>3} | 平均奖励: {rm_reward.mean():.3f} | "
                      f"KL: {kl:.4f} | 最可能动作: {best}")

    print("\nPPO-RLHF: 奖励模型信号 + KL 约束 → 安全对齐\n")


# ============================================================
# 第六部分: 思考题
# ============================================================

def print_questions():
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    questions = [
        "1. PPO 裁剪范围 ε 如何影响训练?\n"
        "   ε 太大或太小分别导致什么问题?",
        "2. KL 惩罚系数 β 通常需要动态调整。\n"
        "   KL 散度持续增大时, 应增大还是减小 β?",
        "3. RLHF 四步中, SFT 和 RM 能否互换顺序?\n"
        "   先训练 RM 再 SFT 会有什么问题?",
        "4. 除 KL 惩罚外, 还有哪些方法缓解 reward hacking?",
        "5. PPO 需同时维护策略/参考/奖励/价值四个模型,\n"
        "   内存开销巨大。有哪些工程优化方法?",
    ]
    for q in questions:
        print(f"\n{q}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    reinforce_demo()
    demonstrate_ppo_clipping()
    print_rlhf_pipeline()
    demonstrate_kl_penalty()
    simplified_ppo_rlhf()
    print_questions()
