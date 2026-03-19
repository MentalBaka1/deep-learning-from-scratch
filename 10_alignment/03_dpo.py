"""
第10章·第3节·DPO直接偏好优化 (Direct Preference Optimization)
===============================================================
核心: RLHF的问题, DPO绕过奖励模型直接优化, DPO损失函数, DPO vs RLHF对比

关键概念:
- RLHF 问题: 训练复杂(4个模型), 不稳定, 超参数多
- DPO 洞察: 最优策略与奖励函数存在闭式映射
  r(x,y) = β·log[π(y|x)/π_ref(y|x)] + const
- DPO 损失: L = -E[log σ(β·(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
- 无需奖励模型和 RL, 直接用偏好数据优化策略

运行: python 03_dpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy

# ============================================================
# 第一部分: RLHF 的问题与 DPO 的动机
# ============================================================

def print_rlhf_problems():
    print("=" * 60)
    print("RLHF 的问题 → DPO 的动机")
    print("=" * 60)
    print("""
    RLHF 主要问题:
    1. 训练复杂: 需 4 个模型 (策略/参考/奖励/价值), GPU 内存巨大
    2. 训练不稳定: RL 本身不稳定, 奖励模型误差被放大
    3. 超参数敏感: β, ε, 学习率等需大量调参
    4. Reward hacking: 策略可能利用奖励模型缺陷

    DPO 核心洞察:
    ┌───────────────────────────────────────────────────────┐
    │ 最优策略与奖励函数的闭式关系:                          │
    │   π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)       │
    │ 反解:                                                  │
    │   r(x,y) = β · log[π*(y|x) / π_ref(y|x)] + const     │
    │ 代入 Bradley-Terry 损失 → 消除奖励模型 r!              │
    │ → 直接用偏好数据优化策略模型                            │
    └───────────────────────────────────────────────────────┘
    """)


# ============================================================
# 第二部分: DPO 损失函数实现
# ============================================================

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO 损失函数

    参数:
        policy_chosen/rejected_logps:  策略模型的 log 概率, (batch,)
        ref_chosen/rejected_logps:     参考模型的 log 概率, (batch,)
        beta: 温度参数 (控制偏离程度)

    返回: (loss, metrics_dict)

    公式: L = -E[log σ(β · (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))]
    """
    # 隐式奖励差
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)

    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'reward_margin': margin.item(),
    }
    return loss, metrics


# ============================================================
# 第三部分: 简单 DPO 训练 Demo
# ============================================================

class SimpleLM(nn.Module):
    """简化语言模型: Embedding → TransformerEncoder → LM Head"""
    def __init__(self, vocab_size=100, d_model=64, nhead=4,
                 num_layers=2, max_len=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()
        x = self.transformer(x, mask=mask)
        return self.lm_head(x)

    def get_sequence_logprob(self, input_ids):
        """
        计算序列 log 概率: P(y1,..,yT|x) = Σ log P(yt|y<t, x)
        """
        logits = self.forward(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        return token_logps.sum(dim=-1)  # (batch,)


class DPOPreferenceDataset(Dataset):
    """偏好数据集: chosen (大token) vs rejected (小token)"""
    def __init__(self, num_samples=500, vocab_size=100, seq_len=16):
        self.data = []
        for _ in range(num_samples):
            chosen = torch.randint(vocab_size // 2, vocab_size, (seq_len,))
            rejected = torch.randint(0, vocab_size // 2, (seq_len,))
            self.data.append({'chosen': chosen, 'rejected': rejected})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_dpo():
    """
    DPO 训练: 无需奖励模型, 无需 RL, 只需策略+参考模型
    直接在偏好数据上做监督学习
    """
    print("=" * 60)
    print("DPO 训练演示")
    print("=" * 60)

    vocab_size, seq_len, batch_size = 100, 16, 32
    num_epochs, beta, lr = 10, 0.1, 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    policy = SimpleLM(vocab_size=vocab_size, d_model=64).to(device)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    dataset = DPOPreferenceDataset(num_samples=500, vocab_size=vocab_size, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"参数量: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"DPO β = {beta}\n")

    for epoch in range(num_epochs):
        total_metrics = {'loss': 0, 'accuracy': 0, 'reward_margin': 0}
        n = 0
        for batch in dataloader:
            chosen = batch['chosen'].to(device)
            rejected = batch['rejected'].to(device)

            pi_chosen = policy.get_sequence_logprob(chosen)
            pi_rejected = policy.get_sequence_logprob(rejected)
            with torch.no_grad():
                ref_chosen = ref_policy.get_sequence_logprob(chosen)
                ref_rejected = ref_policy.get_sequence_logprob(rejected)

            loss, metrics = dpo_loss(pi_chosen, pi_rejected,
                                     ref_chosen, ref_rejected, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in total_metrics:
                total_metrics[k] += metrics[k]
            n += 1

        avg = {k: v / n for k, v in total_metrics.items()}
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>2}/{num_epochs} | "
                  f"损失: {avg['loss']:.4f} | "
                  f"准确率: {avg['accuracy']:.4f} | "
                  f"奖励差距: {avg['reward_margin']:.4f}")

    print("\nDPO 训练完成! 无需奖励模型和 RL, 流程大大简化\n")


# ============================================================
# 第四部分: DPO vs RLHF 对比表
# ============================================================

def print_comparison():
    print("=" * 60)
    print("DPO vs RLHF 对比")
    print("=" * 60)
    print("""
    ┌──────────────┬──────────────────────┬──────────────────────┐
    │    维度       │       RLHF           │       DPO            │
    ├──────────────┼──────────────────────┼──────────────────────┤
    │ 核心思路     │ 先学奖励, 再 RL 优化  │ 直接从偏好数据优化    │
    │ 所需模型     │ 策略+参考+奖励+价值  │ 策略+参考 (2个)       │
    │ 训练复杂度   │ 高 (RL循环)          │ 低 (监督学习)         │
    │ 训练稳定性   │ 较差                 │ 较好                  │
    │ GPU 内存     │ 非常大               │ 较小                  │
    │ 超参数       │ 多 (β,ε,GAE等)      │ 少 (主要是β)          │
    │ 在线采样     │ 需要                 │ 不需要 (离线)         │
    │ 奖励可解释   │ 有显式分数           │ 隐式奖励              │
    │ 扩展性       │ 奖励模型可复用       │ 每次需重新训练        │
    │ 代表工作     │ InstructGPT/ChatGPT  │ Zephyr/NeuralChat     │
    └──────────────┴──────────────────────┴──────────────────────┘

    总结:
    - DPO 更简单, 适合资源有限场景
    - RLHF 更灵活, 适合迭代优化
    - 实际效果接近, DPO 因简单性获广泛采用
    - DPO 局限: 离线数据可能过时, 缺乏在线探索
    """)


# ============================================================
# 第五部分: 思考题
# ============================================================

def print_questions():
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    questions = [
        "1. DPO 的 β 控制偏离参考模型的程度。\n"
        "   β 太大/太小分别导致什么问题? 如何选择?",
        "2. DPO 是离线算法, 训练数据固定。\n"
        "   与 RLHF 在线采样相比有何优缺点?",
        "3. 理论上 DPO 与 RLHF 等价, 实际效果可能不同, 为什么?",
        "4. 如果参考模型 π_ref 质量很差, 会对 DPO 有何影响?\n"
        "   为何 DPO 通常要求 π_ref 是 SFT 后的模型?",
        "5. IPO 指出 DPO 可能过拟合偏好数据。\n"
        "   分析过拟合原因及可能的解决方法。",
    ]
    for q in questions:
        print(f"\n{q}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print_rlhf_problems()
    train_dpo()
    print_comparison()
    print_questions()
