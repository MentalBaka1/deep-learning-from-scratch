"""
第10章·第1节·奖励模型 (Reward Model)
======================================
核心: 人类偏好数据(prompt, chosen, rejected), Bradley-Terry模型 P(A>B)=σ(r(A)-r(B)), 奖励模型训练

关键概念:
- 人类偏好数据: 给定同一个 prompt, 人类标注哪个回答更好(chosen vs rejected)
- Bradley-Terry 模型: 用 sigmoid 将两个回答的奖励差映射为偏好概率
- 奖励模型: 一个输出标量分数的模型, 用于评估回答质量

运行: python 01_reward_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# ============================================================
# 第一部分: 偏好数据集构造 (Synthetic Preference Pairs)
# ============================================================
# 真实场景中, 偏好数据来自人类标注:
#   给定 prompt, 模型生成多个回答, 人类选出更好的那个
# 这里用合成数据模拟

class PreferenceDataset(Dataset):
    """
    合成偏好数据集
    每条数据: prompt(输入), chosen(偏好回答), rejected(非偏好回答)
    合成逻辑: chosen 的 token id 偏大 → 模拟"质量更好"
    """
    def __init__(self, num_samples=500, vocab_size=100, seq_len=16):
        super().__init__()
        self.data = []
        for _ in range(num_samples):
            prompt = torch.randint(0, vocab_size, (seq_len,))
            chosen = torch.randint(vocab_size // 2, vocab_size, (seq_len,))
            rejected = torch.randint(0, vocab_size // 2, (seq_len,))
            self.data.append({
                'prompt': prompt, 'chosen': chosen, 'rejected': rejected,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# 第二部分: Bradley-Terry 模型
# ============================================================
# 偏好学习的理论基础:
#   P(A 优于 B) = sigmoid(r(A) - r(B))
# r(·) 是奖励函数, 输出标量。若 r(A) >> r(B), sigmoid → 1
# 损失: L = -log(sigmoid(r(chosen) - r(rejected)))

def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry 偏好损失
    参数: reward_chosen/rejected shape: (batch,)
    返回: 标量损失值
    公式: L = -E[log σ(r_chosen - r_rejected)]
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def demonstrate_bradley_terry():
    """演示 Bradley-Terry 模型的行为"""
    print("=" * 60)
    print("Bradley-Terry 模型演示")
    print("=" * 60)
    print("公式: P(A > B) = σ(r(A) - r(B))\n")

    diffs = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    probs = torch.sigmoid(diffs)

    print(f"{'r(A)-r(B)':>10} | {'P(A>B)':>8} | 解释")
    print("-" * 45)
    for d, p in zip(diffs, probs):
        if d < 0:
            desc = "B 明显更好" if d < -2 else "B 略好"
        elif d > 0:
            desc = "A 明显更好" if d > 2 else "A 略好"
        else:
            desc = "难以区分"
        print(f"{d.item():>10.1f} | {p.item():>8.4f} | {desc}")

    # 不同奖励差距下的损失值
    print()
    print("不同奖励组合下的损失:")
    print(f"{'r_chosen':>10} | {'r_rejected':>10} | {'损失':>8} | 解释")
    print("-" * 55)
    cases = [
        (3.0, 0.0, "chosen 远好于 rejected"),
        (1.0, 0.0, "chosen 略好于 rejected"),
        (0.5, 0.5, "两者相当"),
        (-0.5, 1.0, "rejected 更好 (标注错误?)"),
    ]
    for r_c, r_r, desc in cases:
        rc = torch.tensor([r_c])
        rr = torch.tensor([r_r])
        l = bradley_terry_loss(rc, rr)
        print(f"{r_c:>10.1f} | {r_r:>10.1f} | {l.item():>8.4f} | {desc}")

    print("\n关键: 当 chosen 奖励高于 rejected 时损失小, 反之损失大\n")


# ============================================================
# 第三部分: 奖励模型 (Reward Model)
# ============================================================
# 在预训练 LM 基础上去掉 LM head, 加线性层输出标量奖励

class SimpleRewardModel(nn.Module):
    """
    简化版奖励模型
    结构: Embedding → Transformer Encoder → 最后token池化 → 标量奖励头
    真实系统中基座是预训练 LLM (如 LLaMA), 奖励头: Linear(hidden, 1)
    """
    def __init__(self, vocab_size=100, d_model=64, nhead=4,
                 num_layers=2, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 奖励头: 隐藏状态 → 标量
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1),
        )

    def forward(self, input_ids):
        """
        参数: input_ids (batch, seq_len)
        返回: reward (batch,) 标量奖励值
        """
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.transformer(x)                    # (batch, seq_len, d_model)
        last_hidden = x[:, -1, :]                  # 取最后 token
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


# ============================================================
# 第四部分: 训练循环
# ============================================================

def train_reward_model():
    """
    训练奖励模型: 准备偏好数据 → 计算奖励 → Bradley-Terry 损失 → 更新
    """
    print("=" * 60)
    print("奖励模型训练")
    print("=" * 60)

    vocab_size, seq_len, batch_size, num_epochs, lr = 100, 16, 32, 8, 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dataset = PreferenceDataset(num_samples=500, vocab_size=vocab_size, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleRewardModel(vocab_size=vocab_size, d_model=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"数据集大小: {len(dataset)}\n")

    for epoch in range(num_epochs):
        total_loss, total_acc, num_batches = 0.0, 0.0, 0
        for batch in dataloader:
            chosen_input = batch['chosen'].to(device)
            rejected_input = batch['rejected'].to(device)
            reward_chosen = model(chosen_input)
            reward_rejected = model(rejected_input)
            loss = bradley_terry_loss(reward_chosen, reward_rejected)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (reward_chosen > reward_rejected).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>2}/{num_epochs} | "
                  f"损失: {avg_loss:.4f} | 偏好准确率: {avg_acc:.4f}")

    # 评估: 检查模型是否学到了正确的偏好
    print("\n训练完成! 测试奖励模型的判断能力:")
    model.eval()
    with torch.no_grad():
        # 构造不同质量的样本
        good = torch.randint(70, 100, (4, seq_len)).to(device)   # 高质量
        mid = torch.randint(30, 70, (4, seq_len)).to(device)     # 中等质量
        bad = torch.randint(0, 30, (4, seq_len)).to(device)      # 低质量

        r_good = model(good)
        r_mid = model(mid)
        r_bad = model(bad)

        print(f"  高质量回答平均奖励: {r_good.mean().item():.4f}")
        print(f"  中等质量回答平均奖励: {r_mid.mean().item():.4f}")
        print(f"  低质量回答平均奖励: {r_bad.mean().item():.4f}")
        print(f"  高-低奖励差距: {(r_good.mean() - r_bad.mean()).item():.4f}")

        # 检查排序是否正确
        ranking_correct = (r_good.mean() > r_mid.mean() > r_bad.mean()).item()
        print(f"  排序正确 (高>中>低): {'是' if ranking_correct else '否'}")

    return model


# ============================================================
# 第五部分: 思考题
# ============================================================

def print_questions():
    """课后思考题"""
    print("\n" + "=" * 60)
    print("思考题")
    print("=" * 60)
    questions = [
        "1. Bradley-Terry 假设偏好传递 (A>B, B>C → A>C),\n"
        "   人类偏好未必满足。这会带来什么问题? 如何缓解?",
        "2. 奖励模型用最后 token 的隐藏状态预测奖励。\n"
        "   改用平均池化有何影响? 哪种更合理?",
        "3. 偏好数据存在标注噪声 (标注者不一致) 时,\n"
        "   如何改进训练来提高鲁棒性?",
        "4. 奖励分数只有相对排序意义, 没有绝对意义。\n"
        "   这对后续 RLHF 训练有什么影响?",
        "5. InstructGPT 中奖励模型通常比策略模型小, 为什么?\n"
        "   如果奖励模型更大, 有什么优缺点?",
    ]
    for q in questions:
        print(f"\n{q}")
    print()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    demonstrate_bradley_terry()
    model = train_reward_model()
    print_questions()
