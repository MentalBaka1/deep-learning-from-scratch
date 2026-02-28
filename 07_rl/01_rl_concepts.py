"""
==============================================================
第7章 第1节：强化学习核心概念 —— 从零开始学"试错"
==============================================================

【为什么需要它？】
监督学习的局限：需要大量(输入, 正确输出)标注对
  - 怎么教机器人走路？没法标注每个关节的"正确角度"
  - 怎么教 AI 下围棋？棋局中间根本不知道哪步是"正确的"
  - 怎么优化工厂生产线？中间过程太复杂，无法直接标注

强化学习（Reinforcement Learning, RL）：
  不需要"正确答案"！
  只需要给出"好不好"的反馈（奖励/惩罚）
  智能体（Agent）自己探索，从奖励信号中学习

【生活类比：训狗】
  狗（智能体）在世界（环境）中行动：
    - 坐下 → 给零食（奖励 +1）
    - 乱跑 → 不给零食（奖励 0）
    - 咬人 → 惩罚（奖励 -1）

  狗会慢慢学会：什么行为能得到更多零食
  这就是强化学习的本质！

  智能体观察状态（现在狗在哪里？主人在说什么？）
  选择动作（坐/跑/摇尾巴）
  获得奖励信号（零食或惩罚）
  更新策略（下次在这种情况下该怎么做）

【存在理由】
解决问题：当"正确答案"未知但可以评价"结果好坏"时的学习问题
核心思想：通过与环境的交互，最大化累积奖励，学习最优行为策略
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# ============================================================
# Part 1: 核心概念定义
# ============================================================
print("=" * 50)
print("Part 1: 强化学习核心概念")
print("=" * 50)

"""
强化学习的五大要素：

1. 智能体（Agent）：学习者和决策者
   - 就像"狗"，是做决策的主体

2. 环境（Environment）：智能体交互的世界
   - 就像"真实世界"，接受动作并返回新状态和奖励

3. 状态（State, s）：对当前情况的描述
   - 棋盘的当前布局、机器人的位置和速度...
   - 状态空间（所有可能状态的集合）

4. 动作（Action, a）：智能体可以执行的操作
   - 下一步棋、向左/右/上/下移动...
   - 动作空间（所有可能动作的集合）

5. 奖励（Reward, r）：对动作好坏的即时评价
   - 到达终点 +100，踩到陷阱 -10，每步 -1...
   - 奖励函数由设计者决定（这是 RL 最难的部分之一！）

6. 策略（Policy, π）：从状态到动作的映射
   - π(a|s) = 在状态 s 下选择动作 a 的概率
   - 我们的目标：学习最优策略 π*

7. 价值函数（Value Function）：
   - V(s) = 从状态 s 出发，遵循策略 π，期望累积奖励
   - Q(s,a) = 在状态 s 执行动作 a 后，期望累积奖励
   - Q 值 > V 值：多了一个"动作"维度

8. 折扣因子（Discount Factor, γ）：
   - 0 < γ < 1，控制对未来奖励的重视程度
   - γ 接近 1：重视长期奖励
   - γ 接近 0：只重视眼前奖励
   - 总回报 G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
"""

print("强化学习交互循环：")
print()
print("  ┌─────────────────────────────────────────────┐")
print("  │                                             │")
print("  │   状态 s_t                                  │")
print("  │         ↓                                   │")
print("  │   Agent（策略 π）→ 动作 a_t               │")
print("  │                          ↓                  │")
print("  │   Environment → 奖励 r_t + 新状态 s_{t+1}  │")
print("  │                    ↑                        │")
print("  └────────────────────┘────────────────────────┘")
print()

# 折扣回报计算演示
print("折扣回报（γ=0.9）计算演示：")
rewards = [10, -5, 20, 3, 15]
gamma = 0.9
G = 0
print(f"  奖励序列：{rewards}")
for t in range(len(rewards) - 1, -1, -1):
    G = rewards[t] + gamma * G
    print(f"  G_{t} = r_{t} + {gamma}*G_{t+1} = {rewards[t]} + {gamma}*{G/(1):.2f} "
          f"→ 实际 G_{t} = {G:.2f}")

print(f"\n  → 总期望回报 G_0 = {G:.2f}")
print(f"  → 折扣让智能体不只追求眼前奖励，也考虑长远收益！")

# ============================================================
# Part 2: 探索 vs 利用 —— 多臂赌博机
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 探索 vs 利用 —— 多臂赌博机")
print("=" * 50)

"""
最简单的 RL 问题：多臂赌博机（Multi-Armed Bandit）

类比：赌场里有 K 台老虎机（多臂赌博机），每台中奖概率不同。
  你有 N 次机会，如何最大化总奖励？

挑战：不知道哪台最好！
  - 利用（Exploitation）：选择当前认为最好的那台（赚最多钱）
  - 探索（Exploration）：尝试其他机器（可能发现更好的）

纯利用：万一最开始选错了，永远不知道有更好的选择
纯探索：不断尝试新的，永远不充分利用已知好机器

最优策略：ε-greedy
  以概率 ε 随机探索（选任意一台）
  以概率 1-ε 利用（选当前最优的那台）
"""

class MultiArmedBandit:
    """多臂赌博机环境"""
    def __init__(self, n_arms=10, seed=42):
        np.random.seed(seed)
        self.n_arms = n_arms
        # 每台机器的真实期望奖励（我们不知道，智能体需要发现）
        self.true_rewards = np.random.randn(n_arms)
        self.optimal_arm = self.true_rewards.argmax()

    def pull(self, arm):
        """拉动一台机器，返回随机奖励"""
        return self.true_rewards[arm] + np.random.randn()  # 加噪声

class EpsilonGreedyAgent:
    """ε-greedy 智能体"""
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.Q = np.zeros(n_arms)    # 对每台机器的期望奖励估计
        self.N = np.zeros(n_arms)    # 每台机器被尝试的次数

    def choose_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)  # 探索
        else:
            return self.Q.argmax()  # 利用（选择当前最优估计）

    def update(self, arm, reward):
        """在线更新 Q 值（增量均值）"""
        self.N[arm] += 1
        # 新估计 = 旧估计 + 步长 * (实际奖励 - 旧估计)
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

# 实验：不同 ε 值的对比
n_arms = 10
n_steps = 1000
n_runs = 200  # 重复实验次数（取平均）
epsilons = [0.0, 0.01, 0.1, 0.5]  # 0=纯利用，0.5=几乎随机

print(f"多臂赌博机实验：{n_arms}台机器，{n_steps}步，{n_runs}次平均")

results = {}
for eps in epsilons:
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        bandit = MultiArmedBandit(n_arms=n_arms, seed=run)
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=eps)

        for step in range(n_steps):
            action = agent.choose_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            all_rewards[run, step] = reward
            all_optimal[run, step] = (action == bandit.optimal_arm)

    results[eps] = {
        'avg_reward': all_rewards.mean(axis=0),
        'pct_optimal': all_optimal.mean(axis=0)
    }
    final_reward = results[eps]['avg_reward'][-100:].mean()
    final_optimal = results[eps]['pct_optimal'][-100:].mean()
    print(f"  ε={eps:.2f}: 最终平均奖励={final_reward:.3f}, 选最优机率={final_optimal:.1%}")

# ============================================================
# Part 3: 策略设计 —— 软化的 Softmax 策略
# ============================================================
print("\nPart 3: Softmax 动作选择（温度参数）")
print("=" * 50)

"""
ε-greedy 的问题：探索是完全随机的（任何动作等概率）
  应该更多探索"可能更好的"动作，而不是随机探索

Softmax 动作选择（Boltzmann 探索）：
  P(a) = exp(Q(a)/τ) / Σ_b exp(Q(b)/τ)

  τ（温度参数）：
    τ → ∞：均匀随机探索（什么也不知道）
    τ → 0：贪婪利用（选 Q 值最高的）
    中间值：根据 Q 值的大小按比例探索

  这就像：
    - 温度高 = 对所有美食一视同仁地尝试
    - 温度低 = 只吃最喜欢的食物
    - 中间 = 最喜欢的食物吃得多，其他的也偶尔尝试
"""

def softmax_action(Q, temperature=1.0):
    """根据 Q 值和温度参数选择动作"""
    if temperature <= 0:
        return Q.argmax()
    e = np.exp((Q - Q.max()) / temperature)  # 数值稳定
    probs = e / e.sum()
    return np.random.choice(len(Q), p=probs)

# 演示温度的效果
Q_demo = np.array([1.0, 2.0, 3.0, 1.5, 0.5])
print("Q 值:", Q_demo)
print("不同温度下的动作概率：")
for temp in [0.1, 0.5, 1.0, 5.0, 100.0]:
    e = np.exp((Q_demo - Q_demo.max()) / temp)
    probs = e / e.sum()
    print(f"  τ={temp:5.1f}: {[f'{p:.3f}' for p in probs]}")

# ============================================================
# Part 4: 可视化
# ============================================================
print("\n可视化中...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('强化学习核心概念', fontsize=14)

# 1. 多臂赌博机的奖励曲线
ax = axes[0][0]
for eps in epsilons:
    smoothed = np.convolve(results[eps]['avg_reward'], np.ones(20)/20, mode='valid')
    ax.plot(smoothed, linewidth=2, label=f'ε={eps}')
ax.set_xlabel('步数')
ax.set_ylabel('平均奖励')
ax.set_title('多臂赌博机：不同ε的平均奖励\n（移动平均平滑）')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 选择最优臂的概率
ax = axes[0][1]
for eps in epsilons:
    smoothed = np.convolve(results[eps]['pct_optimal'], np.ones(20)/20, mode='valid')
    ax.plot(smoothed, linewidth=2, label=f'ε={eps}')
ax.set_xlabel('步数')
ax.set_ylabel('选择最优臂的概率')
ax.set_title('多臂赌博机：选择最优动作的频率\nε=0无法找到最优！')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 探索 vs 利用的权衡（概念图）
ax = axes[0][2]
ax.axis('off')
tradeoff_text = """
探索 vs 利用的经典权衡：

纯探索（ε=1）：
  + 能发现所有可能性
  - 永远不充分利用
  适合：初期探索阶段

纯利用（ε=0）：
  + 充分利用当前知识
  - 可能陷入局部最优
  适合：知识充分后

ε-greedy（0<ε<1）：
  平衡探索与利用
  ε 通常随时间衰减：
  刚开始多探索（学习）
  后来多利用（收获）

实际应用中的策略：
  ε从1.0衰减到0.01
  UCB（置信上界）
  Thompson Sampling
  Softmax（温度参数）
"""
ax.text(0.05, 0.95, tradeoff_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('探索 vs 利用')

# 4. 折扣因子对价值估计的影响
ax = axes[1][0]
rewards_seq = [1] * 20  # 每步奖励为1的序列
gammas = [0.5, 0.9, 0.99, 1.0]
colors = ['red', 'blue', 'green', 'purple']
for gamma, color in zip(gammas, colors):
    G = 0
    G_values = []
    for r in reversed(rewards_seq):
        G = r + gamma * G
        G_values.insert(0, G)
    ax.plot(G_values, color=color, linewidth=2, label=f'γ={gamma}')
ax.set_xlabel('时间步（从末尾往前）')
ax.set_ylabel('期望总回报 G_t')
ax.set_title('折扣因子γ的影响\n（每步奖励=1的序列）')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Softmax 温度参数的效果
ax = axes[1][1]
Q_vis = np.array([0.5, 1.0, 2.0, 1.5, 0.3])
temps = [0.1, 0.5, 1.0, 5.0]
x_pos = np.arange(len(Q_vis))
width = 0.2

for i, temp in enumerate(temps):
    e = np.exp((Q_vis - Q_vis.max()) / temp)
    probs = e / e.sum()
    ax.bar(x_pos + i*width - 0.3, probs, width, label=f'τ={temp}', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels([f'a{i+1}\n(Q={q})' for i, q in enumerate(Q_vis)], fontsize=8)
ax.set_ylabel('选择概率')
ax.set_title('Softmax 温度参数\n温度低→更贪婪，温度高→更随机')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 6. 奖励信号设计示例
ax = axes[1][2]
ax.axis('off')
reward_text = """
奖励信号设计的挑战（奖励工程）：

【迷宫导航】
  到达终点：+100
  撞墙：-1
  每步：-0.1（鼓励走短路径）

【棋类游戏】
  赢棋：+1
  输棋：-1
  中间：0（稀疏奖励！）
  → 需要 1000 步才得到反馈
  → MCTS（蒙特卡洛树搜索）辅助

【自动驾驶】
  安全行驶：+0.01/步
  超速：-1
  违规：-10
  碰撞：-100

奖励稀疏性问题：
  很多任务只有最终结果才有奖励
  → 智能体很难学习
  → 解决：奖励塑形（Reward Shaping）
          课程学习（Curriculum Learning）
          内在奖励（Intrinsic Motivation）
"""
ax.text(0.05, 0.95, reward_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
ax.set_title('奖励设计')

plt.tight_layout()
plt.savefig('07_rl/rl_concepts.png', dpi=80, bbox_inches='tight')
print("图片已保存：07_rl/rl_concepts.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【ε 的衰减策略】
   在实践中，ε 不是固定的，而是随训练步数衰减：
     ε(t) = max(ε_min, ε_start * decay^t)

   实现一个衰减 ε-greedy 策略，初始 ε=1.0，衰减到 0.01。
   对比固定 ε=0.1 的策略，哪个最终表现更好？
   直觉：开始多探索（不知道什么好），后来多利用（已经学到了）

2. 【UCB（置信上界）策略】
   UCB 的思路：不是随机探索，而是选择"潜力最大"的动作：
     a* = argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]

   第一项：已知的好（利用）
   第二项：不确定性（探索少的动作不确定性高）

   实现 UCB 策略并和 ε-greedy 对比。
   UCB 有个好的理论保证，为什么？

3. 【奖励塑形的危险】
   如果奖励设计不当，智能体可能找到"作弊"的方法：
   - 清洁机器人奖励"看不见垃圾" → 机器人把摄像头盖住了！
   - 赛车奖励收集道具 → 一直围着道具兜圈，不走完赛道！

   这叫"奖励黑客"（Reward Hacking）。
   设计一个你自己的任务，想想智能体可能会如何"作弊"？
""")
