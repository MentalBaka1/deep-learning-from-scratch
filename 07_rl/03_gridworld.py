"""
==============================================================
第7章 第3节：格世界实战 —— 策略可视化
==============================================================

【为什么需要它？】
前两节学了 RL 概念和 Q-Learning 算法。
这一节把所有东西结合起来，在一个更复杂的格世界上：
  1. 展示策略从"随机混乱"到"有序最优"的演变过程
  2. 可视化 Q 表的热力图
  3. 对比不同超参数（γ, α, ε）的效果
  4. 实现动画展示学习过程

这是强化学习"百闻不如一见"的精华：
  亲眼看到智能体从完全随机到找到最优路径的过程，
  深刻理解 Q-Learning 是如何"工作"的！

【本节的格世界更复杂】
  10×10 的格子
  多个障碍物、陷阱
  需要规划较长的路径
  展示策略学习的完整过程

【存在理由】
将前面所有 RL 概念具体化、可视化，
通过实验理解超参数的影响，为学习 DQN 等深度 RL 打好基础
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

np.random.seed(42)

# ============================================================
# Part 1: 更复杂的格世界
# ============================================================
print("=" * 50)
print("Part 1: 复杂格世界设计")
print("=" * 50)

EMPTY = 0
WALL = 1
GOAL = 2
TRAP = 3
START = 4

class ComplexGridWorld:
    """
    10×10 复杂格世界
    包含障碍物通道、陷阱区域和奖励终点
    """
    def __init__(self, size=8):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4

        # 动作：上/右/下/左
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['↑', '→', '↓', '←']
        self.action_arrows = [(-0.3, 0), (0, 0.3), (0.3, 0), (0, -0.3)]

        # 定义地图
        self.grid = np.full((size, size), EMPTY)

        # 起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.grid[self.start_pos] = START
        self.grid[self.goal_pos] = GOAL

        # 添加墙（L形障碍）
        self.walls = set()
        # 竖墙
        for r in range(0, size-1):
            self.walls.add((r, size//2))
        # 留一个通道
        passage_row = size // 3
        self.walls.discard((passage_row, size//2))

        # 横墙
        for c in range(1, size-1):
            self.walls.add((size//2, c))
        # 留一个通道
        passage_col = 2 * size // 3
        self.walls.discard((size//2, passage_col))

        for pos in self.walls:
            self.grid[pos] = WALL

        # 添加陷阱
        self.traps = set()
        trap_positions = [
            (size//3 + 1, size//2 - 1),
            (size//2 + 1, size//2 + 1),
            (size - 2, 1),
        ]
        for pos in trap_positions:
            if pos not in self.walls and pos != self.start_pos and pos != self.goal_pos:
                self.traps.add(pos)
                self.grid[pos] = TRAP

        self.reset()

    def reset(self):
        self.pos = self.start_pos
        return self.pos_to_idx(self.pos)

    def pos_to_idx(self, pos):
        return pos[0] * self.size + pos[1]

    def idx_to_pos(self, idx):
        return (idx // self.size, idx % self.size)

    def step(self, action):
        r, c = self.pos
        dr, dc = self.action_deltas[action]
        nr, nc = r + dr, c + dc

        # 边界和墙检查
        if (nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or
                (nr, nc) in self.walls):
            nr, nc = r, c

        self.pos = (nr, nc)
        state_idx = self.pos_to_idx(self.pos)

        if self.pos == self.goal_pos:
            return state_idx, 10.0, True
        elif self.pos in self.traps:
            return state_idx, -5.0, True
        else:
            return state_idx, -0.1, False

    def is_valid(self, pos):
        r, c = pos
        return (0 <= r < self.size and 0 <= c < self.size and
                pos not in self.walls)

# 创建环境
env = ComplexGridWorld(size=8)
print(f"格世界大小：{env.size}×{env.size}={env.size**2} 个状态")
print(f"起点：{env.start_pos}, 终点：{env.goal_pos}")
print(f"墙数量：{len(env.walls)}, 陷阱数量：{len(env.traps)}")
print(f"动作空间：{env.n_actions}（上/右/下/左）")

# ============================================================
# Part 2: 训练 Q-Learning（记录中间过程）
# ============================================================
print("\nPart 2: 训练 Q-Learning（记录学习过程）")
print("=" * 50)

class QLearningAgentWithHistory:
    """记录训练历史的 Q-Learning 智能体"""
    def __init__(self, n_states, n_actions, lr=0.15, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

        # 历史记录（保存不同训练阶段的快照）
        self.Q_history = {}
        self.reward_history = []
        self.success_rate_history = []

    def choose_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.Q[state_idx].argmax()

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.lr * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return self.Q.argmax(axis=1)

    def evaluate(self, env, n_eval=50):
        """评估当前策略的胜率"""
        successes = 0
        for _ in range(n_eval):
            s = env.reset()
            for _ in range(200):
                a = self.Q[s].argmax()  # 纯贪婪
                s, r, done = env.step(a)
                if done:
                    if r > 0:
                        successes += 1
                    break
        return successes / n_eval

# 训练
agent = QLearningAgentWithHistory(
    n_states=env.n_states,
    n_actions=env.n_actions,
    lr=0.15,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.998
)

n_episodes = 2000
snapshot_episodes = [50, 200, 500, 1000, 2000]  # 记录这些轮次的策略

episode_rewards = []
success_rates = []

print(f"训练 {n_episodes} 个 episode...")
for episode in range(1, n_episodes + 1):
    s = env.reset()
    total_reward = 0

    for step in range(300):
        a = agent.choose_action(s)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next
        total_reward += r
        if done:
            break

    agent.decay_epsilon()
    episode_rewards.append(total_reward)

    # 记录快照
    if episode in snapshot_episodes:
        agent.Q_history[episode] = agent.Q.copy()
        success_rate = agent.evaluate(env)
        success_rates.append((episode, success_rate))
        print(f"  Episode {episode:4d}: ε={agent.epsilon:.3f}, "
              f"胜率={success_rate:.1%}, 平均奖励={np.mean(episode_rewards[-50:]):.2f}")

# ============================================================
# Part 3: 可视化学习过程演变
# ============================================================
print("\nPart 3: 可视化策略演变")
print("=" * 50)

def draw_grid(ax, env, Q, title="", show_values=False):
    """画格世界和当前策略"""
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_aspect('equal')

    # 颜色映射
    color_map = {
        EMPTY: 'white', WALL: '#555555', GOAL: '#90EE90',
        TRAP: '#FF9999', START: '#FFFACD'
    }

    for r in range(env.size):
        for c in range(env.size):
            cell_type = env.grid[r, c]
            color = color_map.get(cell_type, 'white')
            rect = patches.Rectangle([c-0.5, r-0.5], 1, 1,
                                      facecolor=color, edgecolor='black',
                                      linewidth=0.5)
            ax.add_patch(rect)

    # 标注起点/终点/陷阱
    ax.text(env.start_pos[1], env.start_pos[0], 'S', ha='center', va='center',
            fontsize=10, fontweight='bold', color='darkblue')
    ax.text(env.goal_pos[1], env.goal_pos[0], 'G', ha='center', va='center',
            fontsize=10, fontweight='bold', color='darkgreen')
    for trap in env.traps:
        ax.text(trap[1], trap[0], 'T', ha='center', va='center',
                fontsize=10, fontweight='bold', color='darkred')

    # 画策略箭头
    policy = Q.argmax(axis=1)
    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            if pos in env.walls or pos == env.goal_pos:
                continue
            state_idx = env.pos_to_idx(pos)
            best_a = policy[state_idx]
            dr, dc = env.action_arrows[best_a]

            q_max = Q[state_idx].max()
            if q_max != 0:
                alpha_val = min(1.0, abs(q_max) / 5)  # Q 值越大越不透明
            else:
                alpha_val = 0.2

            ax.annotate('', xy=(c + dc, r + dr),
                       xytext=(c, r),
                       arrowprops=dict(arrowstyle='->', color='darkblue',
                                       lw=1.2, alpha=alpha_val))

    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.tick_params(labelsize=7)

def draw_q_heatmap(ax, env, Q, title=""):
    """画 Q 值热力图"""
    q_map = np.full((env.size, env.size), np.nan)
    for r in range(env.size):
        for c in range(env.size):
            pos = (r, c)
            if pos not in env.walls:
                state_idx = env.pos_to_idx(pos)
                q_map[r, c] = Q[state_idx].max()

    masked = np.ma.masked_invalid(q_map)
    cmap = plt.cm.RdYlGn
    cmap.set_bad('gray')

    vmin = np.nanmin(q_map)
    vmax = np.nanmax(q_map)
    im = ax.imshow(masked, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax)

    # 标注
    ax.text(env.start_pos[1], env.start_pos[0], 'S', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(env.goal_pos[1], env.goal_pos[0], 'G', ha='center', va='center',
            fontsize=10, fontweight='bold')
    for trap in env.traps:
        ax.text(trap[1], trap[0], 'T', ha='center', va='center',
                fontsize=8, fontweight='bold')

    ax.set_title(title, fontsize=9)
    return im

# 主图：策略演变（4个快照）
fig = plt.figure(figsize=(20, 16))
fig.suptitle('格世界 Q-Learning：从随机到最优的策略演变', fontsize=14)

# 上半部分：4个训练阶段的策略
snapshot_to_show = [50, 200, 500, 2000]
for i, ep in enumerate(snapshot_to_show):
    ax = fig.add_subplot(3, 4, i + 1)
    if ep in agent.Q_history:
        draw_grid(ax, env, agent.Q_history[ep],
                  title=f'Episode {ep}\n（早期：随机探索）' if ep == 50
                  else f'Episode {ep}')
    else:
        ax.text(0.5, 0.5, f'Episode {ep}', ha='center', va='center')

# 中间：最终 Q 值热力图
ax5 = fig.add_subplot(3, 4, 5)
draw_grid(ax5, env, agent.Q, title='最终策略（贪婪箭头）')

ax6 = fig.add_subplot(3, 4, 6)
im = draw_q_heatmap(ax6, env, agent.Q, title='最终 Q 值热力图\n（绿=高价值，红=低价值）')
plt.colorbar(im, ax=ax6, fraction=0.046)

# 训练曲线
ax7 = fig.add_subplot(3, 4, 7)
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax7.plot(episode_rewards, alpha=0.2, color='blue')
ax7.plot(range(window-1, len(episode_rewards)), smoothed, 'r-', linewidth=2)
for ep, sr in success_rates:
    ax7.axvline(x=ep, color='green', linestyle='--', alpha=0.5)
ax7.set_xlabel('Episode')
ax7.set_ylabel('总奖励')
ax7.set_title('训练奖励曲线')
ax7.grid(True, alpha=0.3)

# 胜率变化
ax8 = fig.add_subplot(3, 4, 8)
ep_list = [ep for ep, sr in success_rates]
sr_list = [sr for ep, sr in success_rates]
ax8.bar(range(len(ep_list)), sr_list, tick_label=[str(ep) for ep in ep_list],
        color='steelblue', alpha=0.7)
ax8.set_xlabel('训练 Episode')
ax8.set_ylabel('胜率（到达终点）')
ax8.set_title('胜率随训练的提升')
ax8.set_ylim(0, 1)
ax8.grid(True, alpha=0.3, axis='y')
for i, (ep, sr) in enumerate(success_rates):
    ax8.text(i, sr + 0.02, f'{sr:.0%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('07_rl/gridworld.png', dpi=80, bbox_inches='tight')
print("图片已保存：07_rl/gridworld.png")
plt.show()

# ============================================================
# Part 4: 超参数对比实验
# ============================================================
print("\nPart 4: 超参数对比实验")
print("=" * 50)

def train_and_eval(lr, gamma, n_ep=800):
    """给定超参数，训练并返回成功率历史"""
    env_tmp = ComplexGridWorld(size=8)
    agent_tmp = QLearningAgentWithHistory(
        env_tmp.n_states, env_tmp.n_actions,
        lr=lr, gamma=gamma,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.998
    )
    rewards = []
    for episode in range(1, n_ep + 1):
        s = env_tmp.reset()
        total_r = 0
        for _ in range(200):
            a = agent_tmp.choose_action(s)
            s_next, r, done = env_tmp.step(a)
            agent_tmp.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
            if done:
                break
        agent_tmp.decay_epsilon()
        rewards.append(total_r)
    return rewards

print("对比不同学习率（γ=0.95）：")
configs = [
    (0.05, 0.95, 'lr=0.05, γ=0.95'),
    (0.15, 0.95, 'lr=0.15, γ=0.95（默认）'),
    (0.5, 0.95, 'lr=0.5, γ=0.95'),
    (0.15, 0.5, 'lr=0.15, γ=0.5'),
    (0.15, 0.99, 'lr=0.15, γ=0.99'),
]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('超参数对学习效果的影响', fontsize=13)

window = 30
colors = ['blue', 'green', 'red', 'orange', 'purple']
for (lr, gamma, label), color in zip(configs[:3], colors[:3]):
    rewards = train_and_eval(lr, gamma)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes2[0].plot(smoothed, color=color, linewidth=2, label=label, alpha=0.8)
axes2[0].set_xlabel('Episode')
axes2[0].set_ylabel('平均奖励（平滑）')
axes2[0].set_title('不同学习率的效果\n（过小→学习慢，过大→不稳定）')
axes2[0].legend(fontsize=8)
axes2[0].grid(True, alpha=0.3)

for (lr, gamma, label), color in zip(configs[1:], colors[1:]):
    rewards = train_and_eval(lr, gamma)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes2[1].plot(smoothed, color=color, linewidth=2, label=label, alpha=0.8)
axes2[1].set_xlabel('Episode')
axes2[1].set_ylabel('平均奖励（平滑）')
axes2[1].set_title('不同折扣因子的效果\n（γ小→短视，γ大→长视）')
axes2[1].legend(fontsize=8)
axes2[1].grid(True, alpha=0.3)

print("  对比实验运行中（需要一点时间）...")
plt.tight_layout()
plt.savefig('07_rl/hyperparameter_comparison.png', dpi=80, bbox_inches='tight')
print("图片已保存：07_rl/hyperparameter_comparison.png")
plt.show()

# ============================================================
# Part 5: 最优路径追踪
# ============================================================
print("\nPart 5: 追踪最优路径")
print("=" * 50)

def trace_optimal_path(env, agent, max_steps=100):
    """用学到的贪婪策略追踪路径"""
    s = env.reset()
    path = [env.start_pos]
    total_reward = 0
    success = False

    for step in range(max_steps):
        a = agent.Q[s].argmax()
        s_next, r, done = env.step(a)
        path.append(env.pos)
        total_reward += r
        s = s_next
        if done:
            if r > 0:
                success = True
            break

    return path, total_reward, success

path, path_reward, success = trace_optimal_path(env, agent)
print(f"最优路径（{len(path)}步）：")
print(f"  路径：{' → '.join(str(p) for p in path)}")
print(f"  总奖励：{path_reward:.2f}")
print(f"  成功到达终点：{success}")

# 画最优路径
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 8))
draw_grid(ax3, env, agent.Q, title='最终学到的最优路径（蓝线）')

# 画路径
path_r = [p[0] for p in path]
path_c = [p[1] for p in path]
ax3.plot(path_c, path_r, 'b-', linewidth=3, alpha=0.8, zorder=10, label='最优路径')
ax3.plot(path_c[0], path_r[0], 'go', markersize=12, zorder=11, label='起点')
ax3.plot(path_c[-1], path_r[-1], 'r*', markersize=15, zorder=11, label='终点')
ax3.legend(fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('07_rl/optimal_path.png', dpi=80, bbox_inches='tight')
print("图片已保存：07_rl/optimal_path.png")
plt.show()

# ============================================================
# Part 6: 课程总结
# ============================================================
print("\n" + "=" * 60)
print("恭喜！你完成了整个课程体系！")
print("=" * 60)
print("""
你学习了以下内容：

第0章：数学基础
  ✓ 向量、矩阵、点积（线性代数）
  ✓ 导数、梯度、偏导数
  ✓ 概率、期望、信息熵
  ✓ 链式法则：反向传播的数学心脏

第1章：经典机器学习
  ✓ 线性回归（MSE损失、梯度下降）
  ✓ 逻辑回归（Sigmoid、交叉熵）
  ✓ 决策树（信息增益、ID3算法）
  ✓ SVM（间隔最大化、核技巧）

第2章：神经网络基础
  ✓ 感知机（单神经元学习规则）
  ✓ MLP + 手写反向传播（链式法则实战）
  ✓ 激活函数（ReLU、Sigmoid、GELU）
  ✓ 优化器（SGD、Momentum、Adam）

第3章：卷积神经网络
  ✓ 卷积直觉（局部连接、权重共享）
  ✓ im2col（把卷积变成矩阵乘法）
  ✓ 数字识别实战（1-6的CNN分类）
  ✓ 残差网络（梯度高速公路）

第4章：序列模型
  ✓ RNN（循环结构、BPTT）
  ✓ LSTM/GRU（门控机制）
  ✓ 时间序列预测实战

第5章：注意力与Transformer
  ✓ 注意力机制（QKV：英文阅读理解类比）
  ✓ 自注意力 + 多头注意力
  ✓ Transformer Encoder（LayerNorm + FFN + 残差）
  ✓ BERT（双向预训练语言模型）

第6章：生成模型
  ✓ 自编码器（压缩重建、潜空间）
  ✓ VAE（概率编码、重参数化技巧）

第7章：强化学习
  ✓ 核心概念（状态/动作/奖励/策略）
  ✓ 多臂赌博机（探索vs利用）
  ✓ Q-Learning（贝尔曼方程）
  ✓ 格世界实战（策略可视化）

下一步学习建议：
  1. PyTorch/TensorFlow：在真实框架上实现这些模型
  2. 计算机视觉：在 CIFAR-10/ImageNet 上训练 ResNet
  3. NLP：用 HuggingFace Transformers 使用预训练 BERT
  4. DQN：用神经网络代替 Q 表，玩 Atari 游戏
  5. 现代 LLM：GPT-4、Claude 等大语言模型的原理
""")

# ============================================================
# 思考题
# ============================================================
print("=" * 50)
print("思考题（最后一节）")
print("=" * 50)
print("""
1. 【DQN（深度 Q 网络）】
   Q-Learning 用表格，只适合小状态空间。
   DQN 用神经网络代替 Q 表：
     输入：状态（像素图像、传感器数据）
     输出：每个动作的 Q 值

   实现一个简单 DQN（用本课程第2章的 MLP + Q-Learning 更新）
   在格世界上测试，对比 tabular Q-Learning 的效果。

2. 【策略梯度（Policy Gradient）】
   Q-Learning 学的是"价值函数"（Q 值），再用贪婪策略。
   策略梯度直接学习策略 π(a|s;θ)（参数化的动作概率）。

   REINFORCE 算法：
     ∇J(θ) = E[∇log π(a|s;θ) * G_t]

   直觉：如果某个轨迹奖励高，增加其中每步动作的概率。
   这就是 AlphaGo、ChatGPT(RLHF) 的基础！

3. 【迁移学习】
   回顾整个课程：
     预训练 BERT → 微调 NLP 任务
     预训练 ResNet → 微调视觉任务
     预训练 RL 策略 → 迁移到新环境

   这种"先在大数据上预训练，再用小数据微调"的范式
   是现代 AI（GPT-4、Claude、Gemini）成功的核心。

   你认为下一个"预训练"的突破会发生在哪个领域？
   （机器人、科学发现、药物设计...？）
""")
