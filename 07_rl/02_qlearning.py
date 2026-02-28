"""
==============================================================
第7章 第2节：Q-Learning —— 贝尔曼方程变成代码
==============================================================

【为什么需要它？】
多臂赌博机只有动作选择，没有状态转移。
真实世界中，行动会改变状态，需要规划多步决策：
  下棋：现在这步会影响未来很多步
  导航：每次转向都改变位置，需要规划整条路线

Q-Learning 解决了这个问题：
  维护一个 Q(s, a) 表，记录"在状态s执行动作a的期望总回报"
  用贝尔曼方程不断更新这个估计

【生活类比：导航软件的路程估计】
  你要从A开到Z，导航软件怎么估计各条路的"期望总时间"？

  1. 从Z（终点）往回看
  2. 到Z的时间 = 0
  3. 到Y的时间 = Y到Z的距离 + 到Z的时间
  4. 到X的时间 = X到Y的距离 + 到Y的时间
  5. 依此类推...

  但路况是随机的（有可能堵车），所以用"期望"：
    Q(位置, 方向) = 即时代价 + γ * 最优Q(下一个位置)

  这就是贝尔曼方程！Q-Learning 就是用真实体验不断修正这个估计。

【贝尔曼方程（Bellman Equation）】
  Q*(s, a) = E[r + γ * max_a' Q*(s', a')]

  解读：
    Q*(s, a) = 在状态 s 执行动作 a 的最优价值
             = 立即奖励 r + 折扣 γ × 下一状态的最优价值
  这是一个递推方程，最优 Q 值满足这个等式！

【存在理由】
解决问题：有序列状态的多步决策问题，不需要环境模型（无模型 RL）
核心思想：用 TD（时序差分）更新估计 Q 值，迭代趋近最优策略
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(42)

# ============================================================
# Part 1: 贝尔曼方程的直觉
# ============================================================
print("=" * 50)
print("Part 1: 贝尔曼方程")
print("=" * 50)

"""
核心公式：
  Q(s, a) ← Q(s, a) + α * [目标 - 当前估计]

  目标 = r + γ * max_a' Q(s', a')  ← 贝尔曼方程的右侧

  TD 误差（Temporal Difference Error）= 目标 - 当前估计
  就是"现实"和"预期"的差距

α（学习率）：多快相信新信息

  类比：导航软件如何修正估计？
    - 预估A→Z要1小时
    - 实际走A→B用了20分钟，B→Z预估45分钟（总65分钟）
    - TD 误差 = 65 - 60 = +5分钟（实际比预估慢5分钟）
    - 修正：Q(A, 向B走) += α * 5
"""

print("TD 更新规则演示：")
# 简单的两状态示例
Q_table = {('A', 'go_B'): 60.0, ('B', 'go_end'): 45.0}
r_AB = 20  # A→B 的实际时间
gamma = 0.9
alpha = 0.1

Q_B_best = 45.0
target = r_AB + gamma * Q_B_best
td_error = target - Q_table[('A', 'go_B')]

print(f"  Q(A, go_B) 当前估计 = {Q_table[('A', 'go_B')]}分钟")
print(f"  实际 A→B 时间 = {r_AB}分钟")
print(f"  B→终点 最优估计 Q(B, go_end) = {Q_B_best}分钟")
print(f"  TD 目标 = {r_AB} + {gamma}*{Q_B_best} = {target:.1f}分钟")
print(f"  TD 误差 = {target:.1f} - {Q_table[('A', 'go_B')]} = {td_error:.1f}")
print(f"  更新：Q(A, go_B) += {alpha} * {td_error:.1f} = {Q_table[('A', 'go_B')] + alpha*td_error:.2f}")
Q_table[('A', 'go_B')] += alpha * td_error
print(f"  新的 Q(A, go_B) = {Q_table[('A', 'go_B')]:.2f}分钟")

# ============================================================
# Part 2: 简单格世界环境
# ============================================================
print("\nPart 2: 格世界（GridWorld）环境")
print("=" * 50)

"""
格世界（GridWorld）是 RL 的经典测试环境：

  ┌───┬───┬───┬───┐
  │ S │   │   │   │    S = 起点（Start）
  ├───┼───┼───┼───┤    G = 终点（Goal），奖励 +10
  │   │ W │   │ G │    W = 墙（Wall），不可进入
  ├───┼───┼───┼───┤    T = 陷阱（Trap），奖励 -10
  │   │   │ T │   │    每步奖励 -0.1（鼓励走短路径）
  └───┴───┴───┴───┘

动作：上、下、左、右（4个）
状态：(行, 列)，共 4×4=16 个状态
"""

class GridWorld:
    """
    简单格世界环境
    """
    def __init__(self, grid_size=4):
        self.size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 上/右/下/左

        # 特殊格子的位置（行, 列）
        self.start = (0, 0)
        self.goal = (1, 3)    # 终点
        self.traps = [(2, 2)] # 陷阱
        self.walls = [(1, 1)] # 墙

        # 动作向量：上/右/下/左
        self.action_deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['↑', '→', '↓', '←']

        self.reset()

    def reset(self):
        """重置到起点"""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        执行动作，返回 (新状态, 奖励, 是否结束)
        """
        r, c = self.state
        dr, dc = self.action_deltas[action]
        nr, nc = r + dr, c + dc

        # 检查边界和墙
        if (nr < 0 or nr >= self.size or nc < 0 or nc >= self.size or
                (nr, nc) in self.walls):
            nr, nc = r, c  # 撞墙：留在原地

        self.state = (nr, nc)

        # 奖励和终止条件
        if self.state == self.goal:
            return self.state, 10.0, True   # 到达终点
        elif self.state in self.traps:
            return self.state, -10.0, True  # 掉入陷阱
        else:
            return self.state, -0.1, False  # 普通步骤

    def state_to_idx(self, state):
        return state[0] * self.size + state[1]

    def idx_to_state(self, idx):
        return (idx // self.size, idx % self.size)

    def render_grid(self, Q_table=None, policy=None):
        """打印格世界（可选：显示 Q 值或策略）"""
        symbols = {self.start: 'S', self.goal: 'G'}
        for t in self.traps:
            symbols[t] = 'T'
        for w in self.walls:
            symbols[w] = 'W'

        print("  " + " ".join([f"{c:4d}" for c in range(self.size)]))
        for r in range(self.size):
            row_str = f"{r} "
            for c in range(self.size):
                cell = (r, c)
                if cell in symbols:
                    row_str += f"[{symbols[cell]:2s}] "
                elif policy is not None:
                    state_idx = self.state_to_idx(cell)
                    best_action = policy[state_idx]
                    row_str += f"[{self.action_names[best_action]:2s}] "
                else:
                    row_str += "[   ] "
            print(row_str)

env = GridWorld(grid_size=4)
print("格世界地图：")
env.render_grid()
print(f"\n  S=起点, G=终点(+10), T=陷阱(-10), W=墙, 每步-0.1")

# ============================================================
# Part 3: Q-Learning 算法实现
# ============================================================
print("\nPart 3: Q-Learning 实现")
print("=" * 50)

"""
Q-Learning 算法：

  初始化 Q(s, a) = 0（对所有 s, a）

  对每个 episode：
    重置到起始状态 s
    循环直到终止：
      1. 选择动作：ε-greedy
         以概率 ε 随机选择（探索）
         以概率 1-ε 选择 argmax_a Q(s, a)（利用）
      2. 执行动作，获得 r, s'
      3. TD 更新：
         Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
      4. s ← s'

  Q-Learning 是"离策略"（off-policy）算法：
    更新时用 max_a' Q(s', a')（贪婪策略）
    选择动作时用 ε-greedy（行为策略）
    两者不一致没关系！Q 收敛到最优策略
"""

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q 表初始化为 0
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx):
        """ε-greedy 动作选择"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # 随机探索
        else:
            return self.Q[state_idx].argmax()  # 贪婪利用

    def update(self, state_idx, action, reward, next_state_idx, done):
        """
        Q-Learning 更新（贝尔曼方程）
        """
        # 当前 Q 值
        current_Q = self.Q[state_idx, action]

        # TD 目标
        if done:
            target = reward  # 终止状态没有未来奖励
        else:
            target = reward + self.gamma * self.Q[next_state_idx].max()

        # TD 误差和更新
        td_error = target - current_Q
        self.Q[state_idx, action] += self.lr * td_error

        return td_error

    def decay_epsilon(self):
        """逐步减少探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """返回贪婪策略（每个状态的最优动作）"""
        return self.Q.argmax(axis=1)

# 训练 Q-Learning
env = GridWorld(grid_size=4)
agent = QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    lr=0.1,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995
)

n_episodes = 1000
episode_rewards = []
episode_lengths = []
epsilon_history = []

print(f"训练 Q-Learning（{n_episodes}个episode）...")

for episode in range(n_episodes):
    state = env.reset()
    state_idx = env.state_to_idx(state)
    total_reward = 0
    steps = 0
    max_steps = 100

    while steps < max_steps:
        action = agent.choose_action(state_idx)
        next_state, reward, done = env.step(action)
        next_state_idx = env.state_to_idx(next_state)

        agent.update(state_idx, action, reward, next_state_idx, done)

        state_idx = next_state_idx
        total_reward += reward
        steps += 1

        if done:
            break

    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    epsilon_history.append(agent.epsilon)

    if episode % 200 == 0:
        avg_reward = np.mean(episode_rewards[-50:]) if episode >= 50 else np.mean(episode_rewards)
        print(f"  Episode {episode:4d}: ε={agent.epsilon:.3f}, "
              f"最近50轮平均奖励={avg_reward:.2f}")

# 打印学到的策略
print("\n学到的最优策略（Q-Learning）：")
policy = agent.get_policy()
env.render_grid(policy=policy)

# 评估：用贪婪策略跑几轮
success_count = 0
n_eval = 100
for _ in range(n_eval):
    state = env.reset()
    state_idx = env.state_to_idx(state)
    for _ in range(50):
        action = agent.Q[state_idx].argmax()  # 纯贪婪
        state, reward, done = env.step(action)
        state_idx = env.state_to_idx(state)
        if done:
            if reward > 0:
                success_count += 1
            break

print(f"\n评估结果（{n_eval}次纯贪婪）：到达终点率 = {success_count/n_eval:.1%}")

# ============================================================
# Part 4: Q 表分析
# ============================================================
print("\nPart 4: Q 表分析")
print("=" * 50)

print("最优 Q 值（每个状态的最大 Q 值）：")
for r in range(env.size):
    row_str = ""
    for c in range(env.size):
        state = (r, c)
        if state in env.walls:
            row_str += "  W    "
        else:
            state_idx = env.state_to_idx(state)
            best_q = agent.Q[state_idx].max()
            row_str += f"{best_q:6.2f} "
    print(row_str)

print("\n动作选择（四个方向的 Q 值）：")
for r in range(env.size):
    for c in range(env.size):
        state = (r, c)
        if state not in env.walls:
            state_idx = env.state_to_idx(state)
            q_vals = agent.Q[state_idx]
            best_action = q_vals.argmax()
            print(f"  ({r},{c}): {[f'{q:.2f}' for q in q_vals]} → {env.action_names[best_action]}")

# ============================================================
# Part 5: 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Q-Learning：格世界寻路', fontsize=14)

# 1. 训练奖励曲线
ax = axes[0][0]
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax.plot(episode_rewards, alpha=0.3, color='blue', linewidth=0.5)
ax.plot(range(window-1, len(episode_rewards)), smoothed, 'r-', linewidth=2, label='移动平均')
ax.set_xlabel('Episode')
ax.set_ylabel('总奖励')
ax.set_title(f'训练曲线\n（最终平均奖励={smoothed[-1]:.2f}）')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. ε 衰减曲线
ax = axes[0][1]
ax.plot(epsilon_history, 'g-', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('ε（探索率）')
ax.set_title('ε 衰减过程\n（从探索到利用）')
ax.grid(True, alpha=0.3)

# 3. Q 值热力图（最大 Q 值）
ax = axes[0][2]
q_map = np.zeros((env.size, env.size))
for r in range(env.size):
    for c in range(env.size):
        state = (r, c)
        if state in env.walls:
            q_map[r, c] = np.nan
        else:
            state_idx = env.state_to_idx(state)
            q_map[r, c] = agent.Q[state_idx].max()

# 处理 NaN（墙）
masked_q = np.ma.masked_invalid(q_map)
cmap = plt.cm.RdYlGn
cmap.set_bad('gray')
im = ax.imshow(masked_q, cmap=cmap, aspect='auto')
plt.colorbar(im, ax=ax)

# 标注特殊格子
symbols = {env.start: 'S', env.goal: 'G'}
for t in env.traps:
    symbols[t] = 'T'
for w in env.walls:
    symbols[w] = 'W'
for (r, c), sym in symbols.items():
    ax.text(c, r, sym, ha='center', va='center', fontsize=16,
            fontweight='bold', color='black')

ax.set_title('状态值热力图\n（绿=价值高，红=价值低，灰=墙）')
ax.set_xticks(range(env.size))
ax.set_yticks(range(env.size))

# 4. 策略箭头图
ax = axes[1][0]
ax.set_xlim(-0.5, env.size - 0.5)
ax.set_ylim(env.size - 0.5, -0.5)
ax.set_aspect('equal')

# 画格子
for r in range(env.size):
    for c in range(env.size):
        state = (r, c)
        color = 'white'
        if state in env.walls:
            color = 'gray'
        elif state == env.goal:
            color = 'lightgreen'
        elif state in env.traps:
            color = 'salmon'
        elif state == env.start:
            color = 'lightyellow'

        rect = plt.Rectangle([c-0.5, r-0.5], 1, 1,
                              facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        if state in symbols:
            ax.text(c, r, symbols[state], ha='center', va='center',
                    fontsize=14, fontweight='bold')
        elif state not in env.walls:
            state_idx = env.state_to_idx(state)
            best_action = agent.Q[state_idx].argmax()
            dr, dc = env.action_deltas[best_action]
            ax.annotate('', xy=(c + dc*0.3, r + dr*0.3),
                       xytext=(c, r),
                       arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))

ax.set_title('学到的最优策略（箭头）\n箭头=最优动作方向')
ax.grid(False)

# 5. TD 误差的收敛过程（理论说明）
ax = axes[1][1]
td_errors_window = []
# 重新跑一遍记录 TD 误差
env2 = GridWorld(grid_size=4)
agent2 = QLearningAgent(env2.n_states, env2.n_actions, lr=0.1, gamma=0.95,
                        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
for episode in range(500):
    state = env2.reset()
    state_idx = env2.state_to_idx(state)
    ep_td = []
    for _ in range(100):
        action = agent2.choose_action(state_idx)
        next_state, reward, done = env2.step(action)
        next_idx = env2.state_to_idx(next_state)
        td = agent2.update(state_idx, action, reward, next_idx, done)
        ep_td.append(abs(td))
        state_idx = next_idx
        if done:
            break
    td_errors_window.append(np.mean(ep_td))
    agent2.decay_epsilon()

smooth_td = np.convolve(td_errors_window, np.ones(20)/20, mode='valid')
ax.plot(smooth_td, 'purple', linewidth=2)
ax.set_xlabel('Episode')
ax.set_ylabel('平均 |TD 误差|')
ax.set_title('TD 误差收敛\n（TD误差趋近0 → Q值收敛）')
ax.grid(True, alpha=0.3)

# 6. Q-Learning 算法伪码
ax = axes[1][2]
ax.axis('off')
algo_text = """
Q-Learning 算法伪码：

初始化：Q(s,a) = 0，对所有 s, a

For episode = 1, 2, ..., N:
  s = 重置环境

  While not done:
    # 选择动作（ε-greedy）
    if rand() < ε:
      a = 随机动作（探索）
    else:
      a = argmax_a Q(s, a)（贪婪）

    # 与环境交互
    s', r, done = env.step(a)

    # Q 值更新（贝尔曼方程）
    if done:
      target = r
    else:
      target = r + γ * max_a' Q(s', a')

    Q(s,a) += α * (target - Q(s,a))

    s = s'  # 移动到下一状态

  ε *= 衰减系数  # 减少探索

提取策略：
  π(s) = argmax_a Q(s, a)
"""
ax.text(0.05, 0.95, algo_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('Q-Learning 算法')

plt.tight_layout()
plt.savefig('07_rl/qlearning.png', dpi=80, bbox_inches='tight')
print("\n图片已保存：07_rl/qlearning.png")
plt.show()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【SARSA vs Q-Learning】
   SARSA 是另一种 TD 算法，但它是"在策略"（on-policy）：
     Q(s,a) += α * [r + γ * Q(s', a') - Q(s,a)]
   其中 a' 是实际选择的动作（而不是贪婪的 max_a' Q(s',a')）

   SARSA 的更新用"实际行为"，Q-Learning 用"最优行为"
   - 在有陷阱的格世界里，哪个更保守？为什么？
   - 修改代码实现 SARSA，对比两者的策略差异

2. 【Q 表的局限性】
   Q-Learning 需要一个 Q(s,a) 表。
   如果状态空间是连续的（如机器人的关节角度），无法枚举！
   → 解决方案：用神经网络代替 Q 表 = Deep Q-Network（DQN）
   Q(s, a; θ) ≈ 神经网络（输入状态，输出每个动作的Q值）
   这就是 DeepMind 用来玩 Atari 游戏的方法。

3. 【奖励的折扣因子实验】
   在上面的格世界中，修改 γ：
   - γ=0.5：智能体更关注眼前奖励，路径可能较短但不总是最优
   - γ=0.99：智能体更关注长期回报，可能找到更优路径
   - γ=1.0：无折扣，可能不收敛（如果有无限长的 episode）
   运行对比实验，观察 Q 表的差异。
""")
