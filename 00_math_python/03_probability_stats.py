"""
==============================================================
第0章 第3节：概率与统计 —— 不确定性的语言
==============================================================

【为什么需要它？】
深度学习处理的是"不确定"的世界：
- 分类网络输出的是概率（"这张图有80%可能是猫"）
- 损失函数（交叉熵）来自信息论
- 权重初始化用正态分布
- 数据增强随机性到处都是

如果不懂概率，很多深度学习的设计决策会显得莫名其妙。

【生活类比】
掷骰子 → 概率分布
天气预报"70%降雨" → 条件概率
考试平均分和标准差 → 期望与方差
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ============================================================
# Part 1: 概率的本质 —— 长期频率
# ============================================================
print("=" * 50)
print("Part 1: 概率 —— 用模拟验证理论")
print("=" * 50)

"""
概率的频率派解释：
  做 N 次实验，事件 A 发生了 k 次，则 P(A) ≈ k/N
  当 N → ∞ 时，频率 → 真实概率
"""

# 模拟掷骰子
def simulate_dice(n_rolls):
    rolls = np.random.randint(1, 7, size=n_rolls)  # 1-6
    counts = Counter(rolls)
    probs = {k: v/n_rolls for k, v in sorted(counts.items())}
    return probs

print("掷骰子 N 次，每个面出现的频率：")
for n in [100, 1000, 100000]:
    probs = simulate_dice(n)
    print(f"  N={n:6d}: {[f'{p:.3f}' for p in probs.values()]}", end="")
    print(f"  （理论值都是 {1/6:.3f}）")

# ============================================================
# Part 2: 概率分布 —— 描述随机变量的行为
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 重要概率分布")
print("=" * 50)

"""
深度学习中最常见的分布：

1. 均匀分布：每个值等可能出现（随机初始化某些权重）
2. 正态（高斯）分布：自然界最常见，权重初始化常用
3. 伯努利分布：0或1（Dropout！以p概率"关掉"神经元）
4. 类别分布：多个类别（分类问题的输出就是类别分布）
"""

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('深度学习中常见的概率分布', fontsize=14)

# 1. 正态分布
ax = axes[0][0]
x = np.linspace(-4, 4, 1000)
for mu, sigma, label in [(0, 1, 'μ=0,σ=1（标准正态）'),
                           (0, 2, 'μ=0,σ=2（更宽）'),
                           (2, 1, 'μ=2,σ=1（右移）')]:
    y = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2)
    ax.plot(x, y, linewidth=2, label=label)
ax.set_title('正态分布（权重初始化）')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. 模拟神经网络权重初始化
ax = axes[0][1]
n_weights = 10000
xavier_weights = np.random.randn(n_weights) * np.sqrt(1/512)  # Xavier初始化
he_weights = np.random.randn(n_weights) * np.sqrt(2/512)       # He初始化
ax.hist(xavier_weights, bins=50, alpha=0.5, label='Xavier初始化', density=True)
ax.hist(he_weights, bins=50, alpha=0.5, label='He初始化', density=True)
ax.set_title('不同权重初始化方案')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 伯努利分布（Dropout模拟）
ax = axes[1][0]
p_keep = 0.8  # 保留概率
n_neurons = 1000
kept = np.random.binomial(1, p_keep, n_neurons)
ax.bar(['关闭(0)', '保留(1)'], [np.sum(kept==0), np.sum(kept==1)],
       color=['red', 'green'], alpha=0.7)
ax.set_title(f'Dropout（p_keep={p_keep}）\n每次随机关掉一些神经元')
ax.set_ylabel('神经元数量')
ax.grid(True, alpha=0.3, axis='y')

# 4. Softmax 输出（类别分布）
ax = axes[1][1]
logits = np.array([2.0, 1.0, 0.5, 3.0, -1.0])  # 网络原始输出
softmax = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
classes = [f'类{i}' for i in range(5)]
ax.bar(classes, softmax, color='steelblue', alpha=0.7)
ax.set_title('Softmax输出（分类概率分布）\n所有概率之和=1')
ax.set_ylabel('概率')
for i, (c, p) in enumerate(zip(classes, softmax)):
    ax.text(i, p+0.01, f'{p:.2f}', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('00_math_python/probability_distributions.png', dpi=100, bbox_inches='tight')
print("图片已保存：00_math_python/probability_distributions.png")
plt.show()

# ============================================================
# Part 3: 期望与方差
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 期望与方差 —— 分布的概括")
print("=" * 50)

"""
期望 E[X]：随机变量的"平均值"（加权平均）
  E[X] = Σ x * P(X=x)

方差 Var[X]：随机变量离均值的"平均偏离程度"
  Var[X] = E[(X - E[X])²] = E[X²] - (E[X])²

标准差 σ = √Var[X]：和 X 同单位，更直观
"""

# 用掷骰子验证
dice_outcomes = np.arange(1, 7)
dice_probs = np.ones(6) / 6  # 均匀分布

expected_value = np.sum(dice_outcomes * dice_probs)
variance = np.sum((dice_outcomes - expected_value)**2 * dice_probs)
std_dev = np.sqrt(variance)

print(f"骰子（1-6均匀分布）：")
print(f"  期望 E[X] = {expected_value:.2f}  （=3.5，长期平均）")
print(f"  方差 Var = {variance:.2f}")
print(f"  标准差 σ = {std_dev:.2f}")

# 用模拟验证
samples = np.random.randint(1, 7, 100000)
print(f"\n  模拟验证（10万次）：")
print(f"  样本均值 = {np.mean(samples):.4f}  （接近3.5）")
print(f"  样本标准差 = {np.std(samples):.4f}")

# 为什么神经网络关心方差？
print("\n为什么神经网络关心方差？")
print("  如果权重初始化方差太大 → 激活值爆炸，训练不稳定")
print("  如果权重初始化方差太小 → 激活值消失，梯度消失")
print("  Xavier/He 初始化就是通过计算'合适的方差'来保持激活值稳定！")

# ============================================================
# Part 4: 信息熵 —— 不确定性的度量
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 信息熵 —— 不确定性有多大？")
print("=" * 50)

"""
香农信息熵（Shannon Entropy）：
  H(P) = -Σ p(x) * log₂(p(x))

直觉：
  - H = 0：完全确定（一个面概率为1）
  - H 最大：完全不确定（均匀分布）

单位：bits（用log₂）
  抛一枚公平硬币：H = 1 bit（需要1个比特来描述结果）
  掷一个公平骰子：H = log₂(6) ≈ 2.58 bits

深度学习中的用途：
  - 决策树用信息增益（熵的减少量）来选择最优分裂特征
  - 交叉熵损失函数（Cross-Entropy Loss）= 预测分布和真实分布的差异
"""

def entropy(probs):
    """计算概率分布的熵（以nats为单位，用ln）"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # 去掉0，避免 log(0)
    return -np.sum(probs * np.log(probs))

# 不同确定程度的分布
distributions = [
    ([1.0, 0.0], "完全确定（知道答案）"),
    ([0.9, 0.1], "很确定"),
    ([0.7, 0.3], "比较确定"),
    ([0.5, 0.5], "完全不确定（均匀分布）"),
]

print("二元分类问题，不同确定程度的熵：")
for probs, desc in distributions:
    h = entropy(probs)
    print(f"  P = {probs}，H = {h:.4f}  —— {desc}")

# ============================================================
# Part 5: 交叉熵 —— 深度学习的损失函数
# ============================================================
print("\n" + "=" * 50)
print("Part 5: 交叉熵损失 —— 为什么分类不用MSE？")
print("=" * 50)

"""
交叉熵 H(P, Q) = -Σ p(x) * log q(x)
  P = 真实分布（one-hot，只有正确类是1，其余0）
  Q = 预测分布（softmax输出的概率）

直觉：衡量"用预测分布Q来描述真实分布P，需要多少比特"
  预测越准（Q接近P）→ 交叉熵越小

为什么不用均方误差（MSE）做分类？
  1. MSE 假设误差服从高斯分布，不适合分类（类别是离散的）
  2. 交叉熵配合 softmax，梯度非常干净（推导见 02_neural_networks）
  3. 交叉熵对"错误且自信"的预测惩罚极大
"""

def cross_entropy(y_true, y_pred_probs):
    """
    y_true: 真实类别的 one-hot 向量
    y_pred_probs: softmax 输出的概率向量
    """
    eps = 1e-10  # 防止 log(0)
    return -np.sum(y_true * np.log(y_pred_probs + eps))

# 示例：3分类问题，真实类别是第1类
y_true = np.array([1.0, 0.0, 0.0])  # one-hot

print("真实标签：[1, 0, 0]（第一类）")
print("\n不同预测的交叉熵损失：")
predictions = [
    ([0.9, 0.05, 0.05], "预测正确且很自信"),
    ([0.6, 0.2, 0.2],   "预测正确但不太确定"),
    ([0.34, 0.33, 0.33], "基本随机猜"),
    ([0.1, 0.8, 0.1],    "预测错了（第2类）"),
    ([0.01, 0.98, 0.01], "预测错了且很自信（最差）"),
]

for pred, desc in predictions:
    loss = cross_entropy(y_true, np.array(pred))
    print(f"  预测={pred}，Loss={loss:.4f}  —— {desc}")

print("\n注意：错误且自信的预测，loss 非常大！这正是我们想要的惩罚。")

# ============================================================
# Part 6: 贝叶斯公式（直觉）
# ============================================================
print("\n" + "=" * 50)
print("Part 6: 贝叶斯公式 —— 用证据更新信念")
print("=" * 50)

"""
P(A|B) = P(B|A) * P(A) / P(B)

直觉：
  P(A) = 先验概率（看到证据B之前的信念）
  P(A|B) = 后验概率（看到证据B之后更新的信念）
  P(B|A) = 似然（如果A成立，B出现的概率）

例子：医疗诊断
  疾病患病率 P(病) = 0.001（先验）
  检测灵敏度 P(阳|病) = 0.99
  假阳性率 P(阳|健康) = 0.05
  如果检测结果阳性，真的患病的概率是多少？
"""

p_disease = 0.001     # 患病率（先验）
p_pos_given_disease = 0.99   # 检测灵敏度
p_pos_given_healthy = 0.05   # 假阳性率
p_healthy = 1 - p_disease

# P(阳性) = P(阳|病)*P(病) + P(阳|健康)*P(健康)
p_positive = p_pos_given_disease * p_disease + p_pos_given_healthy * p_healthy

# P(病|阳性) = P(阳|病) * P(病) / P(阳性)
p_disease_given_pos = p_pos_given_disease * p_disease / p_positive

print(f"患病率：{p_disease:.1%}")
print(f"检测阳性后，真的患病概率：{p_disease_given_pos:.2%}")
print(f"（尽管检测很准，但因为患病率极低，阳性结果大多是假阳性！）")
print(f"\n深度学习与贝叶斯：")
print(f"  分类网络的 softmax 输出 ≈ P(类别|输入图像)（后验概率）")
print(f"  这就是为什么我们说网络输出'概率'")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【信息熵实验】
   计算以下分布的熵，解释为什么结果不同：
   - 6面骰子：[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
   - 偏骰子：  [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
   - 固定骰子：[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

2. 【交叉熵 vs MSE】
   对3分类问题，设真实标签 y_true = [1, 0, 0]
   计算以下两种预测的 CrossEntropy 和 MSE：
   预测A = [0.7, 0.2, 0.1]
   预测B = [0.4, 0.3, 0.3]
   哪个损失区分度更大？

3. 【正态分布与权重初始化】
   对于 100 → 50 → 10 的三层网络：
   - 如果第一层权重用 randn(100, 50) * 1.0 初始化
   - 输入 x = randn(batch=32, features=100)
   - 第一层输出 (x @ W1) 的标准差大概是多少？
   - 为什么会激活值"爆炸"？Xavier 初始化怎么解决这个问题？
""")
