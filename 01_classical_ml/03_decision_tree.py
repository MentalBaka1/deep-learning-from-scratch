"""
==============================================================
第1章 第3节：决策树 —— 20个问题游戏
==============================================================

【为什么需要它？】
线性模型的局限：决策边界只能是直线（或超平面）。
真实世界很多规律不是线性的：
  "如果年龄 < 30 且 收入 > 5万 且 有工作，则贷款批准"
这种"嵌套的条件判断"，决策树天然擅长处理。

【生活类比】
玩"20个问题"猜物品：
  "是动物吗？" → 是
  "能飞吗？" → 不能
  "是哺乳动物吗？" → 是
  → "是狗！"

决策树就是这个游戏的自动学习版本：
从数据中找出"最有效的问题"来问。

【存在理由】
解决问题：线性模型无法捕捉非线性决策边界
核心思想：递归地选择"最能区分数据"的特征来分裂
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ============================================================
# Part 1: 信息熵 —— 衡量数据集的"纯度"
# ============================================================
print("=" * 50)
print("Part 1: 信息熵 —— 衡量数据混乱程度")
print("=" * 50)

"""
信息熵 H(S) = -Σ p_k * log₂(p_k)

直觉：
  - H = 0：数据集完全纯（所有样本同一类），不需要再问问题了
  - H 最大：数据集完全混乱（各类等比例），需要问很多问题

我们希望每次分裂后，子节点的熵更小（数据更纯）。
"""

def entropy(labels):
    """计算标签列表的信息熵"""
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    probs = [c / n for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

# 几种不同纯度的数据集
print("不同数据集的熵：")
datasets = [
    ([1, 1, 1, 1, 1], "全是类1（完全纯）"),
    ([1, 1, 1, 1, 0], "4个1，1个0（较纯）"),
    ([1, 1, 0, 0, 1], "3个1，2个0（中等）"),
    ([1, 1, 0, 0, 0, 0], "2个1，4个0（中等）"),
    ([1, 0, 1, 0], "各半（完全混乱）"),
]
for data, desc in datasets:
    h = entropy(data)
    print(f"  {data} → H = {h:.4f}  （{desc}）")

# ============================================================
# Part 2: 信息增益 —— 选择最佳分裂特征
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 信息增益 —— 哪个问题最有用？")
print("=" * 50)

"""
信息增益 IG = 父节点熵 - 加权子节点熵

IG(S, 特征A) = H(S) - Σ (|S_v|/|S|) * H(S_v)
  S = 当前数据集
  S_v = 按特征A的取值v分裂后的子集

我们选信息增益最大的特征来分裂！
"""

def information_gain(parent_labels, splits):
    """
    parent_labels: 父节点的标签列表
    splits: [(子集1的标签), (子集2的标签), ...]
    """
    parent_entropy = entropy(parent_labels)
    n_parent = len(parent_labels)

    weighted_child_entropy = sum(
        (len(split) / n_parent) * entropy(split)
        for split in splits
    )
    return parent_entropy - weighted_child_entropy

# 示例：预测一个人是否喜欢打篮球
# 特征：身高（高/矮），年龄（年轻/年老）
# 标签：1=喜欢，0=不喜欢

data = [
    # (身高, 年龄, 标签)
    ('高', '年轻', 1), ('高', '年轻', 1), ('高', '年老', 1),
    ('矮', '年轻', 0), ('矮', '年轻', 0), ('矮', '年老', 0),
    ('高', '年老', 0), ('矮', '年轻', 1),
]

all_labels = [d[2] for d in data]
print(f"完整数据集：{all_labels}，H = {entropy(all_labels):.4f}")

# 按"身高"分裂
tall_labels = [d[2] for d in data if d[0] == '高']
short_labels = [d[2] for d in data if d[0] == '矮']
ig_height = information_gain(all_labels, [tall_labels, short_labels])
print(f"\n按'身高'分裂：")
print(f"  高：{tall_labels}，H = {entropy(tall_labels):.4f}")
print(f"  矮：{short_labels}，H = {entropy(short_labels):.4f}")
print(f"  信息增益 IG = {ig_height:.4f}")

# 按"年龄"分裂
young_labels = [d[2] for d in data if d[1] == '年轻']
old_labels = [d[2] for d in data if d[1] == '年老']
ig_age = information_gain(all_labels, [young_labels, old_labels])
print(f"\n按'年龄'分裂：")
print(f"  年轻：{young_labels}，H = {entropy(young_labels):.4f}")
print(f"  年老：{old_labels}，H = {entropy(old_labels):.4f}")
print(f"  信息增益 IG = {ig_age:.4f}")

print(f"\n结论：{'身高' if ig_height > ig_age else '年龄'}的信息增益更大，优先用它分裂！")

# ============================================================
# Part 3: 手写决策树（ID3算法）
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 手写决策树")
print("=" * 50)

class DecisionNode:
    """决策树节点"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分裂特征索引
        self.threshold = threshold  # 分裂阈值（连续特征）
        self.left = left            # 左子树（≤ threshold）
        self.right = right          # 右子树（> threshold）
        self.value = value          # 叶节点的类别值

class DecisionTreeClassifier:
    """手写决策树分类器（连续特征版）"""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件（叶节点）
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # 叶节点：返回最多数类别
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)

        # 找最佳分裂点
        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionNode(value=leaf_value)

        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 递归构建子树
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    def _best_split(self, X, y, n_features):
        """找所有特征中，信息增益最大的分裂点"""
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        parent_entropy = entropy(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = information_gain(
                    y,
                    [y[left_mask], y[right_mask]]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:  # 叶节点
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# ============================================================
# Part 4: 在真实数据上测试
# ============================================================
print("Part 4: 测试决策树")
print("=" * 50)

# 生成非线性数据集（圆形边界，线性模型搞不定！）
n = 300
angles = np.random.uniform(0, 2*np.pi, n)
radii = np.random.uniform(0, 3, n)
X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
# 内圈是类0，外圈是类1
y = (radii > 1.5).astype(int)
# 加噪声
y_noisy = y.copy()
flip_idx = np.random.choice(n, size=20)
y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

# 训练决策树
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X, y_noisy)
accuracy = tree.score(X, y_noisy)
print(f"决策树准确率：{accuracy:.2%}")

# 可视化决策边界
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (depth, title) in zip(axes, [(2, '浅决策树（depth=2）'), (8, '深决策树（depth=8）')]):
    tree_depth = DecisionTreeClassifier(max_depth=depth)
    tree_depth.fit(X, y_noisy)

    h = 0.05
    xx, yy = np.meshgrid(np.arange(-3.5, 3.5, h), np.arange(-3.5, 3.5, h))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = tree_depth.predict(X_grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.4)
    ax.scatter(X[y_noisy==0, 0], X[y_noisy==0, 1], c='red', s=20, alpha=0.5, label='类0')
    ax.scatter(X[y_noisy==1, 0], X[y_noisy==1, 1], c='blue', s=20, alpha=0.5, label='类1')
    acc = tree_depth.score(X, y_noisy)
    ax.set_title(f'{title}\n训练准确率：{acc:.2%}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('01_classical_ml/decision_tree.png', dpi=100, bbox_inches='tight')
print("图片已保存：01_classical_ml/decision_tree.png")
plt.show()

print("\n观察：深度树把训练数据记得很好（过拟合），浅度树更泛化。")
print("这就是为什么需要控制树的深度（超参数！）")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【熵 vs 基尼系数】
   除了信息熵，CART 算法用基尼系数：
   Gini = 1 - Σ p_k²
   对同样的数据集，计算熵和基尼系数，看结果是否排序一致。
   （通常两者效果相近，基尼系数计算更快）

2. 【过拟合实验】
   把 max_depth 从 1 逐渐增加到 15。
   如果有训练集和测试集，画出"训练准确率"和"测试准确率"随深度的变化。
   找到最优深度（测试准确率最高的点）。

3. 【随机森林直觉】
   单棵决策树容易过拟合。解决方案是"随机森林"：
   - 训练 100 棵不同的决策树（每棵用随机子样本和随机特征）
   - 让所有树投票，取多数
   为什么这样做会比单棵树更好？（提示：偏差-方差权衡）
""")
