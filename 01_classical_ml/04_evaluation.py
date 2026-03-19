"""
====================================================================
第1章 · 第4节 · 模型评估方法
====================================================================

【一句话总结】
好的评估方法比好的模型更重要——不会评估，就不知道模型是否真正学到了东西。

【为什么深度学习需要这个？】
- 训练/验证/测试集划分是所有ML/DL实验的基础
- 准确率不是万能的（类别不平衡时可能误导）
- 理解 Precision/Recall/F1 才能正确评估分类模型
- 交叉验证帮助在数据有限时可靠评估

【核心概念】

1. 训练集 / 验证集 / 测试集
   - 训练集：模型学习用
   - 验证集：调超参数、选模型用（模型不直接学习这部分数据）
   - 测试集：最终评估用（只在最后用一次）
   - 典型比例：70/15/15 或 80/10/10

2. 混淆矩阵（Confusion Matrix）
   - TP（真正例）、FP（假正例）、TN（真负例）、FN（假负例）
   - 是计算所有分类指标的基础

3. 精确率 / 召回率 / F1
   - Precision = TP / (TP + FP)：预测为正的里面有多少真正是正的
   - Recall = TP / (TP + FN)：真正是正的里面有多少被找到了
   - F1 = 2 × P × R / (P + R)：二者的调和平均
   - 类比：精确率=宁缺毋滥，召回率=宁滥勿缺

4. 交叉验证（Cross-Validation）
   - K-Fold：数据分K份，轮流做验证集
   - 优点：充分利用数据，评估更稳定
   - 在深度学习中：因为训练慢，通常用单次划分+验证集

5. 类别不平衡问题
   - 99%负例时，全预测负也有99%准确率（但毫无用处）
   - 解决方案：过采样、欠采样、加权损失、F1评估

【前置知识】
第1章第1-3节
"""

import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子，保证结果可复现
np.random.seed(42)

# =====================================================================
# 第1部分：数据集划分（Train / Validation / Test Split）
# =====================================================================
print("=" * 60)
print("第1部分：数据集划分")
print("=" * 60)


def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    从零实现训练集/验证集/测试集三路划分。

    原理：
    1. 先对索引做随机洗牌（shuffle），打乱数据顺序
    2. 按比例切分成三段

    参数:
        X: 特征矩阵, shape=(n_samples, n_features)
        y: 标签向量, shape=(n_samples,)
        train_ratio: 训练集占比
        val_ratio:   验证集占比
        test_ratio:  测试集占比

    返回:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "三个比例之和必须为1"

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)  # 原地打乱索引

    # 计算分割点
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # 按打乱后的索引切分
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


# --- 演示 ---
# 生成一个简单的二分类数据集
n_samples = 200
X_demo = np.random.randn(n_samples, 2)  # 200个样本, 2个特征
y_demo = (X_demo[:, 0] + X_demo[:, 1] > 0).astype(int)  # 简单线性边界

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
    X_demo, y_demo, 0.7, 0.15, 0.15
)

print(f"总样本数: {n_samples}")
print(f"训练集: {len(X_train)} ({len(X_train)/n_samples:.0%})")
print(f"验证集: {len(X_val)} ({len(X_val)/n_samples:.0%})")
print(f"测试集: {len(X_test)} ({len(X_test)/n_samples:.0%})")

# 检查各集合中正例比例是否大致相同（分层抽样的重要性）
for name, ys in [("训练集", y_train), ("验证集", y_val), ("测试集", y_test)]:
    print(f"  {name}正例比例: {ys.mean():.2%}")


# =====================================================================
# 第2部分：混淆矩阵（Confusion Matrix）
# =====================================================================
print("\n" + "=" * 60)
print("第2部分：混淆矩阵")
print("=" * 60)


def confusion_matrix(y_true, y_pred):
    """
    从零构建二分类混淆矩阵。

    混淆矩阵的含义：
                    预测=0      预测=1
        实际=0      TN          FP
        实际=1      FN          TP

    参数:
        y_true: 真实标签 (0/1)
        y_pred: 预测标签 (0/1)

    返回:
        2×2 的 numpy 数组 [[TN, FP], [FN, TP]]
    """
    # 逐个统计四种情况
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 真正例：实际正，预测正
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 假正例：实际负，预测正
    tn = np.sum((y_true == 0) & (y_pred == 0))  # 真负例：实际负，预测负
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 假负例：实际正，预测负

    return np.array([[tn, fp],
                     [fn, tp]])


def plot_confusion_matrix(cm, title="混淆矩阵"):
    """
    将混淆矩阵可视化为热力图。

    为什么要可视化？
    - 数字看久了会眼花，颜色深浅一目了然
    - 对角线颜色深 = 分类好，反对角线颜色深 = 分类差
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # 用 imshow 画热力图
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)

    # 在每个格子里写上数值
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{labels[i][j]}\n{cm[i][j]}",
                    ha='center', va='center', fontsize=14, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative (0)", "Positive (1)"])
    ax.set_yticklabels(["Negative (0)", "Positive (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[图已保存: confusion_matrix.png]")


# --- 演示 ---
# 模拟一个不完美的分类器
y_true_demo = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
y_pred_demo = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])

cm = confusion_matrix(y_true_demo, y_pred_demo)
print(f"混淆矩阵:\n{cm}")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
plot_confusion_matrix(cm)


# =====================================================================
# 第3部分：分类指标计算（Accuracy / Precision / Recall / F1）
# =====================================================================
print("\n" + "=" * 60)
print("第3部分：分类指标计算")
print("=" * 60)


def compute_metrics(y_true, y_pred):
    """
    从零计算四大分类指标。

    直觉理解：
    - Accuracy（准确率）: 总共猜对了多少？          → 全局视角
    - Precision（精确率）: 我说"是"的里面真是多少？  → 宁缺毋滥
    - Recall（召回率）:    真"是"的我抓到了多少？    → 宁滥勿缺
    - F1: 精确率和召回率的调和平均                  → 二者的平衡

    为什么用调和平均而不是算术平均？
    - 调和平均对极端值更敏感
    - 如果 P=1.0, R=0.01 → 算术平均=0.505（看起来还行）
    -                    → 调和平均=0.0198（暴露问题）
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # 准确率：(TP + TN) / 全部
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 精确率：TP / (TP + FP)，预测为正的里面有多少真正是正的
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 召回率：TP / (TP + FN)，真正是正的里面有多少被找到了
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1：精确率和召回率的调和平均
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


# --- 演示 ---
metrics = compute_metrics(y_true_demo, y_pred_demo)
print("分类指标:")
for name in ['accuracy', 'precision', 'recall', 'f1']:
    print(f"  {name:>10s}: {metrics[name]:.4f}")


# =====================================================================
# 第4部分：K-Fold 交叉验证
# =====================================================================
print("\n" + "=" * 60)
print("第4部分：K-Fold 交叉验证")
print("=" * 60)


def k_fold_split(n_samples, k=5):
    """
    从零实现 K-Fold 划分。

    原理：
    1. 将索引随机打乱
    2. 均匀切成 K 份
    3. 每次取 1 份做验证集，其余 K-1 份做训练集
    4. 轮 K 次，每个样本都恰好做过一次验证

    参数:
        n_samples: 总样本数
        k: 折数

    返回:
        生成器，每次 yield (train_indices, val_indices)
    """
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // k  # 每折的大小

    for i in range(k):
        # 第 i 折的起止位置
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n_samples

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        yield train_idx, val_idx


def simple_classifier_accuracy(X_train, y_train, X_val, y_val):
    """
    用一个简单的"最近质心分类器"做演示。

    原理：
    - 计算每个类别的中心点（质心）
    - 新样本离哪个质心近，就预测为那个类别
    - 虽然简单，但足以演示交叉验证的效果
    """
    # 计算每个类别的质心
    centroid_0 = X_train[y_train == 0].mean(axis=0)
    centroid_1 = X_train[y_train == 1].mean(axis=0)

    # 计算验证样本到两个质心的距离
    dist_0 = np.linalg.norm(X_val - centroid_0, axis=1)
    dist_1 = np.linalg.norm(X_val - centroid_1, axis=1)

    # 离哪个质心近就归为哪类
    y_pred = (dist_1 < dist_0).astype(int)

    accuracy = np.mean(y_pred == y_val)
    return accuracy


# --- 演示：对比不同 K 值的交叉验证 ---
# 生成一个有一定难度的二分类问题
np.random.seed(42)
n_cv = 300
X_cv = np.random.randn(n_cv, 2)
# 用非线性边界增加难度
y_cv = ((X_cv[:, 0] ** 2 + X_cv[:, 1] ** 2) < 1.5).astype(int)

print("不同 K 值下交叉验证的准确率统计：\n")
print(f"{'K':>4s} | {'均值':>8s} | {'标准差':>8s} | {'各折准确率'}")
print("-" * 60)

cv_results = {}
for k in [3, 5, 10]:
    np.random.seed(42)  # 每种 K 值用相同的种子
    fold_accs = []
    for train_idx, val_idx in k_fold_split(n_cv, k):
        acc = simple_classifier_accuracy(
            X_cv[train_idx], y_cv[train_idx],
            X_cv[val_idx], y_cv[val_idx]
        )
        fold_accs.append(acc)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    cv_results[k] = (mean_acc, std_acc)
    accs_str = ", ".join(f"{a:.3f}" for a in fold_accs)
    print(f"{k:>4d} | {mean_acc:>8.4f} | {std_acc:>8.4f} | [{accs_str}]")

# 可视化：K越大，方差越小（估计越稳定）
fig, ax = plt.subplots(figsize=(6, 4))
ks = list(cv_results.keys())
means = [cv_results[k][0] for k in ks]
stds = [cv_results[k][1] for k in ks]
ax.bar([str(k) for k in ks], means, yerr=stds, capsize=8,
       color=['#4C72B0', '#55A868', '#C44E52'], alpha=0.85)
ax.set_xlabel("K (fold count)")
ax.set_ylabel("Accuracy")
ax.set_title("K-Fold: K increases -> variance of estimate decreases")
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("kfold_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print(f"\n[图已保存: kfold_comparison.png]")
print("结论: K 越大，每折验证集越小但折数越多，估计的方差通常越小。")


# =====================================================================
# 第5部分：类别不平衡演示
# =====================================================================
print("\n" + "=" * 60)
print("第5部分：类别不平衡演示")
print("=" * 60)

# 构造一个极度不平衡的数据集：正例仅占 5%
np.random.seed(42)
n_imb = 1000
n_positive = 50   # 5% 正例
n_negative = 950  # 95% 负例

X_imb = np.random.randn(n_imb, 2)
y_imb = np.zeros(n_imb, dtype=int)
y_imb[:n_positive] = 1
# 正例的特征偏移一下，让它们有一定可分性
X_imb[:n_positive] += 1.5

# 打乱顺序
shuffle_idx = np.random.permutation(n_imb)
X_imb, y_imb = X_imb[shuffle_idx], y_imb[shuffle_idx]

print(f"数据集: {n_imb} 样本, 正例 {n_positive} ({n_positive/n_imb:.1%}), "
      f"负例 {n_negative} ({n_negative/n_imb:.1%})")

# --- 策略1: "偷懒"分类器——全预测为负例 ---
y_pred_lazy = np.zeros(n_imb, dtype=int)
metrics_lazy = compute_metrics(y_imb, y_pred_lazy)

# --- 策略2: 最近质心分类器（有实际分类能力） ---
centroid_pos = X_imb[y_imb == 1].mean(axis=0)
centroid_neg = X_imb[y_imb == 0].mean(axis=0)
dist_pos = np.linalg.norm(X_imb - centroid_pos, axis=1)
dist_neg = np.linalg.norm(X_imb - centroid_neg, axis=1)
y_pred_centroid = (dist_pos < dist_neg).astype(int)
metrics_centroid = compute_metrics(y_imb, y_pred_centroid)

print("\n对比两种分类器在不平衡数据上的表现：")
print(f"{'指标':>12s} | {'偷懒(全预测负)':>14s} | {'最近质心':>10s}")
print("-" * 48)
for name in ['accuracy', 'precision', 'recall', 'f1']:
    print(f"{name:>12s} | {metrics_lazy[name]:>14.4f} | "
          f"{metrics_centroid[name]:>10.4f}")

print("\n关键洞察:")
print("  偷懒分类器准确率高达 95%，但 Recall=0, F1=0 —— 完全无用！")
print("  最近质心分类器准确率可能稍低，但 F1 分数更高 —— 实际上更有用。")
print("  启示：类别不平衡时，不要看准确率，要看 F1（或 Precision/Recall）。")


# =====================================================================
# 第6部分：学习曲线（Learning Curve）
# =====================================================================
print("\n" + "=" * 60)
print("第6部分：学习曲线")
print("=" * 60)


def learning_curve(X_train, y_train, X_val, y_val, train_sizes):
    """
    绘制学习曲线：训练集大小 vs 训练/验证准确率。

    学习曲线能告诉我们什么？
    - 训练误差高 + 验证误差高 → 欠拟合（模型太简单）
    - 训练误差低 + 验证误差高 → 过拟合（模型太复杂、数据太少）
    - 两条线收敛且都低     → 模型和数据量都合适

    参数:
        X_train, y_train: 训练数据
        X_val, y_val:     验证数据
        train_sizes:      要尝试的训练集大小列表
    """
    train_accs = []
    val_accs = []

    for size in train_sizes:
        # 取前 size 个训练样本
        X_sub = X_train[:size]
        y_sub = y_train[:size]

        # 用最近质心分类器
        if np.sum(y_sub == 0) == 0 or np.sum(y_sub == 1) == 0:
            # 如果某个类别没有样本，跳过
            train_accs.append(np.nan)
            val_accs.append(np.nan)
            continue

        c0 = X_sub[y_sub == 0].mean(axis=0)
        c1 = X_sub[y_sub == 1].mean(axis=0)

        # 在训练子集上的准确率
        d0_tr = np.linalg.norm(X_sub - c0, axis=1)
        d1_tr = np.linalg.norm(X_sub - c1, axis=1)
        y_pred_tr = (d1_tr < d0_tr).astype(int)
        train_accs.append(np.mean(y_pred_tr == y_sub))

        # 在验证集上的准确率
        d0_val = np.linalg.norm(X_val - c0, axis=1)
        d1_val = np.linalg.norm(X_val - c1, axis=1)
        y_pred_val = (d1_val < d0_val).astype(int)
        val_accs.append(np.mean(y_pred_val == y_val))

    return train_accs, val_accs


# --- 生成数据并划分 ---
np.random.seed(42)
n_lc = 500
X_lc = np.random.randn(n_lc, 2)
y_lc = (X_lc[:, 0] * 0.8 + X_lc[:, 1] * 0.6 + np.random.randn(n_lc) * 0.3 > 0).astype(int)

# 80% 训练, 20% 验证
split = int(n_lc * 0.8)
X_lc_train, y_lc_train = X_lc[:split], y_lc[:split]
X_lc_val, y_lc_val = X_lc[split:], y_lc[split:]

# 从很少的样本逐步增加到全部
train_sizes = [5, 10, 20, 40, 80, 120, 160, 200, 280, 350, split]
train_accs, val_accs = learning_curve(
    X_lc_train, y_lc_train, X_lc_val, y_lc_val, train_sizes
)

# --- 可视化学习曲线 ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(train_sizes, train_accs, 'o-', label='Train Accuracy', color='#4C72B0')
ax.plot(train_sizes, val_accs, 's-', label='Validation Accuracy', color='#C44E52')
ax.fill_between(train_sizes, train_accs, val_accs,
                alpha=0.15, color='gray', label='Gap (overfitting indicator)')
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve: Diagnose Overfitting vs Underfitting")
ax.legend(loc='lower right')
ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)

# 添加注释说明
ax.annotate("gap large = overfitting",
            xy=(40, (train_accs[3] + val_accs[3]) / 2),
            xytext=(100, 0.55),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

plt.tight_layout()
plt.savefig("learning_curve.png", dpi=100, bbox_inches='tight')
plt.close()
print("[图已保存: learning_curve.png]")

print("\n学习曲线解读:")
print(f"  训练集={train_sizes[0]}时: 训练准确率={train_accs[0]:.3f}, "
      f"验证准确率={val_accs[0]:.3f} (差距大 → 过拟合)")
print(f"  训练集={train_sizes[-1]}时: 训练准确率={train_accs[-1]:.3f}, "
      f"验证准确率={val_accs[-1]:.3f} (差距小 → 收敛)")


# =====================================================================
# 第7部分：综合可视化 —— 所有指标一图看清
# =====================================================================
print("\n" + "=" * 60)
print("综合可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- 子图1: 混淆矩阵 ---
ax = axes[0]
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar(im, ax=ax, fraction=0.046)
labels_cm = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i][j]}",
                ha='center', va='center', fontsize=13, color=color,
                fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred 0", "Pred 1"])
ax.set_yticklabels(["True 0", "True 1"])
ax.set_title("Confusion Matrix")

# --- 子图2: 类别不平衡对比 ---
ax = axes[1]
metric_names = ['accuracy', 'precision', 'recall', 'f1']
x_pos = np.arange(len(metric_names))
width = 0.35
bars1 = ax.bar(x_pos - width / 2,
               [metrics_lazy[m] for m in metric_names],
               width, label='Lazy (all negative)', color='#C44E52', alpha=0.8)
bars2 = ax.bar(x_pos + width / 2,
               [metrics_centroid[m] for m in metric_names],
               width, label='Centroid classifier', color='#4C72B0', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Acc', 'Prec', 'Recall', 'F1'])
ax.set_ylim(0, 1.15)
ax.set_title("Imbalanced Data: Accuracy is Misleading")
ax.legend(fontsize=8, loc='upper right')
# 在柱子上方标数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{bar.get_height():.2f}', ha='center', fontsize=8)

# --- 子图3: 学习曲线 ---
ax = axes[2]
ax.plot(train_sizes, train_accs, 'o-', label='Train', color='#4C72B0')
ax.plot(train_sizes, val_accs, 's-', label='Validation', color='#C44E52')
ax.fill_between(train_sizes, train_accs, val_accs, alpha=0.12, color='gray')
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve")
ax.legend(fontsize=9)
ax.set_ylim(0.4, 1.05)
ax.grid(True, alpha=0.3)

plt.suptitle("Model Evaluation Overview", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("evaluation_overview.png", dpi=120, bbox_inches='tight')
plt.close()
print("[图已保存: evaluation_overview.png]")


# =====================================================================
# 思考题
# =====================================================================
print("\n" + "=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 为什么测试集只能在最后用一次？
   如果你在调模型的过程中反复用测试集评估，会发生什么？
   提示：想想"信息泄露"——你可能不自觉地让模型去适配测试集。

2. 在医疗诊断场景下（预测患者是否有癌症），你更关注 Precision
   还是 Recall？为什么？如果反过来，在垃圾邮件过滤场景下呢？
   提示：漏诊（FN）和误诊（FP），哪个后果更严重？

3. 假设你有一个100万样本的数据集，还有必要用 K-Fold 交叉验证吗？
   深度学习中通常怎么做？
   提示：考虑计算成本和单次划分的统计可靠性。

4. 学习曲线中，如果训练误差和验证误差都很高且差距不大，
   说明什么问题？应该怎么改进？
   提示：这是欠拟合，模型太简单或特征不够好。

5. 除了 F1，还有哪些处理类别不平衡的方法？
   在实际项目中，你会怎么组合使用它们？
   提示：过采样（SMOTE）、欠采样、代价敏感学习、阈值调整。
""")

print("=" * 60)
print("第4节完成！")
print("下一步：第5节将介绍决策树与集成方法，")
print("你将看到如何把多个弱分类器组合成强分类器。")
print("=" * 60)
