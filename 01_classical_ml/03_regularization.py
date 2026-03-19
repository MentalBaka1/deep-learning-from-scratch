"""
====================================================================
第1章 · 第3节 · 正则化与过拟合
====================================================================

【一句话总结】
正则化是防止模型"死记硬背"的技术——让模型学到真正的规律而非噪声。

【为什么深度学习需要这个？】
- 深度网络参数巨多（GPT-3有1750亿），极易过拟合
- L2正则化（Weight Decay）是训练大模型的标配
- Dropout、数据增强等正则化技术也源于同一思想
- 偏差-方差权衡是理解模型表现的基本框架

【核心概念】

1. 过拟合 vs 欠拟合
   - 过拟合：训练误差低，测试误差高（模型太复杂，记住了噪声）
   - 欠拟合：训练误差也高（模型太简单，学不到规律）
   - 类比：考试只做原题满分（过拟合），但换题就不会

2. 偏差-方差权衡（Bias-Variance Tradeoff）
   - 偏差：模型的系统性错误（太简单→高偏差）
   - 方差：模型对不同训练集的敏感度（太复杂→高方差）
   - 总误差 = 偏差² + 方差 + 不可约噪声
   - 最佳模型：偏差和方差都不太高

3. L1 正则化（Lasso）
   - 损失 + λΣ|w_i|
   - 效果：让一些权重变成精确的0（特征选择）
   - 直觉：鼓励稀疏解

4. L2 正则化（Ridge / Weight Decay）
   - 损失 + λΣw_i²
   - 效果：让权重保持较小（不会有极端大的权重）
   - 直觉：惩罚复杂模型，偏好简单模型
   - 在深度学习中叫 weight decay，几乎所有大模型都用

5. 正则化强度 λ
   - λ 太大：欠拟合（过度约束）
   - λ 太小：过拟合（约束不够）
   - 通常通过验证集选择最佳 λ

【前置知识】
第1章第1-2节
"""

import numpy as np
import matplotlib.pyplot as plt

# ===== 工具函数 =====
def true_fn(x):
    """真实函数：sin(2*pi*x)"""
    return np.sin(2 * np.pi * x)

def gen_data(n, noise=0.3, seed=None):
    """生成带噪声的数据"""
    if seed is not None:
        np.random.seed(seed)
    x = np.sort(np.random.uniform(0, 1, n))
    return x, true_fn(x) + np.random.randn(n) * noise

def poly_features(x, deg):
    """多项式特征 [1, x, x², ..., x^deg]"""
    return np.column_stack([x ** i for i in range(deg + 1)])

def mse(y, yp):
    """均方误差"""
    return np.mean((y - yp) ** 2)

def ridge_solve(X, y, lam):
    """Ridge闭式解: w = (X^T X + λI)^{-1} X^T y"""
    I = np.eye(X.shape[1])
    I[0, 0] = 0  # 不惩罚偏置
    return np.linalg.solve(X.T @ X + lam * I, X.T @ y)

def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    """坐标下降法求解Lasso: 逐个优化权重，软阈值操作产生稀疏解"""
    n, p = X.shape
    w = np.zeros(p)
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            r = y - X @ w + X[:, j] * w[j]  # 去掉第j个特征的残差
            rho = X[:, j] @ r                # rho_j = X_j^T * 残差
            z = X[:, j] @ X[:, j]            # z_j = ||X_j||^2
            if j == 0:
                w[j] = rho / z               # 偏置不惩罚
            else:
                w[j] = np.sign(rho) * max(abs(rho) - lam * n, 0) / z
        if np.linalg.norm(w - w_old) < tol:
            break
    return w

# ===== 演示1：过拟合演示 =====
def demo_overfitting():
    """高阶多项式拟合，展示训练/测试误差的差距"""
    print("=" * 60)
    print("演示1：过拟合 —— 多项式阶数越高，过拟合越严重")
    print("=" * 60)
    x_tr, y_tr = gen_data(15, seed=42)
    x_te, y_te = gen_data(100, seed=99)
    xp = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("过拟合演示：多项式阶数从低到高", fontsize=14, fontweight="bold")
    for ax, d in zip(axes, [1, 3, 5, 12]):
        Xtr, Xte, Xp = poly_features(x_tr, d), poly_features(x_te, d), poly_features(xp, d)
        w = np.linalg.lstsq(Xtr, y_tr, rcond=None)[0]
        tr_e, te_e = mse(y_tr, Xtr @ w), mse(y_te, Xte @ w)
        ax.scatter(x_tr, y_tr, c="steelblue", s=30, label="训练数据", zorder=3)
        ax.plot(xp, true_fn(xp), "g--", alpha=0.5, label="真实函数")
        ax.plot(xp, np.clip(Xp @ w, -2, 2), "r-", lw=2, label="拟合曲线")
        ax.set_title(f"阶数={d}\n训练MSE={tr_e:.3f}  测试MSE={te_e:.3f}", fontsize=10)
        ax.set_ylim(-2, 2); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("overfitting_demo.png", dpi=100, bbox_inches="tight"); plt.close()
    print("[已保存] overfitting_demo.png")
    print("观察：阶数=12训练误差极低，但曲线剧烈振荡 → 过拟合\n")

# ===== 演示2：偏差-方差可视化 =====
def demo_bias_variance():
    """多组随机数据集拟合同一模型，观察预测散布程度
    简单模型→高偏差低方差，复杂模型→低偏差高方差
    """
    print("=" * 60)
    print("演示2：偏差-方差权衡 —— 简单 vs 复杂模型")
    print("=" * 60)
    xp = np.linspace(0, 1, 200)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("偏差-方差：同一模型在不同数据集上的表现", fontsize=13, fontweight="bold")
    for ax, d in zip(axes, [1, 4, 12]):
        preds = []
        for i in range(20):  # 20组不同数据集
            xt, yt = gen_data(20, seed=i * 7 + 1)
            w = np.linalg.lstsq(poly_features(xt, d), yt, rcond=None)[0]
            yp = np.clip(poly_features(xp, d) @ w, -2, 2)
            preds.append(yp)
            ax.plot(xp, yp, "steelblue", alpha=0.15, lw=1)
        mean_p = np.mean(preds, axis=0)
        ax.plot(xp, true_fn(xp), "g--", lw=2, label="真实函数")
        ax.plot(xp, mean_p, "r-", lw=2, label="平均预测")
        bias2 = np.mean((mean_p - true_fn(xp)) ** 2)
        var = np.mean(np.var(preds, axis=0))
        ax.set_title(f"阶数={d}\n偏差²={bias2:.3f}  方差={var:.3f}", fontsize=10)
        ax.set_ylim(-2, 2); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("bias_variance.png", dpi=100, bbox_inches="tight"); plt.close()
    print("[已保存] bias_variance.png")
    print("观察：阶数=1偏差大，阶数=12方差大，阶数=4折中\n")

# ===== 演示3：L2 正则化实现 =====
def demo_l2_regularization():
    """Ridge回归：损失+λΣw²，让权重变小、曲线变平滑"""
    print("=" * 60)
    print("演示3：L2正则化（Ridge）—— 权重变小，曲线变平滑")
    print("=" * 60)
    x_tr, y_tr = gen_data(15, seed=42)
    x_te, y_te = gen_data(100, seed=99)
    xp = np.linspace(0, 1, 200)
    deg = 12  # 高阶多项式，没正则化会严重过拟合

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("L2正则化：λ从小到大（12阶多项式）", fontsize=13, fontweight="bold")
    for ax, lam in zip(axes, [0, 1e-6, 1e-3, 0.1]):
        Xtr, Xte, Xp = poly_features(x_tr, deg), poly_features(x_te, deg), poly_features(xp, deg)
        w = ridge_solve(Xtr, y_tr, lam) if lam > 0 else np.linalg.lstsq(Xtr, y_tr, rcond=None)[0]
        tr_e, te_e = mse(y_tr, Xtr @ w), mse(y_te, Xte @ w)
        ax.scatter(x_tr, y_tr, c="steelblue", s=30, zorder=3)
        ax.plot(xp, true_fn(xp), "g--", alpha=0.5, label="真实函数")
        ax.plot(xp, np.clip(Xp @ w, -2, 2), "r-", lw=2, label="拟合曲线")
        ax.set_title(f"λ={lam}\n训练={tr_e:.3f}  测试={te_e:.3f}", fontsize=10)
        ax.set_ylim(-2, 2); ax.legend(fontsize=8)
        print(f"  λ={lam:<8} | ||w||₂={np.linalg.norm(w):.4f} | 测试MSE={te_e:.4f}")
    plt.tight_layout()
    plt.savefig("l2_regularization.png", dpi=100, bbox_inches="tight"); plt.close()
    print("[已保存] l2_regularization.png\n")

# ===== 演示4：L1 正则化实现 =====
def demo_l1_regularization():
    """Lasso回归：损失+λΣ|w|，产生稀疏权重"""
    print("=" * 60)
    print("演示4：L1正则化（Lasso）—— 让权重变稀疏")
    print("=" * 60)
    x_tr, y_tr = gen_data(30, seed=42)
    Xtr = poly_features(x_tr, 10)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("L1正则化：权重稀疏程度随λ增大", fontsize=13, fontweight="bold")
    for ax, lam in zip(axes, [0, 1e-4, 1e-3, 0.01]):
        w = np.linalg.lstsq(Xtr, y_tr, rcond=None)[0] if lam == 0 else lasso_cd(Xtr, y_tr, lam)
        colors = ["tomato" if abs(wi) > 1e-8 else "lightgray" for wi in w]
        ax.bar(range(len(w)), w, color=colors, edgecolor="black", lw=0.5)
        ax.set_xlabel("权重编号"); ax.set_ylabel("权重值")
        nz = np.sum(np.abs(w) > 1e-8)
        ax.set_title(f"λ={lam}\n非零权重: {nz}/{len(w)}", fontsize=10)
        ax.axhline(y=0, color="black", lw=0.5)
        print(f"  λ={lam:<8} | 非零权重: {nz}/{len(w)} | ||w||₁={np.sum(np.abs(w)):.4f}")
    plt.tight_layout()
    plt.savefig("l1_regularization.png", dpi=100, bbox_inches="tight"); plt.close()
    print("[已保存] l1_regularization.png")
    print("观察：λ增大 → 越来越多权重变0（灰色柱子）→ 特征选择\n")

# ===== 演示5：L1 vs L2 对比 =====
def demo_l1_vs_l2():
    """并排对比：L1（菱形约束→稀疏解） vs L2（圆形约束→均匀缩小）"""
    print("=" * 60)
    print("演示5：L1 vs L2 —— 权重分布对比")
    print("=" * 60)
    np.random.seed(42)
    x_tr, y_tr = gen_data(30, seed=42)
    Xtr = poly_features(x_tr, 10)
    nf, lam = Xtr.shape[1], 1e-3

    w_l2 = ridge_solve(Xtr, y_tr, lam)
    w_l1 = lasso_cd(Xtr, y_tr, lam)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"L1 vs L2 权重对比（λ={lam}，10阶多项式）", fontsize=13, fontweight="bold")
    # L1权重（跳过偏置）
    c1 = ["tomato" if abs(v) > 1e-8 else "lightgray" for v in w_l1[1:]]
    ax1.bar(range(1, nf), w_l1[1:], color=c1, edgecolor="black", lw=0.5)
    nz = np.sum(np.abs(w_l1[1:]) > 1e-8)
    ax1.set_title(f"L1（Lasso）— 非零权重: {nz}/{nf-1}", fontsize=11)
    ax1.set_xlabel("特征编号"); ax1.set_ylabel("权重值")
    ax1.axhline(y=0, color="black", lw=0.5)
    # L2权重
    ax2.bar(range(1, nf), w_l2[1:], color="steelblue", edgecolor="black", lw=0.5)
    ax2.set_title("L2（Ridge）— 所有权重非零但较小", fontsize=11)
    ax2.set_xlabel("特征编号"); ax2.set_ylabel("权重值")
    ax2.axhline(y=0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig("l1_vs_l2.png", dpi=100, bbox_inches="tight"); plt.close()
    print("[已保存] l1_vs_l2.png")
    print(f"  L1: {nz}/{nf-1} 个非零权重 | L2: {nf-1}/{nf-1} 个非零权重")
    print("结论：L1自动特征选择（稀疏），L2均匀缩小所有权重\n")

# ===== 演示6：正则化强度选择 =====
def demo_lambda_selection():
    """扫描不同λ值，画训练/测试误差曲线——经典U型曲线"""
    print("=" * 60)
    print("演示6：正则化强度选择 —— 寻找最优 λ")
    print("=" * 60)
    x_tr, y_tr = gen_data(20, seed=42)
    x_te, y_te = gen_data(200, seed=99)
    Xtr, Xte = poly_features(x_tr, 10), poly_features(x_te, 10)

    lambdas = np.logspace(-8, 2, 60)  # 对数间隔扫描
    tr_errs = [mse(y_tr, Xtr @ ridge_solve(Xtr, y_tr, l)) for l in lambdas]
    te_errs = [mse(y_te, Xte @ ridge_solve(Xtr, y_tr, l)) for l in lambdas]
    best_i = np.argmin(te_errs)
    best_lam, best_err = lambdas[best_i], te_errs[best_i]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(lambdas, tr_errs, "b-", lw=2, label="训练误差")
    ax.semilogx(lambdas, te_errs, "r-", lw=2, label="测试误差")
    ax.axvline(x=best_lam, color="green", ls="--", lw=1.5, label=f"最优λ={best_lam:.2e}")
    ax.scatter([best_lam], [best_err], c="green", s=100, zorder=5)
    ax.annotate("过拟合区 →", xy=(lambdas[3], te_errs[3]), fontsize=9, color="darkred")
    ax.annotate("← 欠拟合区", xy=(lambdas[-8], te_errs[-8]), fontsize=9, color="darkred")
    ax.set_xlabel("正则化强度 λ（对数坐标）", fontsize=11)
    ax.set_ylabel("均方误差 (MSE)", fontsize=11)
    ax.set_title("训练/测试误差 vs λ（L2正则化，10阶多项式）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.set_ylim(0, min(max(te_errs) * 1.2, 5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lambda_selection.png", dpi=100, bbox_inches="tight"); plt.close()
    print(f"[已保存] lambda_selection.png")
    print(f"  最优λ={best_lam:.2e}，测试MSE={best_err:.4f} | 无正则化测试MSE={te_errs[0]:.4f}")
    print("观察：经典U型曲线——λ太小过拟合，太大欠拟合\n")

# ===== 思考题 =====
def print_questions():
    """本节思考题"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                        本节思考题                           ║
╠══════════════════════════════════════════════════════════════╣
║  1. 为什么L1能产生稀疏解（权重精确为0），而L2不能？          ║
║     提示：画出 |w| 和 w² 在 w=0 附近的梯度。                ║
║  2. 深度学习中为何更常用L2（Weight Decay）而非L1？           ║
║     提示：想想稀疏权重对梯度传播的影响。                     ║
║  3. 训练集非常大（百万级）时，还需要正则化吗？               ║
║     提示：数据量增大相当于自然降低了方差。                   ║
║  4. 早停（Early Stopping）为什么也是一种正则化？             ║
║     提示：训练轮数越多，权重越大→模型越复杂。               ║
║  5. Dropout的正则化效果可以用偏差-方差框架解释吗？           ║
║     提示：Dropout ≈ 训练了很多"子网络"的集成。              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ===== 主程序 =====
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  第1章 · 第3节 · 正则化与过拟合")
    print("=" * 60 + "\n")

    demo_overfitting()           # 演示1：过拟合现象
    demo_bias_variance()         # 演示2：偏差-方差权衡
    demo_l2_regularization()     # 演示3：L2正则化
    demo_l1_regularization()     # 演示4：L1正则化
    demo_l1_vs_l2()              # 演示5：L1 vs L2 对比
    demo_lambda_selection()      # 演示6：最优λ选择
    print_questions()            # 思考题

    print("=" * 60)
    print("本节总结：")
    print("  1. 过拟合 = 训练好、测试差 → 模型记住了噪声")
    print("  2. 偏差-方差权衡 → 模型复杂度的甜蜜点")
    print("  3. L2正则化 → 权重变小，曲线平滑（深度学习标配）")
    print("  4. L1正则化 → 权重变稀疏，自动特征选择")
    print("  5. λ 选择 → 通过验证集找到U型曲线的最低点")
    print("=" * 60)
    print("\n下一节预告：第4节 · 梯度下降法 —— 深度学习的核心优化算法\n")
