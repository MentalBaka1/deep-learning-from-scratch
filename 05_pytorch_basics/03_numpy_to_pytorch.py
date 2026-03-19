"""
====================================================================
第5章 · 第3节 · 从 NumPy 到 PyTorch：代码对照
====================================================================

【一句话总结】
用 PyTorch 复现第2章的 MLP，逐行对比 NumPy vs PyTorch 代码——
你会发现 PyTorch 就是"自动帮你写 backward 的 NumPy"。

【为什么需要这个对照？】
- 你已经用 NumPy 手写了完整的 MLP + 反向传播（第2章）
- 现在用 PyTorch 实现相同功能，看看代码差异
- 理解 PyTorch 帮你省掉了哪些工作（主要是反向传播）
- 建立信心：你已经理解了 PyTorch 内部在做什么

【核心对照】

NumPy 版本你需要自己写的：         PyTorch 自动处理的：
├── 前向传播 forward()             ├── 前向传播 forward() （你仍然要写）
├── 手动求梯度 ∂L/∂W              ├── 自动求梯度 loss.backward()
├── 手动更新 W -= lr * dW         ├── optimizer.step()
├── 保存中间变量用于反向传播       ├── 自动保存（计算图）
└── 数值梯度验证                   └── 不需要（autograd保证正确）

结论：PyTorch = NumPy前向传播 + 自动反向传播 + GPU加速

【前置知识】
第2章第3节 - MLP反向传播（NumPy版），第5章第1-2节
"""

import numpy as np
import torch
import torch.nn as nn
import time

np.random.seed(42)
torch.manual_seed(42)


# ====================================================================
# 第一部分：准备共享数据集
# ====================================================================
# 用同一份数据训练 NumPy 和 PyTorch 两个 MLP，公平对比收敛结果。
# 任务：二分类（环形数据，2维输入 → 1维输出），与第2章相同。

print("=" * 60)
print("第一部分：准备共享数据集")
print("=" * 60)

n_samples = 200
noise = 0.2

# 类别0：内圈 (r≈1.0)  |  类别1：外圈 (r≈2.5)
r0 = np.random.randn(n_samples // 2) * noise + 1.0
t0 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
X0 = np.stack([r0 * np.cos(t0), r0 * np.sin(t0)], axis=1)

r1 = np.random.randn(n_samples // 2) * noise + 2.5
t1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
X1 = np.stack([r1 * np.cos(t1), r1 * np.sin(t1)], axis=1)

X_np = np.vstack([X0, X1]).astype(np.float32)                     # (200, 2)
y_np = np.concatenate([np.zeros(n_samples // 2),
                       np.ones(n_samples // 2)]).astype(np.float32).reshape(-1, 1)

perm = np.random.permutation(n_samples)
X_np, y_np = X_np[perm], y_np[perm]
print(f"  数据集: X{X_np.shape}, y{y_np.shape}")


# ====================================================================
# 第二部分：NumPy 版 MLP（回顾第2章）
# ====================================================================
# 第2章手写 MLP 的精简版。你需要自己实现 forward + backward + update。
# 网络结构：2 → 16 → 1（ReLU + Sigmoid）

print("\n" + "=" * 60)
print("第二部分：NumPy 版 MLP（回顾第2章）")
print("=" * 60)


class NumpyMLP:
    """纯 NumPy 实现的两层 MLP。需要手写 forward/backward/update 三个方法。"""

    def __init__(self, in_d, hid_d, out_d):
        """He初始化权重，零初始化偏置"""
        self.W1 = np.random.randn(in_d, hid_d).astype(np.float32) * np.sqrt(2.0 / in_d)
        self.b1 = np.zeros((1, hid_d), dtype=np.float32)
        self.W2 = np.random.randn(hid_d, out_d).astype(np.float32) * np.sqrt(2.0 / hid_d)
        self.b2 = np.zeros((1, out_d), dtype=np.float32)

    def forward(self, X):
        """前向传播——注意：必须手动保存中间变量给 backward 用"""
        self.X = X                                         # ← 手动保存
        self.z1 = X @ self.W1 + self.b1                    # 线性
        self.a1 = np.maximum(0, self.z1)                   # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2              # 线性
        self.a2 = 1.0 / (1.0 + np.exp(-self.z2))           # Sigmoid
        return self.a2

    def backward(self, y):
        """反向传播——这就是 PyTorch 帮你省掉的全部工作！"""
        m = y.shape[0]
        # 输出层：BCE+Sigmoid 联合梯度
        dz2 = (self.a2 - y) / m
        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0, keepdims=True)
        # 隐藏层：链式法则逐层回传
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(np.float32)      # ReLU 导数
        self.dW1 = self.X.T @ dz1
        self.db1 = np.sum(dz1, axis=0, keepdims=True)

    def update(self, lr):
        """手动逐个更新参数"""
        self.W1 -= lr * self.dW1;  self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2;  self.b2 -= lr * self.db2


def bce_loss_numpy(y_pred, y_true):
    """二元交叉熵损失（NumPy版）"""
    eps = 1e-7
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))


print("  NumpyMLP 定义完成 —— 需要手写: forward(), backward(), update()")


# ====================================================================
# 第三部分：PyTorch 版 MLP
# ====================================================================
# 用 nn.Module 实现同样的网络。关键：只需写 forward()！

print("\n" + "=" * 60)
print("第三部分：PyTorch 版 MLP")
print("=" * 60)


class PyTorchMLP(nn.Module):
    """
    PyTorch 实现的两层 MLP。

    你需要写的：__init__() + forward()
    你不需要写的：backward()（autograd）、update()（optimizer.step()）
    """

    def __init__(self, in_d, hid_d, out_d):
        super().__init__()
        self.layer1 = nn.Linear(in_d, hid_d)   # 自动创建 W 和 b
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hid_d, out_d)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播——不需要手动保存中间变量，计算图自动记录"""
        x = self.layer1(x)      # 对应 X @ W1 + b1
        x = self.relu(x)        # 对应 np.maximum(0, z1)
        x = self.layer2(x)      # 对应 a1 @ W2 + b2
        x = self.sigmoid(x)     # 对应 1/(1+exp(-z2))
        return x


print("  PyTorchMLP 定义完成 —— 只需写 forward()，无需 backward()！")


# ====================================================================
# 第四部分：逐步对照——看 PyTorch 省掉了什么
# ====================================================================

print("\n" + "=" * 60)
print("第四部分：逐步对照")
print("=" * 60)

print("""
  步骤          NumPy 版                     PyTorch 版
  ──────────    ─────────────────────         ─────────────────────
  定义权重      W1 = randn(...) * scale       nn.Linear(in, out)  （自动创建）
  前向传播      z1 = X @ W1 + b1              x = self.layer1(x)  （自动记录）
  计算损失      loss = -mean(y*log(p)+...)    loss = criterion(pred, y)
  反向传播      dz2=(a2-y)/m; dW2=a1.T@dz2   loss.backward()     （一行！）
  （差异最大）  da1=dz2@W2.T; dz1=da1*(z1>0)
                dW1=X.T@dz1 ...每层手推！
  参数更新      W1 -= lr*dW1; b1 -= lr*db1    optimizer.step()    （一行！）
  梯度清零      （不需要，每次覆盖）          optimizer.zero_grad()
""")


# ====================================================================
# 第五部分：训练循环对比
# ====================================================================

print("=" * 60)
print("第五部分：训练循环对比")
print("=" * 60)

# 超参数（两个版本完全相同）
input_dim, hidden_dim, output_dim = 2, 16, 1
lr, epochs = 0.5, 200

# ─── NumPy 训练 ───
print("\n--- NumPy 训练循环 ---")
np_model = NumpyMLP(input_dim, hidden_dim, output_dim)
np_losses = []
t0 = time.time()
for epoch in range(epochs):
    y_pred = np_model.forward(X_np)           # 1. 前向传播
    np_losses.append(bce_loss_numpy(y_pred, y_np))  # 2. 计算损失
    np_model.backward(y_np)                   # 3. 反向传播（手动！12行代码）
    np_model.update(lr)                       # 4. 参数更新（手动！4行代码）
np_time = time.time() - t0

np_preds = (np_model.forward(X_np) > 0.5).astype(np.float32)
np_acc = np.mean(np_preds == y_np)
print(f"  最终损失: {np_losses[-1]:.4f} | 精度: {np_acc:.2%} | 耗时: {np_time:.4f}s")

# ─── PyTorch 训练 ───
print("\n--- PyTorch 训练循环 ---")
X_pt = torch.from_numpy(X_np)                # NumPy → Tensor（零拷贝）
y_pt = torch.from_numpy(y_np)
pt_model = PyTorchMLP(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()                      # 损失函数（一行定义）
optimizer = torch.optim.SGD(pt_model.parameters(), lr=lr)  # 优化器（一行定义）
pt_losses = []

t0 = time.time()
for epoch in range(epochs):
    y_pred = pt_model(X_pt)                   # 1. 前向传播
    loss = criterion(y_pred, y_pt)            # 2. 计算损失
    pt_losses.append(loss.item())
    optimizer.zero_grad()                     # 3a. 清零梯度（PyTorch特有）
    loss.backward()                           # 3b. 反向传播（一行！自动！）
    optimizer.step()                          # 4. 参数更新（一行！自动！）
pt_time = time.time() - t0

with torch.no_grad():
    pt_preds = (pt_model(X_pt) > 0.5).float()
    pt_acc = (pt_preds == y_pt).float().mean().item()
print(f"  最终损失: {pt_losses[-1]:.4f} | 精度: {pt_acc:.2%} | 耗时: {pt_time:.4f}s")


# ====================================================================
# 第六部分：结果验证
# ====================================================================
# 两个版本用相同数据和超参数，虽然初始权重不同，但都应收敛。

print("\n" + "=" * 60)
print("第六部分：结果验证")
print("=" * 60)

print(f"\n  {'指标':<10} {'NumPy':>10} {'PyTorch':>10}")
print(f"  {'─' * 32}")
print(f"  {'最终损失':<10} {np_losses[-1]:>10.4f} {pt_losses[-1]:>10.4f}")
print(f"  {'最终精度':<10} {np_acc:>10.2%} {pt_acc:>10.2%}")
print(f"  {'训练耗时':<10} {np_time:>9.4f}s {pt_time:>9.4f}s")

np_ok = np_losses[-1] < np_losses[0]
pt_ok = pt_losses[-1] < pt_losses[0]
print(f"\n  NumPy 收敛? {'是' if np_ok else '否'} ({np_losses[0]:.4f} → {np_losses[-1]:.4f})")
print(f"  PyTorch 收敛? {'是' if pt_ok else '否'} ({pt_losses[0]:.4f} → {pt_losses[-1]:.4f})")
print("  结论：两个版本都能收敛 —— PyTorch 做的和我们手写的一样！")


# ====================================================================
# 第七部分：代码量对比
# ====================================================================

print("\n" + "=" * 60)
print("第七部分：代码量对比")
print("=" * 60)

print("""
  功能         NumPy    PyTorch       关键发现：
  ────────     ─────    ───────       - 前向传播代码量相同
  定义网络     ~15行    ~10行         - 反向传播 12行→1行（最大节省！）
  前向传播      ~5行     ~5行 ←相同   - 网络越深省越多（50层→手写150+行）
  反向传播     ~12行      1行!        - 省掉的更是调试时间
  参数更新      ~4行      1行!        - 因为你写过 backward，你知道
  训练循环      ~8行     ~8行           loss.backward() 内部在做什么
  ────────     ─────    ───────       - "会用"和"理解"的区别就在于此
  总计         ~44行    ~25行
""")


# ====================================================================
# 第八部分：GPU 加速体验
# ====================================================================
# PyTorch 另一大优势：几行代码即可使用 GPU。
# NumPy 想跑 GPU 需要 CuPy 或手写 CUDA C。

print("=" * 60)
print("第八部分：GPU 加速体验")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  当前设备: {device}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 只需三步：模型→GPU，数据→GPU，训练代码不变
    gpu_model = PyTorchMLP(input_dim, hidden_dim, output_dim).to(device)
    X_gpu, y_gpu = X_pt.to(device), y_pt.to(device)
    gpu_opt = torch.optim.SGD(gpu_model.parameters(), lr=lr)
    gpu_crit = nn.BCELoss()

    _ = gpu_model(X_gpu)  # CUDA 预热
    t0 = time.time()
    for _ in range(epochs):
        pred = gpu_model(X_gpu)
        loss = gpu_crit(pred, y_gpu)
        gpu_opt.zero_grad(); loss.backward(); gpu_opt.step()
    torch.cuda.synchronize()
    gpu_time = time.time() - t0

    print(f"  GPU 训练: {gpu_time:.4f}s  |  CPU 训练: {pt_time:.4f}s")
    if gpu_time < pt_time:
        print(f"  加速比: {pt_time / gpu_time:.2f}x")
    else:
        print("  小数据集 GPU 未体现优势（传输开销 > 计算收益）")
        print("  数据量大、模型更深时，GPU 优势才会显现")
else:
    print("  未检测到 GPU。迁移只需两行：")
    print("    model = model.to('cuda')              # 模型 → GPU")
    print("    X, y = X.to('cuda'), y.to('cuda')     # 数据 → GPU")
    print("  其余训练代码完全不变！对比 NumPy（需 CuPy/CUDA C）简单太多。")


# ====================================================================
# 第九部分：总结与思考题
# ====================================================================

print("\n" + "=" * 60)
print("第九部分：总结与思考题")
print("=" * 60)

print("""
  核心结论：PyTorch = NumPy前向传播 + 自动反向传播 + GPU加速
  • 你仍然需要理解前向传播（定义网络结构）
  • 你不再需要手推梯度公式（autograd 自动完成）
  • 你只需 .to(device) 就能使用 GPU
  • 因为你写过 backward，你比只会调 API 的人多一层理解

  【思考题】
  1. loss.backward() 和手写 backward() 在做同一件事。PyTorch 如何知道
     对哪些参数求梯度？（提示：requires_grad=True 和计算图）
  2. 为什么 PyTorch 需要 optimizer.zero_grad()？不调用会怎样？
     NumPy 版为什么不需要？（提示：累积梯度 vs 每次覆盖旧值）
  3. model.to(device) 背后发生了什么？
     （提示：递归遍历所有 nn.Parameter，逐个搬到 GPU 显存）
  4. 50层网络，NumPy 的 backward() 要写多少行？PyTorch 呢？
     （提示：NumPy 每层3-5行=150-250行；PyTorch 始终1行）
  5. [进阶] PyTorch "动态计算图" vs TensorFlow 1.x "静态计算图"，
     动态图的优点？哪些场景特别有用？（提示：if/for 控制流、调试）
""")

print("=" * 60)
print("第5章 · 第3节 完成！")
print("下一节：第5章 · 第4节 —— 用 PyTorch 实现完整的训练流程")
print("=" * 60)
