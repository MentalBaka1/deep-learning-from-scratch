"""
====================================================================
第5章 · 第1节 · Tensor 与自动微分
====================================================================

【一句话总结】
PyTorch 的 Tensor 就是"带自动微分的 NumPy 数组"——你只需写前向传播，
PyTorch 自动帮你计算所有梯度。

【为什么需要 PyTorch？】
- 前4章我们用 NumPy 手写了所有梯度计算——这帮助你理解原理
- 但实际项目中手写梯度不现实（Transformer 有上亿参数）
- PyTorch 的 autograd 自动完成反向传播，让你专注于模型设计
- GPU 加速让训练速度提升 10-100 倍

【核心概念】

1. Tensor vs ndarray
   - Tensor 几乎等于 ndarray：相同的索引、切片、广播
   - 额外能力：自动微分（requires_grad=True）、GPU 支持
   - 互转：torch.from_numpy(arr) ↔ tensor.numpy()

2. 计算图与自动微分
   - requires_grad=True：告诉 PyTorch "请跟踪这个变量的计算"
   - 前向传播时 PyTorch 自动构建计算图
   - .backward()：自动沿计算图反向计算所有梯度
   - .grad：获取计算好的梯度

3. 叶节点与非叶节点
   - 叶节点：直接创建的 tensor（如权重 w）
   - 非叶节点：通过计算得到的 tensor（如 y = w*x + b）
   - 只有叶节点的 .grad 会被保留
   - 非叶节点的梯度用完即丢（节省内存）

4. 梯度累积
   - .backward() 后梯度会累积到 .grad 上（不是替换！）
   - 每次更新前必须 .grad.zero_() 清零
   - 累积特性在梯度累积（大batch模拟）中反而有用

5. with torch.no_grad()
   - 推理时不需要梯度，用 no_grad 节省内存和计算
   - 更新参数时也要用（否则更新操作也会被记录到计算图）

【前置知识】
第0章第4节 - 链式法则与计算图，第2章第3节 - 反向传播
"""

import torch
import numpy as np
import time


# ====================================================================
# 第1部分：Tensor 基础
# ====================================================================
def part1_tensor_basics():
    """Tensor 基础：创建、运算、与 NumPy 对比。"""
    print("=" * 60)
    print("第1部分：Tensor 基础")
    print("=" * 60)

    # --- 创建 Tensor ---
    t1 = torch.tensor([1, 2, 3, 4])
    t2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(f"从列表:   {t1}  dtype={t1.dtype}")
    print(f"float32:  {t2}  dtype={t2.dtype}")

    # 常用初始化（类比 NumPy）
    print(f"\nzeros(2,3):\n{torch.zeros(2, 3)}")
    print(f"randn(2,3):\n{torch.randn(2, 3)}")
    print(f"arange(0,10,2): {torch.arange(0, 10, 2)}")

    # --- 基本运算 ---
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    print(f"\na + b = {a + b}")             # 逐元素加
    print(f"a * b = {a * b}")               # 逐元素乘（非矩阵乘）
    print(f"a @ b = {a @ b}")               # 点积
    print(f"a.sum()={a.sum()}, a.mean()={a.mean()}")

    # --- 形状操作 ---
    M = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"\nreshape(3,4):\n{M}")
    print(f"转置: {M.shape} → {M.T.shape}")

    # --- 对照表 ---
    print("\n--- PyTorch vs NumPy ---")
    for op, np_f, pt_f in [
        ("零矩阵", "np.zeros((2,3))",     "torch.zeros(2,3)"),
        ("正态",   "np.random.randn(2,3)", "torch.randn(2,3)"),
        ("矩阵乘", "A @ B",               "A @ B / torch.mm"),
        ("求和",   "arr.sum(axis=0)",      "tensor.sum(dim=0)"),
    ]:
        print(f"  {op:<6s}  {np_f:<25s}  {pt_f}")


# ====================================================================
# 第2部分：Tensor ↔ NumPy 互转
# ====================================================================
def part2_numpy_conversion():
    """互转演示，重点：共享内存陷阱。"""
    print("\n" + "=" * 60)
    print("第2部分：Tensor ↔ NumPy 互转")
    print("=" * 60)

    arr = np.array([1.0, 2.0, 3.0])
    t = torch.from_numpy(arr)
    print(f"NumPy→Tensor: {arr} → {t}")
    print(f"Tensor→NumPy: {torch.tensor([4.0, 5.0]).numpy()}")

    # --- 共享内存陷阱 ---
    print("\n--- 共享内存陷阱 ---")
    arr_s = np.array([10.0, 20.0, 30.0])
    t_s = torch.from_numpy(arr_s)
    arr_s[0] = 999.0
    print(f"修改 NumPy[0]=999 → Tensor 也变: {t_s}")
    t_s[2] = 888.0
    print(f"修改 Tensor[2]=888 → NumPy 也变: {arr_s}")

    # --- clone() 断开共享 ---
    arr2 = np.array([1.0, 2.0, 3.0])
    t2 = torch.from_numpy(arr2).clone()
    arr2[0] = 999.0
    print(f"clone() 后修改 NumPy → Tensor 不变: {t2}")


# ====================================================================
# 第3部分：自动微分
# ====================================================================
def part3_autograd():
    """自动微分：requires_grad → backward() → .grad"""
    print("\n" + "=" * 60)
    print("第3部分：自动微分")
    print("=" * 60)

    # y = x²
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2
    y.backward()
    print(f"\ny=x²: x={x.item()}, dy/dx=2x={x.grad.item():.1f}")

    # y = 3x²+2x+1
    x = torch.tensor(2.0, requires_grad=True)
    y = 3 * x**2 + 2 * x + 1
    y.backward()
    print(f"y=3x²+2x+1: x={x.item()}, dy/dx=6x+2={x.grad.item():.1f}")

    # 多变量: z = x²y + y³
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = x**2 * y + y**3
    z.backward()
    print(f"z=x²y+y³: dz/dx=2xy={x.grad.item():.1f}, dz/dy=x²+3y²={y.grad.item():.1f}")

    # --- 叶节点 vs 非叶节点 ---
    print("\n--- 叶节点 vs 非叶节点 ---")
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    out = (w * 3.0 + b) ** 2
    out.backward()
    print(f"w(叶): is_leaf={w.is_leaf}, grad={w.grad.item():.1f}")
    print(f"out(非叶): is_leaf={out.is_leaf}, grad={out.grad}")
    print("→ 非叶节点 grad=None，用完即丢节省内存")


# ====================================================================
# 第4部分：计算图可视化
# ====================================================================
def part4_computation_graph():
    """z=(a+b)*(b+1)，前向值+反向梯度，对比第0章手动计算。"""
    print("\n" + "=" * 60)
    print("第4部分：计算图 —— z = (a+b)*(b+1)")
    print("=" * 60)

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = a + b       # c=5
    d = b + 1       # d=4
    z = c * d        # z=20

    print(f"\n前向: a=2, b=3 → c=a+b=5, d=b+1=4, z=c*d=20")

    z.backward()
    print(f"PyTorch: dz/da={a.grad.item():.0f}, dz/db={b.grad.item():.0f}")

    # 手动反向传播（第0章方法）
    dz_dc = d.item()   # 乘法门交换：4
    dz_dd = c.item()   # 乘法门交换：5
    dz_da = 1.0 * dz_dc                     # 加法门传递：4
    dz_db = 1.0 * dz_dc + 1.0 * dz_dd       # 两条路径累加：4+5=9

    print(f"手动:    dz/da={dz_da:.0f}, dz/db={dz_db:.0f}={dz_dc:.0f}(经c)+{dz_dd:.0f}(经d)")
    print(f"完全一致！autograd 自动完成了第0章的手写工作。")

    # grad_fn 揭示计算图
    print(f"\ngrad_fn: z={z.grad_fn}, c={c.grad_fn}, a={a.grad_fn}(叶=None)")


# ====================================================================
# 第5部分：梯度累积陷阱
# ====================================================================
def part5_gradient_accumulation():
    """演示忘记 zero_grad 的 bug，修复方法，以及累积的正当用途。"""
    print("\n" + "=" * 60)
    print("第5部分：梯度累积陷阱")
    print("=" * 60)

    # --- Bug ---
    print("\n--- Bug：忘记 zero_grad ---")
    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        (w * 3.0).backward()
        print(f"  第{i+1}次: grad={w.grad.item():.0f}（期望3，累加了{i+1}次）")

    # --- 修复 ---
    print("\n--- 修复：每次先清零 ---")
    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        if w.grad is not None:
            w.grad.zero_()
        (w * 3.0).backward()
        print(f"  第{i+1}次: grad={w.grad.item():.0f} ✓")

    # --- 正当用途 ---
    print("\n--- 正当用途：累积模拟大 batch ---")
    w = torch.tensor(1.0, requires_grad=True)
    batches = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    if w.grad is not None:
        w.grad.zero_()
    for mb in batches:
        (w * mb).sum().backward()
        print(f"  batch {mb.tolist()}: 累积 grad={w.grad.item():.0f}")
    w.grad /= len(batches)
    print(f"  平均 grad={w.grad.item():.1f}（等效全量 batch）")


# ====================================================================
# 第6部分：手动 vs 自动梯度对比
# ====================================================================
def part6_manual_vs_auto():
    """2层网络，NumPy 手算 vs PyTorch autograd，验证一致性。"""
    print("\n" + "=" * 60)
    print("第6部分：手动 vs 自动梯度对比")
    print("=" * 60)

    x_val, y_true = 1.5, 0.8
    w1v, b1v, w2v, b2v = 0.5, 0.1, -0.3, 0.2

    # ===== NumPy 手动 =====
    z1 = w1v * x_val + b1v
    a1 = 1.0 / (1.0 + np.exp(-z1))
    z2 = w2v * a1 + b2v
    loss_np = 0.5 * (z2 - y_true) ** 2

    dL_dz2 = z2 - y_true
    dL_dw2 = dL_dz2 * a1;      dL_db2 = dL_dz2
    dL_dz1 = dL_dz2 * w2v * a1 * (1 - a1)
    dL_dw1 = dL_dz1 * x_val;   dL_db1 = dL_dz1

    # ===== PyTorch autograd =====
    w1 = torch.tensor(w1v, requires_grad=True)
    b1 = torch.tensor(b1v, requires_grad=True)
    w2 = torch.tensor(w2v, requires_grad=True)
    b2 = torch.tensor(b2v, requires_grad=True)
    x = torch.tensor(x_val)

    loss_t = 0.5 * (w2 * torch.sigmoid(w1 * x + b1) + b2 - y_true) ** 2
    loss_t.backward()

    # ===== 对比 =====
    print(f"\n  loss: 手动={loss_np:.8f}, 自动={loss_t.item():.8f}")
    all_ok = True
    for name, manual, auto in [("dL/dw1", dL_dw1, w1.grad.item()),
                                ("dL/db1", dL_db1, b1.grad.item()),
                                ("dL/dw2", dL_dw2, w2.grad.item()),
                                ("dL/db2", dL_db2, b2.grad.item())]:
        diff = abs(manual - auto)
        ok = diff < 1e-7;  all_ok = all_ok and ok
        print(f"  {name}: 手动={manual:+.8f}, 自动={auto:+.8f}, "
              f"误差={diff:.2e} [{'PASS' if ok else 'FAIL'}]")

    if all_ok:
        print("\n  全部 PASS！手算需要推导每步，autograd 只需写前向——这就是 PyTorch。")


# ====================================================================
# 第7部分：GPU 基础
# ====================================================================
def part7_gpu_basics():
    """GPU 检测、tensor 移动、CPU vs GPU 性能、no_grad 用法。"""
    print("\n" + "=" * 60)
    print("第7部分：GPU 基础")
    print("=" * 60)

    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if has_cuda else 'cpu')
    print(f"\n  CUDA: {has_cuda}, 设备: {device}")
    if has_cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 设备间移动
    t = torch.randn(3, 3)
    print(f"\n  CPU tensor: {t.device} → .to({device}): {t.to(device).device}")

    if has_cuda:
        t_gpu = torch.randn(3, device='cuda')
        try:
            t_gpu.numpy()
        except RuntimeError:
            print(f"  GPU→NumPy 必须先 .cpu(): {t_gpu.cpu().numpy()}")

    # --- 性能对比 ---
    print("\n--- 矩阵乘法性能 ---")
    for size in [256, 512, 1024, 2048]:
        A, B = torch.randn(size, size), torch.randn(size, size)
        t0 = time.time()
        for _ in range(5): _ = A @ B
        cpu_ms = (time.time() - t0) / 5 * 1000

        if has_cuda:
            Ag, Bg = A.cuda(), B.cuda()
            _ = Ag @ Bg; torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5): _ = Ag @ Bg
            torch.cuda.synchronize()
            gpu_ms = (time.time() - t0) / 5 * 1000
            print(f"  {size}x{size}: CPU={cpu_ms:.1f}ms, GPU={gpu_ms:.2f}ms, "
                  f"加速={cpu_ms/gpu_ms:.1f}x")
        else:
            print(f"  {size}x{size}: CPU={cpu_ms:.1f}ms（无GPU）")

    # --- no_grad ---
    print("\n--- torch.no_grad() ---")
    w = torch.tensor(2.0, requires_grad=True)
    y1 = w * 3.0                          # 训练：跟踪计算图
    with torch.no_grad():
        y2 = w * 3.0                      # 推理：不跟踪，省内存
    print(f"  训练: requires_grad={y1.requires_grad}")
    print(f"  推理: requires_grad={y2.requires_grad}")

    # 参数更新必须在 no_grad 中
    w = torch.tensor(5.0, requires_grad=True)
    (w ** 2).backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
    w.grad.zero_()
    print(f"  参数更新: w=5→{w.item():.1f}（no_grad 防止跟踪更新操作）")


# ====================================================================
# 第8部分：思考题
# ====================================================================
def part8_exercises():
    """思考题：检验你对 Tensor 与自动微分的理解。"""
    print("\n" + "=" * 60)
    print("第8部分：思考题")
    print("=" * 60)

    QA = [
        ("为什么 .grad 默认累加而不是替换？",
         "提示：想想梯度累积在大模型训练中的作用。",
         "累加允许多个小 batch 模拟大 batch——显存不够时分多次\n"
         "    forward-backward 再统一更新。默认累加正好支持此场景。"),

        ("非叶节点的 .grad 为什么不保留？需要时怎么办？",
         "提示：搜索 retain_grad()。",
         "中间 tensor 数量巨大，全保留梯度会撑爆显存。\n"
         "    需要时在 backward() 前调用 tensor.retain_grad()。"),

        ("下面代码有两个 bug：\n"
         "      w = torch.tensor(1.0, requires_grad=True)\n"
         "      for i in range(100):\n"
         "          loss = (w*2-5)**2; loss.backward()\n"
         "          w = w - 0.1 * w.grad",
         "提示：一个关于梯度累积，一个关于计算图。",
         "①没 zero_grad 导致累加发散 ②w=w-...创建新 tensor 不再是叶节点\n"
         "    修复: with torch.no_grad(): w -= 0.1*w.grad; w.grad.zero_()"),

        ("GPU 加速什么时候反而比 CPU 慢？",
         "提示：CPU↔GPU 数据传输的开销。",
         "小矩阵时 GPU 反而慢：传输有固定开销，内核启动有延迟，\n"
         "    小矩阵无法利用 GPU 并行。经验：维度>256 才有优势。"),
    ]

    for i, (q, hint, ans) in enumerate(QA, 1):
        print(f"\n思考题 {i}：{q}")
        print(f"  {hint}")
        print(f"  参考答案：{ans}")


# ====================================================================
# 主程序
# ====================================================================
if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   第5章 · 第1节 · Tensor 与自动微分                       |")
    print("|   从手算梯度到自动微分——PyTorch 的核心机制              |")
    print("+" + "=" * 58 + "+")

    part1_tensor_basics()            # Tensor 基础
    part2_numpy_conversion()         # Tensor ↔ NumPy 互转
    part3_autograd()                 # 自动微分
    part4_computation_graph()        # 计算图可视化
    part5_gradient_accumulation()    # 梯度累积陷阱
    part6_manual_vs_auto()           # 手动 vs 自动梯度对比
    part7_gpu_basics()               # GPU 基础
    part8_exercises()                # 思考题

    print("\n" + "=" * 60)
    print("本节总结")
    print("=" * 60)
    print("""
    1. Tensor ≈ NumPy ndarray + 自动微分 + GPU 支持
    2. requires_grad=True 开启梯度跟踪，backward() 自动求梯度
    3. 叶节点保留 .grad，非叶节点的梯度用完即丢
    4. 每次 backward 前必须 zero_grad（否则梯度累积导致训练崩溃）
    5. with torch.no_grad() 在推理和参数更新时节省内存
    6. 手动计算和 autograd 结果完全一致——PyTorch 只是自动化了链式法则
    7. GPU 加速对大规模矩阵运算效果显著

    下一节预告：第5章·第2节——nn.Module 与网络搭建
    （用 PyTorch 的高级 API 搭建和训练真正的神经网络）
    """)
