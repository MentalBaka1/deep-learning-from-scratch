"""
====================================================================
第5章 · 第2节 · nn.Module 与训练流程
====================================================================

【一句话总结】
nn.Module 是 PyTorch 中所有神经网络的基类——理解它就理解了
PyTorch 模型的标准写法和训练流程。

【为什么深度学习需要这个？】
- 所有 PyTorch 模型（包括 GPT、LLaMA）都继承 nn.Module
- nn.Module 自动管理参数、子模块、训练/评估模式
- 标准训练循环（forward → loss → backward → step）是固定模式
- 学会这个，后面实现 Transformer 就是搭积木

【核心概念】

1. nn.Module
   - 继承它，实现 __init__（定义层）和 forward（前向传播）
   - 自动注册所有 nn.Parameter 和子 Module
   - model.parameters() 返回所有可训练参数
   - model.train() / model.eval() 切换模式（影响 Dropout、BatchNorm）

2. 常用层
   - nn.Linear(in, out)：全连接层 y = xW^T + b
   - nn.ReLU(), nn.GELU()：激活函数
   - nn.Sequential：按顺序组合多个层
   - nn.Embedding(num, dim)：嵌入层（后面 Transformer 要用）

3. 损失函数
   - nn.MSELoss()：回归
   - nn.CrossEntropyLoss()：分类（内含 softmax，不要重复加！）
   - nn.BCEWithLogitsLoss()：二分类

4. 优化器
   - torch.optim.SGD, Adam, AdamW
   - optimizer.zero_grad() → loss.backward() → optimizer.step()

5. DataLoader
   - 自动 batching、shuffling、多进程加载
   - Dataset + DataLoader = 标准数据管道

6. 标准训练循环
   for epoch in range(num_epochs):
       for batch_x, batch_y in dataloader:
           pred = model(batch_x)          # 前向
           loss = criterion(pred, batch_y) # 损失
           optimizer.zero_grad()           # 清梯度
           loss.backward()                 # 反向
           optimizer.step()                # 更新

【前置知识】
第5章第1节 - Tensor与自动微分
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)


# ====================================================================
# 第一部分：自定义 nn.Module —— 写一个简单的 MLP
# ====================================================================
print("=" * 60)
print("第一部分：自定义 nn.Module —— 写一个简单的 MLP")
print("=" * 60)

# nn.Module 的核心规则：
#   1. __init__ 中定义所有层（nn.Linear, nn.ReLU 等）
#   2. forward 中定义前向传播逻辑（数据怎么流过各层）
#   3. 不要手动实现 backward，PyTorch 自动微分会搞定
#
# 重要：一定要调用 super().__init__()，否则 PyTorch 无法注册参数！


class SimpleMLP(nn.Module):
    """
    一个简单的三层 MLP（多层感知机）。
    结构: 输入 → 隐藏层1(ReLU) → 隐藏层2(ReLU) → 输出
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # 必须调用！不调用就没有参数管理
        self.fc1 = nn.Linear(input_dim, hidden_dim)    # 第一个全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 第二个全连接层
        self.fc3 = nn.Linear(hidden_dim, output_dim)   # 输出层
        self.relu = nn.ReLU()                           # 激活函数（可复用）

    def forward(self, x):
        """前向传播：定义数据如何流过网络"""
        x = self.relu(self.fc1(x))   # 第一层 + 激活
        x = self.relu(self.fc2(x))   # 第二层 + 激活
        x = self.fc3(x)              # 输出层（不加激活，损失函数里处理）
        return x


# 实例化模型
model = SimpleMLP(input_dim=2, hidden_dim=32, output_dim=3)
print(f"\n模型结构:\n{model}")

# 测试前向传播
dummy_input = torch.randn(5, 2)  # 5个样本，每个2维
dummy_output = model(dummy_input)  # 直接调用 model() 会触发 forward()
print(f"\n输入形状: {dummy_input.shape}")
print(f"输出形状: {dummy_output.shape}")
print(f"输出示例:\n{dummy_output.detach()}")


# ====================================================================
# 第二部分：参数查看与统计
# ====================================================================
print("\n" + "=" * 60)
print("第二部分：参数查看与统计")
print("=" * 60)

# model.parameters() 返回所有可训练参数的迭代器
# model.named_parameters() 还附带参数名称

print("\n--- model.named_parameters() ---")
for name, param in model.named_parameters():
    print(f"  {name:15s} | 形状 {str(param.shape):15s} | "
          f"元素数 {param.numel():5d} | 需要梯度: {param.requires_grad}")

# 统计总参数量——面试常考！
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params}")
print(f"可训练参数量: {trainable_params}")

# 手动验证: fc1 权重 (32x2) + fc1 偏置 (32) + fc2 权重 (32x32) + fc2 偏置 (32)
#           + fc3 权重 (3x32) + fc3 偏置 (3) = 64+32+1024+32+96+3 = 1251
print(f"手动计算: 2*32+32 + 32*32+32 + 32*3+3 = {2*32+32 + 32*32+32 + 32*3+3}")

# 查看子模块
print("\n--- model.named_modules() ---")
for name, module in model.named_modules():
    if name:  # 跳过根模块自身
        print(f"  {name}: {module.__class__.__name__}")


# ====================================================================
# 第三部分：常用层演示
# ====================================================================
print("\n" + "=" * 60)
print("第三部分：常用层演示")
print("=" * 60)

# --- nn.Linear ---
# y = xW^T + b，其中 W 的形状是 (out_features, in_features)
print("\n--- nn.Linear ---")
linear = nn.Linear(4, 3)  # 4维输入 → 3维输出
x_lin = torch.randn(2, 4)
y_lin = linear(x_lin)
print(f"  输入: {x_lin.shape} → 输出: {y_lin.shape}")
print(f"  权重形状: {linear.weight.shape}, 偏置形状: {linear.bias.shape}")

# --- nn.ReLU vs nn.GELU ---
# ReLU: max(0, x)，简单高效
# GELU: x * Φ(x)，GPT/BERT 使用，更平滑
print("\n--- ReLU vs GELU ---")
x_act = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu_out = nn.ReLU()(x_act)
gelu_out = nn.GELU()(x_act)
print(f"  输入:  {x_act.tolist()}")
print(f"  ReLU:  {relu_out.tolist()}")
print(f"  GELU:  {[f'{v:.4f}' for v in gelu_out.tolist()]}")
print("  区别: GELU 在 x<0 时不是完全为0，而是平滑过渡")

# --- nn.Embedding ---
# 嵌入层：整数索引 → 稠密向量（Transformer 的第一步！）
print("\n--- nn.Embedding ---")
vocab_size = 100   # 词表大小
embed_dim = 8      # 嵌入维度
embedding = nn.Embedding(vocab_size, embed_dim)

# 模拟一个小批次：3个句子，每个4个词
token_ids = torch.tensor([[2, 5, 7, 1],
                           [9, 3, 0, 4],
                           [6, 8, 1, 2]])
embedded = embedding(token_ids)
print(f"  词表大小: {vocab_size}, 嵌入维度: {embed_dim}")
print(f"  输入 token_ids: {token_ids.shape} → 嵌入输出: {embedded.shape}")
print(f"  每个整数 ID 被映射为一个 {embed_dim} 维向量")

# --- nn.Sequential ---
# 按顺序组合多个层，不需要手写 forward
print("\n--- nn.Sequential ---")
seq_model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.GELU(),         # 试试 GELU
    nn.Linear(32, 3),
)
print(f"  Sequential 模型:\n{seq_model}")
seq_out = seq_model(torch.randn(5, 2))
print(f"  输入 (5,2) → 输出 {seq_out.shape}")
print(f"  Sequential 参数量: {sum(p.numel() for p in seq_model.parameters())}")


# ====================================================================
# 第四部分：损失函数——CrossEntropyLoss 的陷阱
# ====================================================================
print("\n" + "=" * 60)
print("第四部分：损失函数——CrossEntropyLoss 的陷阱")
print("=" * 60)

# 【重要】CrossEntropyLoss 内部已经包含了 softmax！
# 如果你在模型输出端又加了 softmax，等于做了两次，结果会错！
#
# 正确用法：模型输出原始 logits（未归一化的分数），直接传给 CrossEntropyLoss
# 错误用法：模型输出 softmax 后的概率，再传给 CrossEntropyLoss

logits = torch.tensor([[2.0, 1.0, 0.1]])   # 原始分数（logits）
target = torch.tensor([0])                   # 真实类别索引

criterion = nn.CrossEntropyLoss()
loss_correct = criterion(logits, target)
print(f"\n正确做法 —— 直接传 logits:")
print(f"  logits = {logits.tolist()}, target = {target.item()}")
print(f"  CrossEntropyLoss = {loss_correct.item():.4f}")

# 手动验证: -log(softmax(logits)[target])
softmax_probs = torch.softmax(logits, dim=1)
loss_manual = -torch.log(softmax_probs[0, target[0]])
print(f"  手动计算: softmax = {[f'{p:.4f}' for p in softmax_probs[0].tolist()]}")
print(f"            -log(softmax[0]) = {loss_manual.item():.4f}  (一致!)")

# 错误示范
loss_wrong = criterion(softmax_probs, target)
print(f"\n错误做法 —— 先 softmax 再传入:")
print(f"  CrossEntropyLoss = {loss_wrong.item():.4f}  (结果不同，这是错的！)")
print("  教训: 模型的 forward 不要在最后加 softmax")

# 其他常用损失函数
print("\n--- 其他损失函数一览 ---")
# MSELoss: 回归任务
pred_reg = torch.tensor([2.5, 0.0, 2.1])
true_reg = torch.tensor([3.0, -0.5, 2.0])
print(f"  MSELoss: {nn.MSELoss()(pred_reg, true_reg).item():.4f}")

# BCEWithLogitsLoss: 二分类（内含 sigmoid，同理不要重复加）
pred_bin = torch.tensor([0.8, -1.2, 2.0])
true_bin = torch.tensor([1.0, 0.0, 1.0])
print(f"  BCEWithLogitsLoss: {nn.BCEWithLogitsLoss()(pred_bin, true_bin).item():.4f}")


# ====================================================================
# 第五部分：Dataset 和 DataLoader
# ====================================================================
print("\n" + "=" * 60)
print("第五部分：Dataset 和 DataLoader")
print("=" * 60)

# Dataset 的规则很简单：
#   1. __len__: 返回数据集大小
#   2. __getitem__: 根据索引返回一个样本


class SpiralDataset(Dataset):
    """
    螺旋形数据集：3类螺旋形数据，用于分类。
    与第2章相同的数据，但这次用 PyTorch 的 Dataset 封装。
    """

    def __init__(self, n_points_per_class=100, n_classes=3, noise=0.2):
        super().__init__()
        self.X, self.y = self._generate(n_points_per_class, n_classes, noise)

    def _generate(self, n_points, n_classes, noise):
        """生成螺旋形数据"""
        X = np.zeros((n_points * n_classes, 2))
        y = np.zeros(n_points * n_classes, dtype=np.int64)

        for cls in range(n_classes):
            start = cls * n_points
            r = np.linspace(0.0, 1.0, n_points)           # 半径
            t = (np.linspace(cls * 4, (cls + 1) * 4, n_points)
                 + np.random.randn(n_points) * noise)      # 角度 + 噪声
            X[start:start + n_points, 0] = r * np.sin(t)
            X[start:start + n_points, 1] = r * np.cos(t)
            y[start:start + n_points] = cls

        return torch.FloatTensor(X), torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据集
dataset = SpiralDataset(n_points_per_class=100, n_classes=3)
print(f"\n数据集大小: {len(dataset)}")
print(f"单个样本: X={dataset[0][0]}, y={dataset[0][1]}")

# 创建 DataLoader
# batch_size: 每批多少样本
# shuffle: 是否打乱顺序（训练时 True，测试时 False）
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 看一个 batch
batch_x, batch_y = next(iter(dataloader))
print(f"\n一个 batch:")
print(f"  batch_x 形状: {batch_x.shape}")
print(f"  batch_y 形状: {batch_y.shape}")
print(f"  batch_y 内容: {batch_y.tolist()}")

# 遍历整个 DataLoader
n_batches = 0
for bx, by in dataloader:
    n_batches += 1
print(f"\n总共 {len(dataset)} 个样本, batch_size=32, "
      f"共 {n_batches} 个 batch（最后一个可能不满32）")


# ====================================================================
# 第六部分：完整训练循环——在螺旋数据上训练 MLP
# ====================================================================
print("\n" + "=" * 60)
print("第六部分：完整训练循环——在螺旋数据上训练 MLP")
print("=" * 60)

# 这是 PyTorch 训练的标准模式，后面所有模型（CNN、RNN、Transformer）
# 都遵循这个框架，只是模型结构不同而已。

# 1. 准备数据
train_dataset = SpiralDataset(n_points_per_class=200, n_classes=3)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 定义模型
spiral_model = SimpleMLP(input_dim=2, hidden_dim=64, output_dim=3)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(spiral_model.parameters(), lr=0.01)

# 4. 训练循环
num_epochs = 50
loss_history = []

print(f"\n开始训练: {num_epochs} 个 epoch, "
      f"数据量 {len(train_dataset)}, batch_size=64")
print("-" * 50)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        # ---- 前向传播 ----
        logits = spiral_model(batch_x)         # 模型输出 logits
        loss = criterion(logits, batch_y)       # 计算损失

        # ---- 反向传播 + 参数更新 ----
        optimizer.zero_grad()                   # 清除上一步的梯度（必须！）
        loss.backward()                         # 反向传播计算梯度
        optimizer.step()                        # 优化器更新参数

        # ---- 记录统计量 ----
        epoch_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(logits, dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / total
    accuracy = correct / total
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

print("-" * 50)
print(f"训练完成! 最终损失: {loss_history[-1]:.4f}")

# 最终在全量数据上评估
spiral_model.eval()  # 切换到评估模式
with torch.no_grad():  # 评估时不需要计算梯度
    all_logits = spiral_model(train_dataset.X)
    _, all_preds = torch.max(all_logits, dim=1)
    final_acc = (all_preds == train_dataset.y).float().mean()
    print(f"全量数据准确率: {final_acc:.2%}")
spiral_model.train()  # 切回训练模式


# ====================================================================
# 第七部分：train vs eval 模式——Dropout 的效果
# ====================================================================
print("\n" + "=" * 60)
print("第七部分：train vs eval 模式——Dropout 的效果")
print("=" * 60)

# model.train() 和 model.eval() 影响的层：
#   - Dropout: train 时随机丢弃神经元，eval 时全部保留
#   - BatchNorm: train 时用当前 batch 统计量，eval 时用全局统计量
#
# 忘记切换模式是常见的 bug！


class MLPWithDropout(nn.Module):
    """带 Dropout 的 MLP，用于演示 train/eval 模式的区别"""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)  # 训练时随机丢弃50%的神经元
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)   # 这一层的行为取决于 train/eval 模式
        x = self.fc2(x)
        return x


dropout_model = MLPWithDropout(2, 16, 3, dropout_rate=0.5)
test_input = torch.randn(1, 2)

# 训练模式：Dropout 生效，每次输出不同
dropout_model.train()
print(f"\n训练模式（Dropout 生效）—— 多次前向传播输出不同:")
for i in range(3):
    out = dropout_model(test_input)
    print(f"  第{i+1}次: {[f'{v:.4f}' for v in out[0].tolist()]}")

# 评估模式：Dropout 关闭，每次输出相同
dropout_model.eval()
print(f"\n评估模式（Dropout 关闭）—— 多次前向传播输出相同:")
for i in range(3):
    out = dropout_model(test_input)
    print(f"  第{i+1}次: {[f'{v:.4f}' for v in out[0].tolist()]}")

print("\n注意:")
print("  - 训练时用 model.train()，Dropout 随机丢弃，防止过拟合")
print("  - 推理时用 model.eval()，Dropout 关闭，使用全部神经元")
print("  - 推理时还应配合 torch.no_grad()，节省显存并加速")


# ====================================================================
# 第八部分：模型保存与加载
# ====================================================================
print("\n" + "=" * 60)
print("第八部分：模型保存与加载")
print("=" * 60)

# 推荐方式：只保存 state_dict（参数字典），而不是整个模型
# 原因：保存整个模型会绑定具体的类定义和文件路径，不够灵活

save_path = "spiral_mlp.pth"

# --- 保存 ---
torch.save(spiral_model.state_dict(), save_path)
print(f"\n模型参数已保存到: {save_path}")

# 看看 state_dict 里有什么
state = spiral_model.state_dict()
print(f"\nstate_dict 包含的 key:")
for key, tensor in state.items():
    print(f"  {key:15s} → {tensor.shape}")

# --- 加载 ---
# 先创建一个新的模型实例（结构必须一致）
loaded_model = SimpleMLP(input_dim=2, hidden_dim=64, output_dim=3)

# 加载保存的参数
loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
loaded_model.eval()
print(f"\n模型参数已从 {save_path} 加载")

# 验证加载后的模型和原模型输出一致
with torch.no_grad():
    spiral_model.eval()
    orig_out = spiral_model(dummy_input)
    load_out = loaded_model(dummy_input)
    diff = (orig_out - load_out).abs().max().item()
    print(f"原模型 vs 加载模型的最大输出差异: {diff:.1e}")
    print(f"验证通过: {diff < 1e-6}")

# 也可以同时保存优化器状态（用于恢复训练）
checkpoint = {
    "epoch": num_epochs,
    "model_state_dict": spiral_model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss_history[-1],
}
torch.save(checkpoint, "spiral_checkpoint.pth")
print("\n完整 checkpoint（模型+优化器+训练状态）已保存到: spiral_checkpoint.pth")
print("  恢复训练时加载:")
print("    ckpt = torch.load('spiral_checkpoint.pth')")
print("    model.load_state_dict(ckpt['model_state_dict'])")
print("    optimizer.load_state_dict(ckpt['optimizer_state_dict'])")

# 清理临时文件
for f in [save_path, "spiral_checkpoint.pth"]:
    if os.path.exists(f):
        os.remove(f)
print("\n(临时文件已清理)")


# ====================================================================
# 总结
# ====================================================================
print("\n" + "=" * 60)
print("总结：本节核心要点")
print("=" * 60)
print("""
  1. nn.Module 是所有模型的基类
     - 继承它，实现 __init__ 和 forward
     - 自动管理参数、子模块

  2. 常用层: Linear, ReLU, GELU, Embedding, Sequential, Dropout
     - Sequential 适合简单堆叠，复杂结构用自定义 Module

  3. CrossEntropyLoss 内含 softmax，不要在模型输出端重复加！
     - 同理 BCEWithLogitsLoss 内含 sigmoid

  4. 标准训练循环（背下来！）：
     for batch_x, batch_y in dataloader:
         logits = model(batch_x)            # 前向
         loss = criterion(logits, batch_y)   # 损失
         optimizer.zero_grad()               # 清梯度
         loss.backward()                     # 反向
         optimizer.step()                    # 更新

  5. train() vs eval(): 影响 Dropout 和 BatchNorm
     - 训练: model.train()
     - 推理: model.eval() + torch.no_grad()

  6. 保存/加载: torch.save(model.state_dict(), path)
     - 推荐只保存 state_dict，不保存整个模型

  下一节预告: 第5章 · 第3节 · GPU 加速与实战技巧
""")


# ====================================================================
# 思考题
# ====================================================================
print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 【为什么 zero_grad 必须在 backward 之前？】
   如果不调用 optimizer.zero_grad()，连续两次 backward 会怎样？
   梯度会累加还是覆盖？在什么场景下你可能故意不清零梯度？
   提示: PyTorch 的梯度默认是累加的。在 gradient accumulation（模拟大
   batch）场景下，你会故意累积多个 mini-batch 的梯度后再 step。

2. 【nn.Sequential vs 自定义 Module】
   以下两个模型在功能上等价吗？各自的优缺点是什么？
   >>> model_a = nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,3))
   >>> class ModelB(nn.Module):
   ...     def __init__(self):
   ...         super().__init__()
   ...         self.fc1 = nn.Linear(2, 64)
   ...         self.fc2 = nn.Linear(64, 3)
   ...     def forward(self, x):
   ...         return self.fc2(torch.relu(self.fc1(x)))
   提示: 功能等价。Sequential 写法简洁但不够灵活（无法实现跳跃连接、
   多输入多输出等）。实际项目中复杂模型都用自定义 Module。

3. 【CrossEntropyLoss 的输入格式】
   CrossEntropyLoss 期望的 target 是类别索引（如 [0, 2, 1]），
   而不是 one-hot 向量（如 [[1,0,0], [0,0,1], [0,1,0]]）。
   如果你的标签是 one-hot 格式，应该怎么转换？
   提示: torch.argmax(one_hot, dim=1)

4. 【DataLoader 的 num_workers】
   DataLoader 有一个 num_workers 参数用于多进程数据加载。
   它设为多少合适？设太大会怎样？在 Windows 上使用时有什么注意事项？
   提示: 通常设为 CPU 核数的一半。Windows 上必须在 if __name__ ==
   '__main__' 中使用，否则会报错。

5. 【保存整个模型 vs 只保存 state_dict】
   torch.save(model, path) 可以保存整个模型，为什么不推荐？
   提示: 保存整个模型用的是 pickle，会绑定类的定义和所在文件路径。
   如果你重构了代码（改了类名、移了文件），加载就会失败。
   state_dict 只保存参数张量，与代码结构无关。
""")

print("下一节预告: 第5章 · 第3节 · GPU 加速与实战技巧")
