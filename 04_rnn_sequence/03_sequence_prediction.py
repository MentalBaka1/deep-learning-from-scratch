"""
==============================================================
第4章 第3节：时间序列预测 —— LSTM 实战
==============================================================

【任务】
用 LSTM 预测正弦波：
  - 给 LSTM 看过去 T 个时间步
  - 预测未来 1 个时间步
  - 这是时间序列预测的标准设置

这个任务虽然简单（正弦波是确定性的），
但完美地展示了：LSTM 如何"学会"一种时间模式

【存在理由】
这是 RNN/LSTM 最经典的应用之一：
  股价预测、天气预测、心电图分析、语音识别……
  都可以用类似的框架处理
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================
# Part 1: 数据准备 —— 正弦波 + 滑动窗口
# ============================================================
print("=" * 50)
print("Part 1: 数据准备")
print("=" * 50)

"""
生成正弦波时间序列，用滑动窗口切成训练样本

  时间序列：[x_1, x_2, x_3, ..., x_N]
  样本 i：输入 = [x_i, x_{i+1}, ..., x_{i+T-1}]
          标签 = x_{i+T}

这就是"给模型看过去T步，预测下一步"的标准做法
"""

# 生成带噪声的正弦波
T = 500  # 总时间步
t = np.linspace(0, 4 * np.pi, T)
# 主频率 + 少量第二谐波 + 噪声
series = np.sin(t) + 0.3 * np.sin(3*t) + np.random.randn(T) * 0.1

# 归一化到 [-1, 1]
series_min, series_max = series.min(), series.max()
series_norm = 2 * (series - series_min) / (series_max - series_min) - 1

# 用滑动窗口创建样本
seq_len = 30  # 看过去30步
X_list, y_list = [], []
for i in range(T - seq_len - 1):
    X_list.append(series_norm[i:i+seq_len])
    y_list.append(series_norm[i+seq_len])

X_all = np.array(X_list)  # (N, seq_len)
y_all = np.array(y_list)  # (N,)

# 划分训练集和测试集
n_train = int(len(X_all) * 0.8)
X_train = X_all[:n_train].reshape(-1, seq_len, 1)  # (N, seq_len, 1)
y_train = y_all[:n_train]
X_test = X_all[n_train:].reshape(-1, seq_len, 1)
y_test = y_all[n_train:]

print(f"总时间序列长度：{T}")
print(f"训练样本：{X_train.shape}，测试样本：{X_test.shape}")

# ============================================================
# Part 2: 简化版 LSTM（适合回归）
# ============================================================
print("\nPart 2: 构建 LSTM 预测模型")
print("=" * 50)

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

class LSTMPredictor:
    """
    LSTM 时间序列预测器
    输入：(batch, seq_len, 1)
    输出：(batch, 1)  下一步的预测值
    """
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        self.hidden_size = hidden_size
        combined = input_size + hidden_size

        scale = 1.0 / np.sqrt(combined)
        self.W = np.random.randn(combined, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)
        self.b[hidden_size:2*hidden_size] = 1.0  # 遗忘门偏置=1

        self.W_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros(output_size)

    def forward(self, X):
        """
        X: (batch, seq_len, input_size)
        返回最后一步的预测
        """
        batch, seq_len, _ = X.shape
        H = self.hidden_size
        h = np.zeros((batch, H))
        c = np.zeros((batch, H))

        self.seq_cache = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            combined = np.concatenate([h, x_t], axis=1)
            gates = combined @ self.W + self.b

            f = sigmoid(gates[:, :H])
            i = sigmoid(gates[:, H:2*H])
            g = np.tanh(gates[:, 2*H:3*H])
            o = sigmoid(gates[:, 3*H:])

            c = f * c + i * g
            h = o * np.tanh(c)

            self.seq_cache.append((combined, f, i, g, o, c.copy(), h.copy(), gates))

        # 只用最后一步的隐状态做预测
        out = h @ self.W_out + self.b_out  # (batch, 1)
        self.last_h = h
        return out.ravel()  # (batch,)

    def backward(self, y_pred, y_true, lr):
        """
        MSE 损失的反向传播
        """
        batch = len(y_true)
        H = self.hidden_size

        # MSE 反向
        d_out = 2 * (y_pred - y_true) / batch  # (batch,)

        # 输出层反向
        dW_out = self.last_h.T @ d_out.reshape(-1, 1)
        db_out = d_out.sum()
        d_h = d_out.reshape(-1, 1) @ self.W_out.T  # (batch, H)

        # 初始化梯度累积
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        d_h_next = d_h
        d_c_next = np.zeros((batch, H))

        # BPTT
        for t in reversed(range(len(self.seq_cache))):
            combined, f, i, g, o, c_t, h_t, gates = self.seq_cache[t]
            c_prev = self.seq_cache[t-1][5] if t > 0 else np.zeros((batch, H))

            tanh_c = np.tanh(c_t)
            d_o = d_h_next * tanh_c
            d_c_next += d_h_next * o * (1 - tanh_c**2)

            d_f = d_c_next * c_prev
            d_i = d_c_next * g
            d_g = d_c_next * i
            d_c_prev = d_c_next * f

            d_gates = np.zeros_like(gates)
            d_gates[:, :H] = d_f * f * (1-f)
            d_gates[:, H:2*H] = d_i * i * (1-i)
            d_gates[:, 2*H:3*H] = d_g * (1-g**2)
            d_gates[:, 3*H:] = d_o * o * (1-o)

            dW += combined.T @ d_gates
            db += d_gates.sum(0)

            d_combined = d_gates @ self.W.T
            d_h_next = d_combined[:, :H]
            d_c_next = d_c_prev

        # 梯度裁剪
        for grad in [dW, db]:
            np.clip(grad, -1, 1, out=grad)

        # 更新
        self.W -= lr * dW
        self.b -= lr * db
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

# ============================================================
# Part 3: 训练
# ============================================================
print("Part 3: 训练 LSTM 预测器")
print("=" * 50)

model = LSTMPredictor(input_size=1, hidden_size=32, output_size=1)
n_epochs = 100
batch_size = 32
lr = 0.005
train_losses = []

for epoch in range(n_epochs):
    idx = np.random.permutation(len(X_train))
    epoch_losses = []

    for start in range(0, len(X_train), batch_size):
        batch_idx = idx[start:start+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        y_pred = model.forward(X_batch)
        loss = np.mean((y_pred - y_batch)**2)
        epoch_losses.append(loss)
        model.backward(y_pred, y_batch, lr)

    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)

    if epoch % 20 == 0:
        # 测试集评估
        test_preds = []
        for start in range(0, len(X_test), 64):
            batch = X_test[start:start+64]
            preds = model.forward(batch)
            test_preds.append(preds)
        test_preds = np.concatenate(test_preds)
        test_mse = np.mean((test_preds - y_test)**2)
        print(f"  Epoch {epoch:3d}: train_loss={avg_loss:.4f}, test_mse={test_mse:.4f}")

# ============================================================
# Part 4: 可视化预测结果
# ============================================================
print("\nPart 4: 可视化预测结果")
print("=" * 50)

# 生成完整预测
test_preds_all = []
for start in range(0, len(X_test), 64):
    test_preds_all.append(model.forward(X_test[start:start+64]))
test_preds_all = np.concatenate(test_preds_all)

# 反归一化
def denorm(x):
    return (x + 1) / 2 * (series_max - series_min) + series_min

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('LSTM 时间序列预测', fontsize=14)

# 训练集 loss
ax = axes[0][0]
ax.semilogy(train_losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss（对数）')
ax.set_title('训练损失曲线')
ax.grid(True, alpha=0.3)

# 预测 vs 真实（归一化）
ax = axes[0][1]
t_test = range(len(y_test))
ax.plot(t_test, y_test, 'b-', linewidth=1.5, alpha=0.8, label='真实值')
ax.plot(t_test, test_preds_all, 'r--', linewidth=1.5, alpha=0.8, label='LSTM 预测')
ax.set_xlabel('时间步')
ax.set_ylabel('归一化值')
ax.set_title('测试集预测（归一化）')
ax.legend()
ax.grid(True, alpha=0.3)

# 详细对比（前80步）
ax = axes[1][0]
n_show = 80
ax.plot(range(n_show), denorm(y_test[:n_show]), 'b-', linewidth=2, label='真实值')
ax.plot(range(n_show), denorm(test_preds_all[:n_show]), 'r--', linewidth=2, label='LSTM预测')
ax.set_xlabel('时间步')
ax.set_ylabel('值（原始尺度）')
ax.set_title('预测细节（前80步）')
ax.legend()
ax.grid(True, alpha=0.3)

# 散点图：预测 vs 真实（越接近对角线越好）
ax = axes[1][1]
ax.scatter(y_test, test_preds_all, alpha=0.3, s=10, c='blue')
min_val = min(y_test.min(), test_preds_all.min())
max_val = max(y_test.max(), test_preds_all.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
correlation = np.corrcoef(y_test, test_preds_all)[0, 1]
ax.set_xlabel('真实值')
ax.set_ylabel('预测值')
ax.set_title(f'真实 vs 预测散点图\n（相关系数 r={correlation:.4f}）')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_rnn_sequence/sequence_prediction.png', dpi=100, bbox_inches='tight')
print("图片已保存：04_rnn_sequence/sequence_prediction.png")
plt.show()

final_test_mse = np.mean((test_preds_all - y_test)**2)
print(f"\n最终测试 MSE：{final_test_mse:.4f}")
print(f"预测相关系数：{correlation:.4f}  （越接近1越好）")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【多步预测】
   当前模型只预测"下一步"。
   要预测"未来10步"，有两种方法：
   方法A：模型直接输出10个值
   方法B：递归预测（用预测值作为下一步的输入）
   修改代码实现方法B，观察预测误差如何随步数累积。

2. 【序列长度实验】
   把 seq_len 从 30 改为 5 和 60。
   - seq_len=5：历史太短，准确率如何？
   - seq_len=60：历史更长，是否更准？
   存在一个最优的 seq_len 吗？

3. 【更复杂的序列】
   把正弦波换成"正弦波 + 随机游走"（非周期，更难预测）：
   series = sin(t) + cumsum(0.01 * randn(T))
   LSTM 还能预测吗？MSE 会增加多少？
""")
