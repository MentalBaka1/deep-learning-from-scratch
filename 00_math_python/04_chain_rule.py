"""
==============================================================
第0章 第4节：链式法则 —— 反向传播的数学心脏
==============================================================

【为什么需要它？】
神经网络是一个复杂的复合函数：
  output = f5(f4(f3(f2(f1(x)))))
训练网络需要计算 loss 对每个参数的导数。
直接求导太复杂！链式法则把这件事变成了一个系统的流程。

【生活类比】
工厂流水线：原材料经过5道工序变成产品。
产品有缺陷时，要找出是哪道工序的问题。
链式法则就像"质检倒查"：
  从成品缺陷率开始，倒推每道工序对缺陷率的"贡献"。
  越靠近原材料的工序，需要把链条上所有工序的影响乘在一起。

这就是"反向传播"（Backpropagation）的本质！
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# Part 1: 链式法则的本质
# ============================================================
print("=" * 50)
print("Part 1: 链式法则基础")
print("=" * 50)

"""
单变量链式法则：
  y = f(g(x))
  dy/dx = f'(g(x)) * g'(x)
        = (df/dg) * (dg/dx)

直觉：
  x 变化 Δx
  → g(x) 变化 Δg ≈ g'(x) * Δx
  → y 变化 Δy ≈ f'(g) * Δg = f'(g(x)) * g'(x) * Δx

所以 dy/dx = f'(g(x)) * g'(x)
"连续的影响相乘"
"""

# 例子：y = sin(x²)
# g(x) = x²   → g'(x) = 2x
# f(g) = sin(g) → f'(g) = cos(g)
# dy/dx = cos(x²) * 2x

def y(x):
    return np.sin(x**2)

def dy_dx_analytical(x):
    return np.cos(x**2) * 2 * x

def dy_dx_numerical(x, h=1e-6):
    return (y(x + h) - y(x - h)) / (2 * h)

x_test = 1.5
print(f"f(x) = sin(x²)，在 x={x_test}：")
print(f"  解析导数（链式法则）= cos(x²)*2x = {dy_dx_analytical(x_test):.8f}")
print(f"  数值导数（直接算）  = {dy_dx_numerical(x_test):.8f}")
print(f"  完全一致！链式法则有效 ✓")

# ============================================================
# Part 2: 计算图 —— 把复合函数可视化
# ============================================================
print("\n" + "=" * 50)
print("Part 2: 计算图 —— 分解复合函数")
print("=" * 50)

"""
计算图把复杂的计算分解成简单节点的组合。
每个节点只做一件简单的事（加、乘、exp等）。
反向传播在这个图上从输出到输入传递梯度。

例子：f = (a + b) * c
  - 节点1：q = a + b
  - 节点2：f = q * c

前向传播：按箭头方向算值
反向传播：按逆箭头方向传梯度

局部梯度（local gradient）：该节点自己的导数
上游梯度（upstream gradient）：从输出传来的梯度
该节点的贡献 = 局部梯度 * 上游梯度
"""

# 手动计算图：f = (a + b) * c
a, b, c = 2.0, 3.0, 4.0

# ===== 前向传播 =====
q = a + b           # q = 5
f = q * c           # f = 20
print(f"前向传播：a={a}, b={b}, c={c}")
print(f"  q = a + b = {q}")
print(f"  f = q * c = {f}")

# ===== 反向传播 =====
# 从 df/df = 1 开始（初始梯度）
df_df = 1.0

# 节点2：f = q * c
# 局部梯度：df/dq = c = 4，df/dc = q = 5
df_dq = c          # = 4
df_dc = q          # = 5

# 节点1：q = a + b
# 局部梯度：dq/da = 1，dq/db = 1（加法节点！梯度平等分配）
# 链式法则：df/da = df/dq * dq/da = c * 1 = c = 4
df_da = df_dq * 1  # = 4
df_db = df_dq * 1  # = 4

print(f"\n反向传播（梯度从右往左传）：")
print(f"  df/df = {df_df}  （输出的梯度初始化为1）")
print(f"  df/dq = c = {df_dq}  （q变1，f变c）")
print(f"  df/dc = q = {df_dc}  （c变1，f变q）")
print(f"  df/da = df/dq * 1 = {df_da}  （加法节点：梯度直通）")
print(f"  df/db = df/dq * 1 = {df_db}  （加法节点：梯度直通）")

# 数值验证
h = 1e-5
print(f"\n数值验证：")
print(f"  df/da ≈ {((a+h+b)*c - f)/h:.4f}  （应为{df_da}）")
print(f"  df/db ≈ {((a+b+h)*c - f)/h:.4f}  （应为{df_db}）")
print(f"  df/dc ≈ {((a+b)*(c+h) - f)/h:.4f}  （应为{df_dc}）")

# ============================================================
# Part 3: 重要节点的局部梯度规律
# ============================================================
print("\n" + "=" * 50)
print("Part 3: 常见节点的反向传播规律")
print("=" * 50)

"""
记住这些规律，就能手写任何网络的反向传播！

1. 加法节点：q = a + b
   dq/da = 1，dq/db = 1
   → 梯度"分叉"后直接传给两个输入（不衰减！）
   → 加法节点是"梯度分配器"

2. 乘法节点：q = a * b
   dq/da = b，dq/db = a
   → 梯度交叉乘：a的梯度=b，b的梯度=a
   → 乘法节点是"梯度交换器"

3. ReLU节点：q = max(0, x)
   dq/dx = 1 if x > 0 else 0
   → 正数：梯度直通；负数：梯度被"杀死"

4. Sigmoid节点：q = 1/(1+e^(-x))
   dq/dx = q * (1 - q)
   → 最大值0.25，可能导致梯度消失！
"""

print("节点类型           | 前向计算          | 反向梯度（局部）")
print("-" * 60)
print("加法 q = a + b    | 相加              | dq/da = 1, dq/db = 1")
print("乘法 q = a * b    | 相乘              | dq/da = b, dq/db = a")
print("ReLU q = max(0,x) | 截断              | dq/dx = 1 if x>0 else 0")
print("Exp  q = e^x      | 指数              | dq/dx = e^x = q")
print("Log  q = ln(x)    | 对数              | dq/dx = 1/x")
print("Sigmoid q=σ(x)    | 压缩到[0,1]       | dq/dx = q*(1-q)")

# ============================================================
# Part 4: 手写一个神经元的完整反向传播
# ============================================================
print("\n" + "=" * 50)
print("Part 4: 完整例子 —— 单个神经元的前向+反向")
print("=" * 50)

"""
单个神经元：
  1. 线性：z = w1*x1 + w2*x2 + b
  2. 激活：a = sigmoid(z)
  3. 损失：L = (a - y)²  （用MSE举例）

反向传播链：
  dL/da = 2*(a - y)
  da/dz = a * (1 - a)     （sigmoid 导数）
  dz/dw1 = x1, dz/dw2 = x2, dz/db = 1

  dL/dw1 = dL/da * da/dz * dz/dw1 = 2*(a-y) * a*(1-a) * x1
  dL/dw2 = 2*(a-y) * a*(1-a) * x2
  dL/db  = 2*(a-y) * a*(1-a) * 1
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SingleNeuron:
    def __init__(self):
        self.w1 = 0.5
        self.w2 = -0.3
        self.b = 0.1
        # 缓存前向传播的中间值（反向传播需要！）
        self.cache = {}

    def forward(self, x1, x2, y_true):
        """前向传播：计算预测值和损失"""
        z = self.w1 * x1 + self.w2 * x2 + self.b
        a = sigmoid(z)
        loss = (a - y_true) ** 2

        # 缓存（以备反向传播使用）
        self.cache = {'x1': x1, 'x2': x2, 'z': z, 'a': a, 'y_true': y_true}
        return a, loss

    def backward(self):
        """反向传播：计算所有参数的梯度"""
        x1 = self.cache['x1']
        x2 = self.cache['x2']
        a  = self.cache['a']
        y  = self.cache['y_true']

        # 从 loss 开始反向传
        dL_da = 2 * (a - y)             # loss 对 a 的梯度
        da_dz = a * (1 - a)             # sigmoid 的局部梯度
        dL_dz = dL_da * da_dz           # 链式法则

        # 线性层的梯度（z = w1*x1 + w2*x2 + b）
        dL_dw1 = dL_dz * x1
        dL_dw2 = dL_dz * x2
        dL_db  = dL_dz * 1

        return {'dw1': dL_dw1, 'dw2': dL_dw2, 'db': dL_db}

    def numerical_gradient(self, x1, x2, y_true, h=1e-5):
        """数值梯度（用来验证反向传播的正确性）"""
        grads = {}

        for param_name in ['w1', 'w2', 'b']:
            # 保存原始值
            orig = getattr(self, param_name)

            # f(param + h)
            setattr(self, param_name, orig + h)
            _, loss_plus = self.forward(x1, x2, y_true)

            # f(param - h)
            setattr(self, param_name, orig - h)
            _, loss_minus = self.forward(x1, x2, y_true)

            # 恢复原始值
            setattr(self, param_name, orig)

            grads[f'd{param_name}'] = (loss_plus - loss_minus) / (2 * h)

        return grads

# 测试
neuron = SingleNeuron()
x1, x2, y_true = 0.8, -0.5, 1.0
a, loss = neuron.forward(x1, x2, y_true)

print(f"输入：x1={x1}, x2={x2}, 目标：y={y_true}")
print(f"预测：a={a:.4f}, 损失 L={loss:.4f}")

analytical_grads = neuron.backward()
numerical_grads = neuron.numerical_gradient(x1, x2, y_true)

print(f"\n梯度验证：")
print(f"{'参数':<8} {'解析梯度':>15} {'数值梯度':>15} {'误差':>15}")
print("-" * 55)
for key in ['dw1', 'dw2', 'db']:
    ag = analytical_grads[key]
    ng = numerical_grads[key]
    err = abs(ag - ng)
    print(f"{key:<8} {ag:>15.8f} {ng:>15.8f} {err:>15.2e}")

print("\n误差 < 1e-7，说明反向传播实现正确！✓")

# ============================================================
# Part 5: 可视化 —— 计算图和梯度流动
# ============================================================
def visualize_computation_graph():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('计算图：f = (a + b) * c 的前向和反向传播', fontsize=14)

    # 节点
    nodes = {
        'a': (1, 6, 'a=2'),
        'b': (1, 4, 'b=3'),
        'c': (1, 2, 'c=4'),
        '+': (4, 5, 'q=a+b\n=5'),
        '*': (7, 4, 'f=q*c\n=20'),
    }

    for name, (x, y, label) in nodes.items():
        if name in ['+', '*']:
            circle = plt.Circle((x, y), 0.7, color='steelblue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center', fontsize=9, color='white')
        else:
            rect = mpatches.FancyBboxPatch((x-0.5, y-0.4), 1, 0.8,
                                           boxstyle="round,pad=0.1",
                                           facecolor='lightgreen', edgecolor='green')
            ax.add_patch(rect)
            ax.text(x, y, label, ha='center', va='center', fontsize=10)

    # 前向箭头（蓝色）
    arrows_forward = [
        ((1.5, 6), (3.3, 5.3)),
        ((1.5, 4), (3.3, 4.7)),
        ((1.5, 2), (6.3, 3.5)),
        ((4.7, 5), (6.3, 4.3)),
    ]
    for start, end in arrows_forward:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 反向箭头（红色，虚线）
    arrows_backward = [
        ((3.3, 5.3), (1.5, 6)),
        ((3.3, 4.7), (1.5, 4)),
        ((6.3, 3.5), (1.5, 2)),
        ((6.3, 4.3), (4.7, 5)),
    ]
    backward_labels = ['df/da=4', 'df/db=4', 'df/dc=5', 'df/dq=4']
    for (start, end), label in zip(arrows_backward, backward_labels):
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='red', lw=2,
                                  linestyle='dashed'))
        ax.text(mid_x, mid_y + 0.3, label, color='red', fontsize=9, ha='center')

    ax.text(0.5, 7.5, '蓝色=前向传播（计算值）', color='blue', fontsize=10)
    ax.text(0.5, 7.0, '红色=反向传播（传递梯度）', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig('00_math_python/computation_graph.png', dpi=100, bbox_inches='tight')
    print("\n图片已保存：00_math_python/computation_graph.png")
    plt.show()

visualize_computation_graph()

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【乘法节点验证】
   对 f = a * b（a=3, b=5），手动计算：
   - df/da = ?（链式法则）
   - 用数值方法验证（h=1e-5）
   - 为什么乘法节点的梯度会"交换"两个输入？

2. 【两层神经网络】
   z1 = w1 * x + b1        （线性层）
   a1 = relu(z1)           （激活层）
   z2 = w2 * a1 + b2       （线性层）
   L  = (z2 - y)²          （损失）

   手推 dL/dw1 的完整链式展开式，
   然后用代码实现并用数值梯度验证。

3. 【梯度累积】
   如果一个节点的输出被两个不同的下游节点使用，
   它的梯度是两个下游梯度的什么关系？（提示：加法）
   在代码里，当多个梯度流汇聚时，我们怎么处理？
""")
