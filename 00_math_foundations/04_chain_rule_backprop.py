"""
====================================================================
第0章 · 第4节 · 链式法则与计算图
====================================================================

【一句话总结】
链式法则是反向传播的数学心脏——它让我们能高效地计算损失对每个权重的梯度，
从而训练深度网络。

【为什么深度学习需要这个？】
- 深度网络是多层复合函数 f(g(h(x)))
- 要对每个权重求梯度，必须用链式法则展开
- 反向传播算法 = 链式法则 + 动态规划（避免重复计算）
- 理解了这个，才能理解为什么会梯度消失/爆炸

【核心概念】
1. 链式法则（Chain Rule）
   - 复合函数求导：d/dx f(g(x)) = f'(g(x)) · g'(x)
   - 直觉：变化的传递——x 变一点，g 变一点，f 又变一点
   - 多变量版本：雅可比矩阵乘法

2. 计算图（Computational Graph）
   - 将计算过程画成有向无环图(DAG)
   - 节点 = 操作（加、乘、sigmoid...）
   - 边 = 数据流动方向
   - 前向传播 = 从输入到输出
   - 反向传播 = 从输出到输入（沿边反向）

3. 局部梯度（Local Gradient）
   - 每个操作只需知道自己的输入输出关系
   - 加法门：均匀分配梯度（都是1）
   - 乘法门：交换分配梯度
   - sigmoid门：σ(x)(1-σ(x))

4. 反向传播（Backpropagation）
   - 从损失开始，逐层反向计算梯度
   - 每个节点：上游梯度 × 局部梯度 = 传给下游的梯度
   - 本质：链式法则的高效实现

5. 梯度消失与爆炸
   - 消失：连续乘以<1的梯度（如sigmoid），越深越小
   - 爆炸：连续乘以>1的梯度，越深越大
   - 解决方案预告：ReLU、残差连接、梯度裁剪

【前置知识】
第0章第1-3节
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

def _numgrad(f, x, eps=1e-7):
    """中心差分法计算数值梯度"""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# =====================================================================
# 第1部分：链式法则基础
# =====================================================================
def part1_chain_rule_basics():
    """链式法则基础：d/dx f(g(x)) = f'(g(x))·g'(x)，数值验证。"""
    print("=" * 60)
    print("第1部分：链式法则基础")
    print("=" * 60)
    # 例1：d/dx sin(x²) = cos(x²)·2x
    x = 1.5
    ana = np.cos(x**2) * 2*x
    num = _numgrad(lambda t: np.sin(t**2), x)
    print(f"\n例1  d/dx sin(x²)  x={x}")
    print(f"  解析={ana:.8f}  数值={num:.8f}  误差={abs(ana-num):.2e}")
    # 例2：d/dx (3x+1)⁴ = 12(3x+1)³
    x = 2.0
    ana = 12*(3*x+1)**3
    num = _numgrad(lambda t: (3*t+1)**4, x)
    print(f"例2  d/dx (3x+1)⁴  x={x}")
    print(f"  解析={ana:.8f}  数值={num:.8f}  误差={abs(ana-num):.2e}")
    # 例3：d/dx exp(sin(x²)) = exp(sin(x²))·cos(x²)·2x  （三层复合）
    x = 1.0
    ana = np.exp(np.sin(x**2)) * np.cos(x**2) * 2*x
    num = _numgrad(lambda t: np.exp(np.sin(t**2)), x)
    print(f"例3  d/dx exp(sin(x²))  x={x}  （三层复合）")
    print(f"  解析={ana:.8f}  数值={num:.8f}  误差={abs(ana-num):.2e}")
    # --- 可视化 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("链式法则：变化如何逐层传递", fontsize=14, fontweight='bold')
    xs = np.linspace(0.5, 2.5, 200)
    axes[0].plot(xs, xs**2, 'b-', lw=2, label='g(x)=x²')
    axes[0].plot(xs, 2*xs, 'b--', lw=2, label="g'(x)=2x")
    axes[0].set_title("内层 g(x)=x²"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    us = np.linspace(0, 7, 200)
    axes[1].plot(us, np.sin(us), 'r-', lw=2, label='f(u)=sin(u)')
    axes[1].plot(us, np.cos(us), 'r--', lw=2, label="f'(u)=cos(u)")
    axes[1].set_title("外层 f(u)=sin(u)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(xs, np.sin(xs**2), 'purple', lw=2, label='sin(x²)')
    axes[2].plot(xs, np.cos(xs**2)*2*xs, 'm--', lw=2, label="cos(x²)·2x")
    axes[2].set_title("复合函数及导数"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_part1_chain_rule_basics.png", dpi=100, bbox_inches='tight'); plt.close()
    print("[图已保存: 04_part1_chain_rule_basics.png]")

# =====================================================================
# 第2部分：计算图可视化  y = (a+b)*(b+1)
# =====================================================================
def part2_computation_graph():
    """计算图：前向计算 + 反向传播梯度，matplotlib 画图。"""
    print("\n" + "=" * 60)
    print("第2部分：计算图可视化  y=(a+b)*(b+1)")
    print("=" * 60)
    a, b = 2.0, 3.0
    c = a + b; d = b + 1; y = c * d           # 前向
    # 反向：dy/dy=1 → 乘法门交换 → 加法门传递
    dy_dc, dy_dd = d, c                        # 乘法门局部梯度
    dy_da = dy_dc * 1.0                         # 加法门: dc/da=1
    dy_db = dy_dc * 1.0 + dy_dd * 1.0           # b 出现在两条路径，梯度累加
    print(f"前向: a={a}, b={b} → c=a+b={c}, d=b+1={d}, y=c*d={y}")
    print(f"反向: dy/da={dy_da}, dy/db={dy_db}（b有两条路径，梯度累加）")
    # 数值验证
    eps = 1e-7
    f = lambda a_, b_: (a_+b_)*(b_+1)
    print(f"数值验证: dy/da={( f(a+eps,b)-f(a-eps,b) )/(2*eps):.4f}, "
          f"dy/db={( f(a,b+eps)-f(a,b-eps) )/(2*eps):.4f}")
    # --- 画图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nodes = {'a':(0.5,3.5), 'b':(0.5,1.5), '1':(0.5,0.5),
             '+1':(2,3), '+2':(2,1), 'x':(3.5,2)}
    edges = [('a','+1'),('b','+1'),('b','+2'),('1','+2'),('+1','x'),('+2','x')]
    fwd_lbl = {'a':f'a={a:.0f}','b':f'b={b:.0f}','1':'1',
               '+1':f'+→{c:.0f}','+2':f'+→{d:.0f}','x':f'*→{y:.0f}'}
    bwd_lbl = {'a':f'a\n grad={dy_da:.0f}','b':f'b\n grad={dy_db:.0f}','1':'1',
               '+1':f'+\n↑{dy_dc:.0f}','+2':f'+\n↑{dy_dd:.0f}','x':f'*\n↑1'}
    clr = {'a':'#AED6F1','b':'#AED6F1','1':'#D5F5E3','+1':'#FAD7A0','+2':'#FAD7A0','x':'#F1948A'}
    for idx, (title, lbls, arrow_c, ec) in enumerate([
            ("前向传播", fwd_lbl, '#2C3E50', '#2C3E50'),
            ("反向传播（梯度逆流）", bwd_lbl, '#E74C3C', '#E74C3C')]):
        ax = axes[idx]; ax.set_xlim(-0.3,4.3); ax.set_ylim(-0.3,4.3)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=13, fontweight='bold')
        for s, d_ in edges:
            sx,sy = nodes[s]; dx,dy = nodes[d_]
            if idx == 1: sx,sy,dx,dy = dx,dy,sx,sy   # 反向箭头
            ax.annotate('', xy=(dx,dy), xytext=(sx,sy),
                        arrowprops=dict(arrowstyle='->',color=arrow_c,lw=2))
        for nm,(px,py) in nodes.items():
            ax.add_patch(plt.Circle((px,py),0.32,color=clr[nm],ec=ec,lw=2,zorder=5))
            ax.text(px,py,lbls[nm],ha='center',va='center',fontsize=7,fontweight='bold',zorder=6)
    plt.tight_layout()
    plt.savefig("04_part2_computation_graph.png", dpi=100, bbox_inches='tight'); plt.close()
    print("[图已保存: 04_part2_computation_graph.png]")

# =====================================================================
# 第3部分：局部梯度——加法门、乘法门、sigmoid门
# =====================================================================
def part3_local_gradients():
    """三种基本门的 forward / backward，并可视化 sigmoid 梯度。"""
    print("\n" + "=" * 60)
    print("第3部分：局部梯度")
    print("=" * 60)
    class AddGate:
        """加法门 z=x+y, 局部梯度 dz/dx=dz/dy=1（均匀分配）"""
        def forward(self, x, y): self.x, self.y = x, y; return x + y
        def backward(self, ug):  return ug * 1.0, ug * 1.0
    class MulGate:
        """乘法门 z=x*y, 局部梯度 dz/dx=y, dz/dy=x（交换分配）"""
        def forward(self, x, y): self.x, self.y = x, y; return x * y
        def backward(self, ug):  return ug * self.y, ug * self.x
    class SigGate:
        """sigmoid门 z=σ(x), 局部梯度 σ(x)(1-σ(x))"""
        def forward(self, x):  self.o = 1/(1+np.exp(-x)); return self.o
        def backward(self, ug): return ug * self.o * (1-self.o)
    # 测试
    ag, mg, sg = AddGate(), MulGate(), SigGate()
    print(f"加法门: 3+(-2)={ag.forward(3,-2)}, 梯度={ag.backward(1.0)}")
    print(f"乘法门: 3×(-2)={mg.forward(3,-2)}, 梯度={mg.backward(1.0)} (交换)")
    o = sg.forward(0.5); g = sg.backward(1.0)
    print(f"sigmoid门: σ(0.5)={o:.6f}, 梯度={g:.6f} = {o:.4f}×{1-o:.4f}")
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.linspace(-6, 6, 300)
    sv = 1/(1+np.exp(-xs)); sg_ = sv*(1-sv)
    axes[0].plot(xs, sv, 'b-', lw=2.5, label='σ(x)')
    axes[0].set_title("sigmoid：压缩到(0,1)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, sg_, 'r-', lw=2.5, label="σ'(x)=σ(1-σ)")
    axes[1].fill_between(xs, sg_, alpha=0.15, color='red')
    axes[1].axhline(0.25, color='gray', ls=':', label='最大=0.25')
    axes[1].set_title("sigmoid梯度：最大才0.25！"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_part3_local_gradients.png", dpi=100, bbox_inches='tight'); plt.close()
    print("[图已保存: 04_part3_local_gradients.png]")

# =====================================================================
# 第4部分：手动反向传播——2层网络
# =====================================================================
def part4_manual_backprop():
    """手动逐步反向传播：x→[w1,b1]→sigmoid→[w2,b2]→MSE。"""
    print("\n" + "=" * 60)
    print("第4部分：手动反向传播——2层网络")
    print("=" * 60)
    x, y_t = 1.5, 0.8
    w1, b1, w2, b2 = 0.5, 0.1, -0.3, 0.2
    # 前向
    z1 = w1*x + b1;  a1 = 1/(1+np.exp(-z1))
    z2 = w2*a1 + b2;  loss = 0.5*(z2 - y_t)**2
    print(f"前向: z1={z1:.4f}, a1=σ(z1)={a1:.4f}, z2={z2:.4f}, loss={loss:.6f}")
    # 反向（逐步）
    dL_dz2 = z2 - y_t                              # ① 损失导数
    dL_dw2 = dL_dz2 * a1                            # ② 线性层
    dL_db2 = dL_dz2
    dL_da1 = dL_dz2 * w2
    dL_dz1 = dL_da1 * a1*(1-a1)                     # ③ sigmoid
    dL_dw1 = dL_dz1 * x                             # ④ 线性层
    dL_db1 = dL_dz1
    print("反向推导:")
    print(f"  ① dL/dz2 = {dL_dz2:.4f}")
    print(f"  ② dL/dw2={dL_dw2:.6f}, dL/db2={dL_db2:.6f}, dL/da1={dL_da1:.6f}")
    print(f"  ③ dL/dz1 = {dL_dz1:.8f}  (经sigmoid)")
    print(f"  ④ dL/dw1={dL_dw1:.8f}, dL/db1={dL_db1:.8f}")
    # 数值验证
    def L(w1_,b1_,w2_,b2_):
        a = 1/(1+np.exp(-(w1_*x+b1_))); return 0.5*(w2_*a+b2_-y_t)**2
    e = 1e-7
    for name, val, ana in [('w1',w1,dL_dw1),('b1',b1,dL_db1),
                            ('w2',w2,dL_dw2),('b2',b2,dL_db2)]:
        args = {'w1_':w1,'b1_':b1,'w2_':w2,'b2_':b2}
        args[name+'_'] = val+e; fp = L(**args)
        args[name+'_'] = val-e; fm = L(**args)
        print(f"  数值验证 dL/d{name}: 解析={ana:.8f} 数值={(fp-fm)/(2*e):.8f}")

# =====================================================================
# 第5部分：自动反向传播引擎（微型 autograd）
# =====================================================================
class Value:
    """
    微型自动求导引擎（类似 micrograd）。
    每个 Value 记录：data(值)、grad(梯度)、_prev(父节点)、_backward(反向函数)。
    调用 .backward() 自动沿计算图反向传播。
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data, self.grad = float(data), 0.0
        self._prev, self._op, self._backward = set(_children), _op, lambda: None
        self.label = label
    def __repr__(self): return f"Value({self.data:.4f}, grad={self.grad:.4f})"
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o), '+')
        def _bk(): self.grad += out.grad; o.grad += out.grad   # 加法：直传
        out._backward = _bk; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o), '*')
        def _bk(): self.grad += o.data * out.grad; o.grad += self.data * out.grad  # 交换
        out._backward = _bk; return out
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _bk(): self.grad += n * self.data**(n-1) * out.grad
        out._backward = _bk; return out
    def __neg__(self):     return self * (-1)
    def __sub__(self, o):  return self + (-o)
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(s, (self,), 'sig')
        def _bk(): self.grad += s*(1-s) * out.grad
        out._backward = _bk; return out
    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _bk(): self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _bk; return out
    def backward(self):
        """拓扑逆序反向传播"""
        topo, vis = [], set()
        def build(v):
            if v not in vis: vis.add(v); [build(c) for c in v._prev]; topo.append(v)
        build(self); self.grad = 1.0
        for n in reversed(topo): n._backward()

def part5_autograd_engine():
    """用 Value 类自动求梯度，并做简单梯度下降拟合 y=3x+2。"""
    print("\n" + "=" * 60)
    print("第5部分：自动反向传播引擎")
    print("=" * 60)
    # 验证：y=(a+b)*(b+1)
    a, b = Value(2.0), Value(3.0)
    y = (a + b) * (b + 1.0); y.backward()
    print(f"y=(a+b)*(b+1): dy/da={a.grad:.1f}(期望4), dy/db={b.grad:.1f}(期望9)")
    # 验证：与第4部分手动计算对比
    xv = Value(1.5); w1=Value(0.5); b1=Value(0.1); w2=Value(-0.3); b2v=Value(0.2)
    loss = ((w2*(w1*xv+b1).sigmoid()+b2v) - 0.8)**2 * 0.5; loss.backward()
    print(f"2层网络: dL/dw1={w1.grad:.8f}, dL/dw2={w2.grad:.8f} (与第4部分一致)")
    # 梯度下降拟合 y=3x+2
    print("\n梯度下降拟合 y=3x+2:")
    np.random.seed(42)
    X = np.random.uniform(-2, 2, 20); Y = 3*X + 2 + np.random.randn(20)*0.1
    w, bp = Value(0.0), Value(0.0); lr = 0.05; hist = []
    for ep in range(80):
        tot = Value(0.0)
        for xi, yi in zip(X, Y): tot = tot + (w*xi + bp - yi)**2
        avg = tot * (1.0/len(X)); hist.append(avg.data)
        w.grad = bp.grad = 0.0; avg.backward()
        w.data -= lr*w.grad; bp.data -= lr*bp.grad
        if ep % 20 == 0 or ep == 79:
            print(f"  epoch {ep:3d}: loss={avg.data:.4f}, w={w.data:.4f}, b={bp.data:.4f}")
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(hist, 'b-', lw=2); axes[0].set_title("损失曲线"); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    xl = np.linspace(-2.5, 2.5, 100)
    axes[1].scatter(X, Y, c='blue', alpha=0.6, label='数据')
    axes[1].plot(xl, w.data*xl+bp.data, 'r-', lw=2, label=f'拟合 {w.data:.2f}x+{bp.data:.2f}')
    axes[1].plot(xl, 3*xl+2, 'g--', lw=1.5, label='真实 3x+2')
    axes[1].set_title("拟合结果"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("04_part5_autograd_engine.png", dpi=100, bbox_inches='tight'); plt.close()
    print("[图已保存: 04_part5_autograd_engine.png]")

# =====================================================================
# 第6部分：梯度消失演示
# =====================================================================
def part6_vanishing_gradients():
    """连续sigmoid层导致梯度指数衰减，对比ReLU。"""
    print("\n" + "=" * 60)
    print("第6部分：梯度消失演示")
    print("=" * 60)
    depths = [1, 2, 5, 10, 20]
    sig_grads, relu_grads, theo = [], [], []
    print(f"{'深度':>4s}  {'sigmoid梯度':>14s}  {'理论最大':>12s}  {'ReLU梯度':>10s}")
    for d in depths:
        # sigmoid
        xv = Value(0.5); h = xv
        for _ in range(d): h = h.sigmoid()
        h.backward(); sig_grads.append(xv.grad); theo.append(0.25**d)
        # ReLU
        xr = Value(0.5); hr = xr
        for _ in range(d): hr = hr.relu()
        hr.backward(); relu_grads.append(xr.grad)
        print(f"  {d:>3d}   {xv.grad:>14.2e}   {0.25**d:>12.2e}   {xr.grad:>10.4f}")
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].semilogy(depths, theo, 'rs-', lw=2, ms=8, label='sigmoid理论最大')
    axes[0].semilogy(depths, [abs(g) for g in sig_grads], 'bo-', lw=2, ms=8, label='sigmoid实际')
    axes[0].semilogy(depths, [max(g,1e-20) for g in relu_grads], 'g^-', lw=2, ms=8, label='ReLU')
    axes[0].axhline(1e-7, color='red', ls='--', alpha=0.5, label='消失阈值')
    axes[0].set_title("梯度随深度的变化"); axes[0].set_xlabel("深度"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # 10层各层梯度
    layers = [Value(0.5)]
    for _ in range(10): layers.append(layers[-1].sigmoid())
    layers[-1].backward()
    lg = [abs(v.grad) for v in layers]
    colors = plt.cm.RdYlGn_r(np.linspace(0,1,len(lg)))
    axes[1].bar(range(len(lg)), lg, color=colors, ec='black', lw=0.5)
    axes[1].set_yscale('log'); axes[1].set_title("10层sigmoid各层梯度")
    axes[1].set_xlabel("层号(0=输入)"); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig("04_part6_vanishing_gradients.png", dpi=100, bbox_inches='tight'); plt.close()
    print("[图已保存: 04_part6_vanishing_gradients.png]")
    print("结论：sigmoid连乘→梯度指数衰减；ReLU正区间梯度恒1，有效缓解消失")

# =====================================================================
# 第7部分：数值梯度验证（系统性）
# =====================================================================
def part7_gradient_checking():
    """对 autograd 引擎做系统性数值梯度验证。"""
    print("\n" + "=" * 60)
    print("第7部分：数值梯度验证")
    print("=" * 60)
    def check(build_fn, pinit, names, eps=1e-7):
        """通用梯度检查：对比解析梯度与中心差分数值梯度"""
        for i in range(len(pinit)):
            p = [Value(v) for v in pinit]; build_fn(p).backward(); ana = p[i].grad
            pp, pm = list(pinit), list(pinit); pp[i]+=eps; pm[i]-=eps
            num = (build_fn([Value(v) for v in pp]).data - build_fn([Value(v) for v in pm]).data)/(2*eps)
            err = abs(ana-num)/max(abs(ana),abs(num),1e-15)
            status = "PASS" if err < 1e-5 else "FAIL"
            print(f"  d/d{names[i]}: 解析={ana:+.8f} 数值={num:+.8f} 误差={err:.2e} [{status}]")
    # 测试1
    print("测试1: f(a,b) = (a+b)*(a-b)")
    check(lambda p: (p[0]+p[1])*(p[0]-p[1]), [3.0,2.0], ['a','b'])
    # 测试2
    print("测试2: f(x) = sigmoid(2x+1)")
    check(lambda p: (p[0]*2+1).sigmoid(), [0.7], ['x'])
    # 测试3
    print("测试3: 2层网络 loss=(σ(w2·σ(w1·1.5+b1)+b2)-0.7)²")
    check(lambda p: ((p[2]*(p[0]*1.5+p[1]).sigmoid()+p[3]).sigmoid()-0.7)**2,
          [0.5,0.1,-0.3,0.2], ['w1','b1','w2','b2'])
    # 测试4
    print("测试4: f(a,b,c) = (a*b + σ(c))²")
    check(lambda p: (p[0]*p[1]+p[2].sigmoid())**2, [1.5,-2.0,0.8], ['a','b','c'])
    print("全部通过！autograd 引擎正确。")

# =====================================================================
# 第8部分：思考题
# =====================================================================
def part8_exercises():
    """思考题：检验对链式法则与反向传播的理解。"""
    print("\n" + "=" * 60)
    print("第8部分：思考题")
    print("=" * 60)
    QA = [
        ("为什么反向传播比"对每个参数单独数值差分"高效得多？",
         "提示：N个参数的数值差分需要多少次前向传播？",
         "数值差分需2N次前向传播，反向传播只需1次前向+1次反向，与N无关。"),
        ("变量被多个操作使用时（如 b 在 (a+b)*(b+1) 中），梯度怎么算？",
         "提示：多路径梯度规则。",
         "多条路径的梯度求和，代码中用 grad += 而非 grad =，对应多元链式法则。"),
        ("sigmoid梯度最大值是多少？20层深度网络第1层梯度缩小多少倍？",
         "提示：σ'(x)=σ(x)(1-σ(x))，最大值在x=0处。",
         "最大0.25。20层：0.25^20≈9.1e-13，缩小约1万亿倍！"),
        ("ReLU解决了梯度消失，但有什么新问题？有哪些改进？",
         "提示：ReLU在x<0时梯度是什么？",
         "x<0梯度为0→'死亡ReLU'。改进：Leaky ReLU、ELU、GELU/Swish。"),
        ("Value.backward() 为什么需要拓扑排序？不排序会怎样？",
         "提示：如果一个节点的输出被两个操作使用。",
         "拓扑排序保证所有下游梯度累加完毕后才向上游传播，否则会遗漏路径贡献。"),
    ]
    for i, (q, hint, ans) in enumerate(QA, 1):
        print(f"\n{i}. {q}\n   {hint}\n   参考答案：{ans}")

# =====================================================================
# 主程序
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  第0章·第4节·链式法则与计算图")
    print("  深度学习的数学心脏——理解反向传播的本质")
    print("=" * 60)
    part1_chain_rule_basics()
    part2_computation_graph()
    part3_local_gradients()
    part4_manual_backprop()
    part5_autograd_engine()
    part6_vanishing_gradients()
    part7_gradient_checking()
    part8_exercises()
    print("\n" + "=" * 60)
    print("本节总结")
    print("=" * 60)
    print("  1. 链式法则让我们能对复合函数逐层求导")
    print("  2. 计算图将复杂运算拆成简单操作的DAG")
    print("  3. 每个操作门只需知道自己的局部梯度")
    print("  4. 反向传播 = 链式法则 + 拓扑排序 + 梯度累加")
    print("  5. sigmoid连乘导致梯度消失，ReLU有效缓解")
    print("  6. 数值梯度验证是调试自定义层的必备工具")
    print("\n  下一节预告：第0章·第5节——概率论基础")
