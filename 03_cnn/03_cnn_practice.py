"""
====================================================================
第3章 · 第3节 · 实战：手写数字识别
====================================================================

【一句话总结】
用前面学到的所有组件（卷积、池化、全连接、ReLU、交叉熵、Adam）
搭建一个完整的 CNN，在手写数字数据集上训练，达到 90%+ 准确率。

【为什么需要这个实战？】
- 把前面的零散知识整合成一个完整的项目
- 体验真实的深度学习工作流：数据处理→模型定义→训练→评估
- 纯 NumPy 实现的 CNN 能达到什么水平？
- 为后面用 PyTorch 做对比打下基础

【核心概念】

1. 数据准备
   - 我们自己生成一个简化版手写数字数据集（0-9）
   - 数据格式：(N, 1, H, W) — batch×通道×高×宽
   - 归一化到 [0, 1]
   - 划分训练集/测试集

2. 模型架构
   - Conv(1→8, 3×3) → ReLU → MaxPool(2×2)
   - Conv(8→16, 3×3) → ReLU → MaxPool(2×2)
   - Flatten → FC(256→10) → Softmax
   - 总参数量分析

3. 训练流程
   - 前向传播 → 损失计算 → 反向传播 → 参数更新
   - Mini-batch 训练
   - 监控训练/测试准确率和损失

4. 可视化分析
   - 卷积核学到了什么（可视化滤波器权重）
   - 特征图是什么样子（中间层输出）
   - 混淆矩阵：哪些数字最容易混淆

【前置知识】
第3章第1-2节，第2章第3-4节
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

# ====================================================================
# 第一部分：生成简化版手写数字数据集（8×8 画布）
# ====================================================================

def _draw_digit_template(digit):
    """返回 8×8 数字模板（二值图），手动设定像素画出 0-9。"""
    g = np.zeros((8, 8))
    if digit == 0:
        g[1,2:6]=1; g[6,2:6]=1; g[2:6,1]=1; g[2:6,6]=1
    elif digit == 1:
        g[1:7,4]=1; g[1,3]=1; g[7,3:6]=1
    elif digit == 2:
        g[1,2:6]=1; g[1:4,6]=1; g[4,2:6]=1; g[4:7,1]=1; g[7,2:7]=1
    elif digit == 3:
        g[1,2:6]=1; g[1:4,6]=1; g[4,2:6]=1; g[4:7,6]=1; g[7,2:6]=1
    elif digit == 4:
        g[1:4,1]=1; g[1:4,6]=1; g[4,1:7]=1; g[4:7,6]=1
    elif digit == 5:
        g[1,1:7]=1; g[1:4,1]=1; g[4,1:7]=1; g[4:7,6]=1; g[7,1:6]=1
    elif digit == 6:
        g[1,2:6]=1; g[1:4,1]=1; g[4,2:6]=1; g[4:7,1]=1; g[4:7,6]=1; g[7,2:6]=1
    elif digit == 7:
        g[1,1:7]=1; g[1:7,5]=1
    elif digit == 8:
        g[1,2:6]=1; g[4,2:6]=1; g[7,2:6]=1; g[1:4,1]=1; g[1:4,6]=1; g[4:7,1]=1; g[4:7,6]=1
    elif digit == 9:
        g[1,2:6]=1; g[1:4,1]=1; g[1:4,6]=1; g[4,2:6]=1; g[4:7,6]=1; g[7,2:6]=1
    return g

def generate_digit_dataset(n_per_class=150, sz=8, noise_std=0.15):
    """生成数据集：模板→随机位移→随机粗细→高斯噪声→裁剪 [0,1]。"""
    imgs, labs = [], []
    for d in range(10):
        tpl = _draw_digit_template(d)
        for _ in range(n_per_class):
            dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)
            s = np.zeros((sz, sz))
            sy=slice(max(0,-dy),min(sz,sz-dy)); sx=slice(max(0,-dx),min(sz,sz-dx))
            ty=slice(max(0,dy),min(sz,sz+dy)); tx=slice(max(0,dx),min(sz,sz+dx))
            s[ty,tx] = tpl[sy,sx]
            if np.random.rand() > 0.5:  # 随机粗细变化
                ys, xs = np.where(s > 0.5)
                for _ in range(max(1, len(ys)//4)):
                    k = np.random.randint(max(1,len(ys)))
                    ey, ex = ys[k]+np.random.randint(-1,2), xs[k]+np.random.randint(-1,2)
                    if 0<=ey<sz and 0<=ex<sz: s[ey,ex]=1.0
            imgs.append(np.clip(s + np.random.randn(sz,sz)*noise_std, 0, 1))
            labs.append(d)
    imgs = np.array(imgs)[:,None,:,:]  # (N,1,8,8)
    labs = np.array(labs, dtype=np.int32)
    p = np.random.permutation(len(labs))
    return imgs[p], labs[p]

def show_sample_digits(images, labels):
    """展示 40 张样本图片"""
    fig, axes = plt.subplots(4, 10, figsize=(12, 5))
    fig.suptitle("生成的手写数字样本（8x8）", fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < 40: ax.imshow(images[i,0], cmap="gray_r", vmin=0, vmax=1); ax.set_title(str(labels[i]), fontsize=9)
        ax.axis("off")
    plt.tight_layout(); plt.savefig("digit_samples.png", dpi=100); plt.show()

# ====================================================================
# 第二部分：CNN 组件（紧凑复用版）
# ====================================================================

class Conv2D:
    """二维卷积层（含 He 初始化与 Adam 状态）"""
    def __init__(self, ci, co, k=3, s=1, p=1):
        self.s, self.p, self.x = s, p, None
        self.W = np.random.randn(co,ci,k,k)*np.sqrt(2.0/(ci*k*k))
        self.b = np.zeros(co)
        self.dW=np.zeros_like(self.W); self.db=np.zeros_like(self.b)
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
    def forward(self, x):
        self.x = x; N,C,H,W = x.shape; F,_,kH,kW = self.W.shape; st=self.s
        xp = np.pad(x, ((0,0),(0,0),(self.p,self.p),(self.p,self.p)))
        oH,oW = (H+2*self.p-kH)//st+1, (W+2*self.p-kW)//st+1
        out = np.zeros((N,F,oH,oW))
        for n in range(N):
            for f in range(F):
                for i in range(oH):
                    for j in range(oW):
                        out[n,f,i,j] = np.sum(xp[n,:,i*st:i*st+kH,j*st:j*st+kW]*self.W[f]) + self.b[f]
        return out
    def backward(self, d):
        N,C,H,W = self.x.shape; F,_,kH,kW = self.W.shape; st=self.s
        xp = np.pad(self.x, ((0,0),(0,0),(self.p,self.p),(self.p,self.p)))
        dxp = np.zeros_like(xp); self.dW[:]=0; self.db[:]=0
        _,_,oH,oW = d.shape
        for n in range(N):
            for f in range(F):
                for i in range(oH):
                    for j in range(oW):
                        self.dW[f] += d[n,f,i,j]*xp[n,:,i*st:i*st+kH,j*st:j*st+kW]
                        self.db[f] += d[n,f,i,j]
                        dxp[n,:,i*st:i*st+kH,j*st:j*st+kW] += d[n,f,i,j]*self.W[f]
        return dxp[:,:,self.p:-self.p,self.p:-self.p] if self.p>0 else dxp
    def params_count(self): return self.W.size + self.b.size

class MaxPool2D:
    """2x2 最大池化"""
    def __init__(self, k=2, s=2): self.k,self.s,self.mask = k,s,None
    def forward(self, x):
        N,C,H,W = x.shape; k,s = self.k,self.s; oH,oW = H//s, W//s
        out = np.zeros((N,C,oH,oW)); self.mask = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                for i in range(oH):
                    for j in range(oW):
                        r = x[n,c,i*s:i*s+k,j*s:j*s+k]; mv = np.max(r)
                        out[n,c,i,j] = mv; self.mask[n,c,i*s:i*s+k,j*s:j*s+k] = (r==mv)
        return out
    def backward(self, d):
        N,C,oH,oW = d.shape; k,s = self.k,self.s; dx = np.zeros_like(self.mask)
        for n in range(N):
            for c in range(C):
                for i in range(oH):
                    for j in range(oW):
                        dx[n,c,i*s:i*s+k,j*s:j*s+k] += d[n,c,i,j]*self.mask[n,c,i*s:i*s+k,j*s:j*s+k]
        return dx

class ReLU:
    def __init__(self): self.m = None
    def forward(self, x): self.m = (x>0).astype(float); return x*self.m
    def backward(self, d): return d*self.m

class Flatten:
    def __init__(self): self.sh = None
    def forward(self, x): self.sh = x.shape; return x.reshape(x.shape[0], -1)
    def backward(self, d): return d.reshape(self.sh)

class Linear:
    """全连接层（含 Adam 状态）"""
    def __init__(self, fi, fo):
        self.W = np.random.randn(fi,fo)*np.sqrt(2.0/fi); self.b = np.zeros(fo); self.x = None
        self.dW=np.zeros_like(self.W); self.db=np.zeros_like(self.b)
        self.mW=np.zeros_like(self.W); self.vW=np.zeros_like(self.W)
        self.mb=np.zeros_like(self.b); self.vb=np.zeros_like(self.b)
    def forward(self, x): self.x = x; return x@self.W + self.b
    def backward(self, d):
        self.dW = self.x.T@d; self.db = d.sum(axis=0); return d@self.W.T
    def params_count(self): return self.W.size + self.b.size

class SoftmaxCrossEntropy:
    """Softmax + 交叉熵（合并计算更稳定，梯度 = softmax - one_hot）"""
    def __init__(self): self.pr = self.lb = None
    def forward(self, z, y):
        self.lb = y; e = np.exp(z - z.max(axis=1, keepdims=True))
        self.pr = e / e.sum(axis=1, keepdims=True)
        return np.mean(-np.log(self.pr[np.arange(len(y)), y] + 1e-12))
    def backward(self):
        N = len(self.lb); dx = self.pr.copy(); dx[np.arange(N), self.lb] -= 1; return dx/N

# ====================================================================
# 第三部分：Adam 优化器
# ====================================================================

def adam_update(layer, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, t=1):
    """Adam：一阶矩→二阶矩→偏差矫正→更新参数"""
    for n in ['W','b']:
        p,g = getattr(layer,n), getattr(layer,'d'+n)
        m,v = getattr(layer,'m'+n), getattr(layer,'v'+n)
        m[:] = b1*m+(1-b1)*g; v[:] = b2*v+(1-b2)*(g**2)
        p -= lr*(m/(1-b1**t)) / (np.sqrt(v/(1-b2**t))+eps)

# ====================================================================
# 第四部分：完整 CNN 模型
# ====================================================================
# 架构：输入(N,1,8,8) → Conv(1→8) → ReLU → Pool → (N,8,4,4)
#       → Conv(8→16) → ReLU → Pool → (N,16,2,2) → Flatten(64) → FC(10)

class SimpleCNN:
    """完整 CNN：两个卷积块 + 全连接分类头"""
    def __init__(self):
        self.conv1=Conv2D(1,8); self.relu1=ReLU(); self.pool1=MaxPool2D()
        self.conv2=Conv2D(8,16); self.relu2=ReLU(); self.pool2=MaxPool2D()
        self.flat=Flatten(); self.fc=Linear(16*2*2, 10)
        self.loss_fn = SoftmaxCrossEntropy()
        self.trainable = [self.conv1, self.conv2, self.fc]
        self.layers = [self.conv1,self.relu1,self.pool1,
                       self.conv2,self.relu2,self.pool2, self.flat,self.fc]
        self.fmaps = {}  # 缓存中间特征图

    def forward(self, x, save=False):
        for i,l in enumerate(self.layers):
            x = l.forward(x)
            if save: self.fmaps[i] = x.copy()
        return x

    def backward(self):
        d = self.loss_fn.backward()
        for l in reversed(self.layers): d = l.backward(d)

    def update(self, lr=0.001, t=1):
        for l in self.trainable: adam_update(l, lr=lr, t=t)

    def params_summary(self):
        """打印每层参数量"""
        print("\n" + "="*50 + "\n模型参数量汇总\n" + "="*50)
        total = 0
        for nm,ly in [("Conv1(1->8,3x3)",self.conv1),("Conv2(8->16,3x3)",self.conv2),("FC(64->10)",self.fc)]:
            c = ly.params_count(); total += c; print(f"  {nm:22s} {c:>6d}")
        # Conv1: 8*1*3*3+8=80  Conv2: 16*8*3*3+16=1168  FC: 64*10+10=650
        print(f"  {'总计':22s} {total:>6d}\n" + "="*50)
        return total

# ====================================================================
# 第五部分：训练循环
# ====================================================================

def _predict_batched(model, X, bs=64):
    parts = [np.argmax(model.forward(X[i:i+bs]),axis=1) for i in range(0,len(X),bs)]
    return np.concatenate(parts)

def train(model, Xtr, ytr, Xte, yte, epochs=15, bs=32, lr=0.002):
    """完整训练：打乱→mini-batch→前向→损失→反向→Adam→评估"""
    N = len(ytr); t_step = 0
    hist = {"train_loss":[],"test_loss":[],"train_acc":[],"test_acc":[]}
    print(f"\n{'='*60}\n开始训练  样本:{N} 测试:{len(yte)} batch:{bs}\n{'='*60}")

    for ep in range(1, epochs+1):
        pm = np.random.permutation(N); Xs,ys = Xtr[pm], ytr[pm]
        eloss, nb = 0.0, 0
        for i in range(0, N, bs):
            xb,yb = Xs[i:i+bs], ys[i:i+bs]
            logits = model.forward(xb)
            eloss += model.loss_fn.forward(logits, yb)
            model.backward(); t_step += 1; model.update(lr=lr, t=t_step); nb += 1
        al = eloss/nb
        tra = np.mean(_predict_batched(model, Xtr)==ytr)
        # 测试集损失和准确率
        te_lg = np.concatenate([model.forward(Xte[i:i+64]) for i in range(0,len(yte),64)])
        e = np.exp(te_lg - te_lg.max(axis=1,keepdims=True))
        pr = e/e.sum(axis=1,keepdims=True)
        tl = np.mean(-np.log(pr[np.arange(len(yte)),yte]+1e-12))
        ta = np.mean(np.argmax(te_lg,axis=1)==yte)
        hist["train_loss"].append(al); hist["test_loss"].append(tl)
        hist["train_acc"].append(tra); hist["test_acc"].append(ta)
        print(f"  Epoch {ep:2d}/{epochs}  训练:{al:.4f}/{tra:.1%}  测试:{tl:.4f}/{ta:.1%}")
    print(f"{'='*60}\n训练完成！测试准确率: {hist['test_acc'][-1]:.1%}")
    return hist

# ====================================================================
# 第六部分：评估与可视化
# ====================================================================

def plot_training_history(hist):
    """绘制损失和准确率曲线"""
    fig,(a1,a2) = plt.subplots(1,2,figsize=(13,5))
    ep = range(1, len(hist["train_loss"])+1)
    a1.plot(ep,hist["train_loss"],"b-o",ms=4,label="训练"); a1.plot(ep,hist["test_loss"],"r-s",ms=4,label="测试")
    a1.set(xlabel="Epoch",ylabel="交叉熵损失",title="损失曲线"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(ep,hist["train_acc"],"b-o",ms=4,label="训练"); a2.plot(ep,hist["test_acc"],"r-s",ms=4,label="测试")
    a2.set(xlabel="Epoch",ylabel="准确率",title="准确率曲线",ylim=(0,1.05)); a2.legend(); a2.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("training_curves.png",dpi=100); plt.show()

def confusion_matrix(yt, yp, nc=10):
    """混淆矩阵：cm[i][j] = 真实 i 预测 j 的样本数"""
    cm = np.zeros((nc,nc),dtype=int)
    for t,p in zip(yt,yp): cm[t,p]+=1
    return cm

def plot_confusion_matrix(cm):
    """可视化混淆矩阵"""
    fig,ax = plt.subplots(figsize=(8,7)); im = ax.imshow(cm,cmap="Blues")
    ax.set(xticks=range(10),yticks=range(10),xlabel="预测",ylabel="真实",title="混淆矩阵")
    for i in range(10):
        for j in range(10):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=10,
                    color="white" if cm[i,j]>cm.max()*.5 else "black")
    plt.colorbar(im,ax=ax); plt.tight_layout(); plt.savefig("confusion_matrix.png",dpi=100); plt.show()

def visualize_filters(model):
    """可视化第一层卷积核（训练后通常学到边缘、角点、笔画检测器）"""
    W = model.conv1.W  # (8,1,3,3)
    fig,axes = plt.subplots(2,4,figsize=(10,5)); fig.suptitle("Conv1 的 8 个 3x3 滤波器",fontsize=13)
    for i,ax in enumerate(axes.flat):
        k = W[i,0]; im = ax.imshow(k,cmap="RdBu_r",vmin=-1,vmax=1)
        ax.set_title(f"滤波器{i}"); ax.axis("off")
        for r in range(3):
            for c in range(3):
                ax.text(c,r,f"{k[r,c]:.2f}",ha="center",va="center",fontsize=7)
    plt.colorbar(im,ax=axes,shrink=0.6); plt.tight_layout()
    plt.savefig("learned_filters.png",dpi=100); plt.show()

def visualize_feature_maps(model, img, label):
    """可视化一张图片通过各层后的特征图"""
    model.forward(img[None], save=True)
    fig,axes = plt.subplots(3,9,figsize=(15,6))
    fig.suptitle(f"特征图（输入数字:{label}）",fontsize=13)
    axes[0,0].imshow(img[0],cmap="gray_r"); axes[0,0].set_title("输入",fontsize=8)
    for c in range(8): axes[0,c+1].imshow(model.fmaps[0][0,c],cmap="viridis"); axes[0,c+1].set_title(f"C1-{c}",fontsize=7)
    axes[1,0].set_visible(False)
    for c in range(8): axes[1,c+1].imshow(model.fmaps[2][0,c],cmap="viridis"); axes[1,c+1].set_title(f"P1-{c}",fontsize=7)
    axes[2,0].set_visible(False)
    for c in range(8): axes[2,c+1].imshow(model.fmaps[3][0,c],cmap="viridis"); axes[2,c+1].set_title(f"C2-{c}",fontsize=7)
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout(); plt.savefig("feature_maps.png",dpi=100); plt.show()

def show_predictions(model, Xt, yt, n=20):
    """展示预测结果，红色标出错误"""
    pr = _predict_batched(model, Xt[:n])
    fig,axes = plt.subplots(2,10,figsize=(14,3.5)); fig.suptitle("预测结果（红=错误）",fontsize=13)
    for i,ax in enumerate(axes.flat):
        if i<n:
            ax.imshow(Xt[i,0],cmap="gray_r",vmin=0,vmax=1)
            ok = pr[i]==yt[i]
            ax.set_title(f"真:{yt[i]}预:{pr[i]}",fontsize=8,color="green" if ok else "red",fontweight="bold")
        ax.axis("off")
    plt.tight_layout(); plt.savefig("predictions.png",dpi=100); plt.show()

# ====================================================================
# 第七部分：参数量分析
# ====================================================================

def detailed_param_analysis():
    """详细拆解每层参数量 —— 面试和论文常考技能"""
    print(f"\n{'='*60}\n详细参数量分析\n{'='*60}")
    print("""
    层                  权重形状            参数量
    ──────────────────────────────────────────────
    Conv1 (1->8, 3x3)  W:(8,1,3,3)  b:(8,)      80
    Conv2 (8->16,3x3)  W:(16,8,3,3) b:(16,)   1168
    FC    (64->10)      W:(64,10)    b:(10,)    650
    ReLU / Pool / Flat  无可学习参数              0
    ──────────────────────────────────────────────
    总计                                       1898

    公式：卷积 = out*in*kH*kW + out   全连接 = in*out + out
    观察：Conv2 参数最多（输入通道 8），FC 在大图上会暴增""")

# ====================================================================
# 第八部分：主程序
# ====================================================================

def main():
    print("="*60 + "\n   纯 NumPy CNN 手写数字识别实战\n" + "="*60)

    # 1. 数据
    print("\n【第1步】生成数据集...")
    imgs, labs = generate_digit_dataset(n_per_class=150, noise_std=0.15)
    n = len(labs); nt = int(n*0.8)
    Xtr,ytr = imgs[:nt],labs[:nt]; Xte,yte = imgs[nt:],labs[nt:]
    print(f"  形状:{imgs.shape} 训练:{nt} 测试:{n-nt}")
    show_sample_digits(imgs, labs)

    # 2. 模型
    print("\n【第2步】构建 CNN...")
    model = SimpleCNN(); model.params_summary(); detailed_param_analysis()

    # 3. 训练
    print("\n【第3步】训练（纯NumPy卷积较慢，这就是为什么要学PyTorch）")
    hist = train(model, Xtr, ytr, Xte, yte, epochs=15, bs=32, lr=0.002)

    # 4. 可视化
    plot_training_history(hist)

    # 5. 评估
    print("\n【第5步】评估...")
    tp = _predict_batched(model, Xte); acc = np.mean(tp==yte)
    print(f"  测试准确率: {acc:.1%}")
    for d in range(10):
        m = yte==d
        if m.sum()>0: print(f"    数字{d}: {np.mean(tp[m]==d):.1%} ({m.sum()}张)")
    cm = confusion_matrix(yte, tp); plot_confusion_matrix(cm)

    # 6. 可视化特征
    print("\n【第6步】可视化特征...")
    visualize_filters(model)
    idx = np.random.randint(len(yte))
    visualize_feature_maps(model, Xte[idx], yte[idx])
    show_predictions(model, Xte, yte)

    # 总结
    print(f"\n{'='*60}\n实战总结\n{'='*60}")
    print(f"  数据:{n}张8x8图 | 参数:1898 | 准确率:{acc:.1%}")
    print("  学到了：数据准备→模型搭建→训练循环→评估→可视化")
    print("  局限：纯NumPy慢（四重循环）、无GPU → 下一章学PyTorch！")

# ====================================================================
# 思考题
# ====================================================================
#
# 1. 【参数量】输入从 8x8 改成 28x28，FC 输入变为 16*7*7=784，
#    参数从 650 暴增到 7850。总参数量怎么变化？
#
# 2. 【过拟合】训练样本增到 1000 张/类，准确率会继续提升吗？
#    通道数改为 1→32→64 会过拟合吗？看训练/测试准确率差值。
#
# 3. 【架构】通道数 1→8→16 递增，反过来 1→16→8 效果如何？
#    深层需要更多通道捕获高级特征。
#
# 4. 【速度】哪个层最慢？Conv2（8输入×16输出，循环次数最多）。
#    实际框架用 im2col+矩阵乘法 加速。
#
# 5. 【实验】a) 去掉一个池化层 b) 加 Dropout c) SGD 替 Adam
#    d) Sigmoid 替 ReLU —— 分别观察准确率变化。
# ====================================================================

if __name__ == "__main__":
    main()
