"""
====================================================================
第7章 · 第6节 · 完整 Transformer：组装与实战
====================================================================

【一句话总结】
将所有组件组装成一个完整的 Transformer，然后在一个简单任务上实际训练它。

【核心概念】
1. 完整架构: input→Embedding→+PE→Encoder×N; target→Embedding→+PE→Decoder×N→Linear→Softmax
2. 训练(Teacher Forcing并行) vs 推理(自回归串行)
3. Label Smoothing

【前置知识】第7章第1-5节全部
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
torch.manual_seed(42)
np.random.seed(42)

# ====================================================================
# 第1部分：所有组件（自包含）
# ====================================================================
print("=" * 60)
print("第1部分：定义所有组件")
print("=" * 60)


def sdp_attention(Q, K, V, mask=None):
    """缩放点积注意力: softmax(QK^T/sqrt(d_k)) * V"""
    d_k = Q.size(-1)
    s = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        s = s.masked_fill(mask, float("-inf"))
    w = F.softmax(s, dim=-1)
    return torch.matmul(w, V), w


class MultiHeadAttention(nn.Module):
    """多头注意力——支持自注意力和交叉注意力。"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # 供可视化

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        Q = self.W_Q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        out, w = sdp_attention(Q, K, V, mask)
        self.attn_weights = w.detach()
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.dropout(self.W_O(out))


class FeedForward(nn.Module):
    """Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码。"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderBlock(nn.Module):
    """Encoder Block (Pre-Norm): Self-Attn + FFN"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        r=x; n=self.n1(x); x = r + self.attn(n, n, n, mask)
        return x + self.ffn(self.n2(x))


class DecoderBlock(nn.Module):
    """Decoder Block (Pre-Norm): Masked-Self-Attn + Cross-Attn + FFN"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.m_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.c_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, enc, cm=None, xm=None):
        r=x; n=self.n1(x); x = r + self.m_attn(n, n, n, cm)
        r=x; n=self.n2(x); x = r + self.c_attn(n, enc, enc, xm)
        return x + self.ffn(self.n3(x))


print("  组件: MultiHeadAttention, FeedForward, PositionalEncoding,")
print("        EncoderBlock, DecoderBlock")

# ====================================================================
# 第2部分：完整 Transformer 类
# ====================================================================
print("\n\n" + "=" * 60)
print("第2部分：完整 Transformer 类")
print("=" * 60)


def causal_mask(seq_len):
    """因果掩码: 上三角=True(遮挡)。"""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
                      ).unsqueeze(0).unsqueeze(0)


class Transformer(nn.Module):
    """
    完整 Encoder-Decoder Transformer。
    src → Embed+PE → Encoder×N → enc_out
    tgt → Embed+PE → Decoder×N (with enc_out) → LN → Linear → logits
    """
    def __init__(self, src_vocab, tgt_vocab, d_model=64, n_heads=4,
                 d_ff=256, n_enc=3, n_dec=3, max_len=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model, self.pad_idx = d_model, pad_idx
        self.scale = math.sqrt(d_model)
        # Encoder
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=pad_idx)
        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        self.enc = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_enc)])
        self.enc_ln = nn.LayerNorm(d_model)
        # Decoder
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=pad_idx)
        self.tgt_pe = PositionalEncoding(d_model, max_len, dropout)
        self.dec = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_dec)])
        self.dec_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, tgt_vocab)
        # Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def encode(self, src, sm=None, verbose=False):
        x = self.src_pe(self.src_emb(src) * self.scale)
        if verbose: print(f"    [Enc] Emb+PE: {list(x.shape)}")
        for i, layer in enumerate(self.enc):
            x = layer(x, sm)
            if verbose: print(f"    [Enc] L{i+1}:    {list(x.shape)}")
        return self.enc_ln(x)

    def decode(self, tgt, enc_out, tm=None, xm=None, verbose=False):
        x = self.tgt_pe(self.tgt_emb(tgt) * self.scale)
        if verbose: print(f"    [Dec] Emb+PE: {list(x.shape)}")
        for i, layer in enumerate(self.dec):
            x = layer(x, enc_out, cm=tm, xm=xm)
            if verbose: print(f"    [Dec] L{i+1}:    {list(x.shape)}")
        logits = self.out_proj(self.dec_ln(x))
        if verbose: print(f"    [Out] Proj:   {list(logits.shape)}")
        return logits

    def forward(self, src, tgt, verbose=False):
        if verbose: print(f"\n  --- Transformer 前向传播 ---")
        sm = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        tm = causal_mask(tgt.size(1)).to(tgt.device)
        enc_out = self.encode(src, sm, verbose)
        return self.decode(tgt, enc_out, tm, sm, verbose)

# ====================================================================
# 第3部分：维度追踪
# ====================================================================
print("\n\n" + "=" * 60)
print("第3部分：维度追踪")
print("=" * 60)

VS = 13  # pad=0, bos=1, eos=2, 数字1~10→token3~12
mdemo = Transformer(VS, VS, 64, 4, 256, 2, 2, 32, 0.0); mdemo.eval()
with torch.no_grad():
    _ = mdemo(torch.tensor([[5,3,6,4,2]]), torch.tensor([[1,3,4,5,6]]), verbose=True)

# ====================================================================
# 第4部分：参数量统计
# ====================================================================
print("\n\n" + "=" * 60)
print("第4部分：参数量统计")
print("=" * 60)

grp = {"源嵌入": mdemo.src_emb, "目标嵌入": mdemo.tgt_emb,
       "Encoder层": mdemo.enc, "Enc LN": mdemo.enc_ln,
       "Decoder层": mdemo.dec, "Dec LN": mdemo.dec_ln, "输出投影": mdemo.out_proj}
total = sum(p.numel() for p in mdemo.parameters())
print(f"\n  {'组件':<14s} {'参数量':>10s} {'占比':>7s}")
print(f"  {'-'*34}")
for nm, mod in grp.items():
    c = sum(p.numel() for p in mod.parameters())
    print(f"  {nm:<14s} {c:>10,} {c/total*100:>6.1f}%")
print(f"  {'总计':<14s} {total:>10,}")
ep = sum(p.numel() for p in mdemo.enc[0].parameters())
dp = sum(p.numel() for p in mdemo.dec[0].parameters())
print(f"\n  单个 EncBlock: {ep:,}, DecBlock: {dp:,} ({dp/ep:.2f}x, 多了交叉注意力)")

# ====================================================================
# 第5部分：训练数据 —— 序列排序
# ====================================================================
print("\n\n" + "=" * 60)
print("第5部分：训练数据（序列排序）")
print("=" * 60)

PAD, BOS, EOS, NOFF = 0, 1, 2, 3
n2t = lambda n: n + NOFF - 1
t2n = lambda t: t - NOFF + 1 if t >= NOFF else None


def make_data(n, slen=4, mx=8):
    """src=[乱序+eos], tgt_in=[bos+有序], tgt_lbl=[有序+eos]"""
    S, I, L = [], [], []
    for _ in range(n):
        nums = np.random.choice(range(1, mx+1), slen, replace=False)
        s = sorted(nums)
        S.append([n2t(x) for x in nums]+[EOS])
        I.append([BOS]+[n2t(x) for x in s])
        L.append([n2t(x) for x in s]+[EOS])
    return torch.tensor(S), torch.tensor(I), torch.tensor(L)


SL, MX, NT, NE = 4, 8, 3000, 200
tr_s, tr_i, tr_l = make_data(NT, SL, MX)
te_s, te_i, te_l = make_data(NE, SL, MX)
print(f"\n  任务: 排序{SL}个数字(1~{MX}), 训练{NT}, 测试{NE}")
for j in range(3):
    sn = [t2n(t.item()) for t in tr_s[j] if t2n(t.item())]
    tn = [t2n(t.item()) for t in tr_l[j] if t2n(t.item())]
    print(f"    {sn} → {tn}")

# ====================================================================
# 第6部分：训练 —— AdamW + Warmup + Cosine LR
# ====================================================================
print("\n\n" + "=" * 60)
print("第6部分：训练")
print("=" * 60)


def get_lr(step, dm, warmup, total):
    if step < warmup:
        return (dm**-0.5) * step * (warmup**-1.5)
    return (dm**-0.5) * 0.5 * (1 + math.cos(math.pi * (step-warmup) / max(1, total-warmup)))


torch.manual_seed(42)
model = Transformer(VS, VS, 64, 4, 256, 2, 2, 32, 0.1, PAD)
crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
EP, BS = 200, 128
WU = 20; TS = EP * (NT//BS+1)

print(f"  参数: {sum(p.numel() for p in model.parameters()):,}, "
      f"Epochs: {EP}, BS: {BS}, LabelSmooth: 0.1")

losses, accs, lrs = [], [], []
st = 0; t0 = time.time()

for ep in range(EP):
    model.train(); perm = torch.randperm(NT); el, nb = 0, 0
    for s in range(0, NT, BS):
        ix = perm[s:s+BS]; st += 1
        lr = get_lr(st, 64, WU, TS)
        for pg in opt.param_groups: pg["lr"] = lr
        lrs.append(lr)
        lo = crit(model(tr_s[ix], tr_i[ix]).reshape(-1, VS), tr_l[ix].reshape(-1))
        opt.zero_grad(); lo.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        el += lo.item(); nb += 1
    losses.append(el / nb)
    if (ep+1) % 20 == 0 or ep == 0:
        model.eval()
        with torch.no_grad():
            pr = model(te_s, te_i).argmax(-1)
            ta = (pr == te_l).float().mean().item()
            sa = (pr == te_l).all(1).float().mean().item()
        accs.append((ep+1, ta, sa))
        print(f"    Ep {ep+1:>3d} | Loss {losses[-1]:.4f} | "
              f"Tok {ta:.3f} Seq {sa:.3f} | LR {lr:.6f} | {time.time()-t0:.1f}s")

print(f"\n  完成! {time.time()-t0:.1f}s")

# 训练曲线
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(losses, color="steelblue"); ax[0].set_title("训练损失"); ax[0].set_xlabel("Epoch")
ax[0].grid(True, alpha=0.3)
es = [a[0] for a in accs]
ax[1].plot(es, [a[1] for a in accs], "o-", label="Token")
ax[1].plot(es, [a[2] for a in accs], "s-", label="Seq")
ax[1].legend(); ax[1].set_title("测试准确率"); ax[1].grid(True, alpha=0.3)
ax[2].plot(lrs, color="crimson"); ax[2].set_title("LR (Warmup+Cosine)"); ax[2].grid(True, alpha=0.3)
plt.suptitle("Transformer 排序任务", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("07_06_training_curves.png", dpi=100, bbox_inches="tight"); plt.show()
print("[已保存] 07_06_training_curves.png")

# ====================================================================
# 第7部分：贪心自回归解码
# ====================================================================
print("\n\n" + "=" * 60)
print("第7部分：贪心自回归解码")
print("=" * 60)

print("""
  训练(Teacher Forcing): Decoder输入=真实标签右移，可并行计算
  推理(自回归):          从<BOS>逐步生成，每步用上一步预测，串行
""")


def greedy_decode(mdl, src, maxl=20):
    """Encoder编码后，Decoder从BOS开始逐步生成直到EOS。"""
    mdl.eval()
    with torch.no_grad():
        sm = (src == mdl.pad_idx).unsqueeze(1).unsqueeze(2)
        enc = mdl.encode(src, sm)
        tgt = torch.tensor([[BOS]])
        for _ in range(maxl):
            cm = causal_mask(tgt.size(1)).to(tgt.device)
            lo = mdl.decode(tgt, enc, cm, sm)
            nxt = lo[:, -1:].argmax(-1)
            tgt = torch.cat([tgt, nxt], 1)
            if nxt.item() == EOS: break
    return tgt[0].tolist()


print(f"  {'输入(乱序)':<18s} → {'目标':>16s} | {'预测':>16s} | ?")
print(f"  {'-'*60}")
nok = 0
for i in range(min(20, NE)):
    sn = [t2n(t.item()) for t in te_s[i] if t2n(t.item())]
    tn = [t2n(t.item()) for t in te_l[i] if t2n(t.item())]
    pn = [t2n(t) for t in greedy_decode(model, te_s[i:i+1]) if t2n(t)]
    ok = pn == tn; nok += ok
    print(f"  {str(sn):<18s} → {str(tn):>16s} | {str(pn):>16s} | {'OK' if ok else 'X'}")
print(f"\n  准确率: {nok}/{min(20,NE)} = {nok/min(20,NE):.0%}")

# ====================================================================
# 第8部分：注意力可视化
# ====================================================================
print("\n\n" + "=" * 60)
print("第8部分：注意力可视化")
print("=" * 60)

vis_s, vis_i = te_s[0:1], te_i[0:1]
sn_v = [t2n(t.item()) for t in vis_s[0] if t2n(t.item())]
tn_v = [t2n(t.item()) for t in te_l[0] if t2n(t.item())]
print(f"\n  样本: {sn_v} → {tn_v}")

model.eval()
with torch.no_grad():
    _ = model(vis_s, vis_i)

sl = [str(t2n(t.item())) if t2n(t.item()) else "eos" for t in vis_s[0]]
tl = ["bos"] + [str(n) for n in tn_v]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 上: Encoder 自注意力
ew = model.enc[-1].attn.attn_weights[0]
for h in range(min(3, ew.size(0))):
    ax = axes[0][h]; d = ew[h].numpy()
    ax.imshow(d, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for ii in range(d.shape[0]):
        for jj in range(d.shape[1]):
            ax.text(jj, ii, f"{d[ii,jj]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if d[ii,jj]>0.5 else "black")
    ax.set_xticks(range(len(sl))); ax.set_xticklabels(sl)
    ax.set_yticks(range(len(sl))); ax.set_yticklabels(sl)
    ax.set_title(f"Enc Self-Attn H{h+1}", fontweight="bold")

# 下: Decoder 交叉注意力
cw = model.dec[-1].c_attn.attn_weights[0]
for h in range(min(3, cw.size(0))):
    ax = axes[1][h]; d = cw[h].numpy()
    ax.imshow(d, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for ii in range(d.shape[0]):
        for jj in range(d.shape[1]):
            ax.text(jj, ii, f"{d[ii,jj]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if d[ii,jj]>0.5 else "black")
    ax.set_xticks(range(len(sl))); ax.set_xticklabels(sl)
    ax.set_yticks(range(len(tl))); ax.set_yticklabels(tl)
    ax.set_title(f"Cross-Attn H{h+1}", fontweight="bold")

plt.suptitle(f"注意力: {sn_v}→{tn_v} (上:Enc自注意力, 下:交叉注意力)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("07_06_attention_heatmaps.png", dpi=100, bbox_inches="tight"); plt.show()
print("[已保存] 07_06_attention_heatmaps.png")
print("  交叉注意力: 生成第k小数时应集中关注源序列中该数字的位置。")

# ====================================================================
# 第9部分：思考题
# ====================================================================
print("\n\n" + "=" * 60)
print("思考题")
print("=" * 60)

for i, (q, a) in enumerate([
    ("Label Smoothing 是什么？为什么 Transformer 常用？",
     "将一部分概率从正确类别匀给其他类别(如0.1)，防止过度自信，\n"
     "    起正则化效果。原始论文用了 0.1 的 smoothing。"),
    ("为什么 Transformer 需要 warmup？",
     "训练初期Adam二阶矩不准，大LR导致不稳定。Warmup让优化器先积累统计信息。"),
    ("Encoder/Decoder共享词表时能否也共享Embedding？",
     "可以(T5就这样做)。减参数、对齐表示。排序任务源/目标相同，很适合共享。"),
    ("如何改成 Decoder-Only？",
     "拼接: [3,1,4,2,<sep>,1,2,3,4,<eos>]，因果掩码，loss只算<sep>之后。"),
    ("Beam Search vs Greedy？排序任务上有用吗？",
     "Beam保留k条路径，适合多种合理输出的任务。排序答案唯一，Greedy足够。"),
], 1):
    print(f"\n思考题 {i}：{q}\n  参考答案：{a}")

# ====================================================================
# 总结
# ====================================================================
print("\n\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
  1. 完整架构: Encoder(Emb+PE→Block×N→LN) + Decoder(Emb+PE→Block×N→LN→Linear)
  2. 训练: Teacher Forcing(并行) vs 推理: 自回归(串行)
  3. 技巧: Label Smoothing + Warmup+Cosine LR + 梯度裁剪 + AdamW
  4. 注意力可视化: Enc自注意力理解输入，交叉注意力对齐源→目标

  恭喜！从位置编码到完整 Transformer，每个组件你都亲手实现过了！
""")
print("=" * 60)
print("第7章 · 第6节 完成！")
print("=" * 60)
