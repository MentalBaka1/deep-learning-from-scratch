"""
第8章 · 第5节 · 文本生成与采样策略
实现贪心、温度采样、Top-K、Top-P、束搜索、重复惩罚，对比同一prompt下的输出差异。
前置知识: 第8章第3-4节 - GPT 模型与训练
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 准备：快速训练一个小模型
# ============================================================
print("=" * 60)
print("准备：快速训练小模型用于演示")
print("=" * 60)

@dataclass
class GPTConfig:
    vocab_size: int = 256; max_seq_len: int = 64; n_layers: int = 3
    n_heads: int = 4; d_model: int = 128; d_ff: int = 512

class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.hd = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3*c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(c.max_seq_len, c.max_seq_len))
                             .view(1, 1, c.max_seq_len, c.max_seq_len))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.nh,self.hd).transpose(1,2)
        k = k.view(B,T,self.nh,self.hd).transpose(1,2)
        v = v.view(B,T,self.nh,self.hd).transpose(1,2)
        a = F.softmax((q @ k.transpose(-2,-1)) / math.sqrt(self.hd)
                      * self.mask[:,:,:T,:T] + (1 - self.mask[:,:,:T,:T]) * -1e9, dim=-1)
        return self.proj((a @ v).transpose(1,2).contiguous().view(B,T,C))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.d_model), Attn(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ffn = nn.Sequential(nn.Linear(c.d_model, c.d_ff, bias=False),
                                 nn.GELU(), nn.Linear(c.d_ff, c.d_model, bias=False))
    def forward(self, x):
        x = x + self.attn(self.ln1(x)); return x + self.ffn(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_seq_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_f, self.head = nn.LayerNorm(c.d_model), nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.tok.weight = self.head.weight
    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    def next_logits(self, idx):
        logits, _ = self(idx[:, -self.config.max_seq_len:]); return logits[:, -1, :]

TEXT = ("深度学习是人工智能的核心技术。深度学习通过神经网络学习数据中的模式。"
        "卷积神经网络用于图像识别。循环神经网络用于序列建模。"
        "注意力机制让模型关注重要信息。Transformer基于自注意力机制。"
        "GPT是基于Transformer的语言模型。GPT通过预测下一个词来学习语言。"
        "大语言模型展现了强大的能力。深度学习改变了人工智能的发展方向。"
        "神经网络由多层神经元组成。反向传播算法用于训练神经网络。"
        "深度学习的成功依赖于大量数据和强大算力。")

chars = sorted(set(TEXT))
c2i = {ch: i for i, ch in enumerate(chars)}
i2c = {i: ch for i, ch in enumerate(chars)}
encode = lambda t: [c2i[c] for c in t if c in c2i]
decode = lambda ids: ''.join(i2c.get(i, '?') for i in ids)

SEQ = 48
data = torch.tensor(encode(TEXT), dtype=torch.long)
cfg = GPTConfig(vocab_size=len(chars), max_seq_len=SEQ+1)
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

print(f"  词汇表: {len(chars)}, 数据: {len(data)} tokens, 参数: {sum(p.numel() for p in model.parameters()):,}")
model.train()
for ep in range(200):
    ix = torch.randint(0, len(data)-SEQ-1, (16,))
    x = torch.stack([data[i:i+SEQ] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+SEQ+1] for i in ix]).to(device)
    _, loss = model(x, targets=y)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if (ep+1) % 50 == 0: print(f"    Epoch {ep+1} | Loss: {loss.item():.4f}")
model.eval()
prompt = "深度学习"
pid = encode(prompt)
print("  训练完成!\n")

# ============================================================
# 第1部分：贪心解码
# ============================================================
print("=" * 60)
print("第1部分：贪心解码 (Greedy)")
print("=" * 60)

@torch.no_grad()
def greedy_decode(model, ids, n=50):
    """每步选概率最高的token"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n):
        idx = torch.cat([idx, model.next_logits(idx).argmax(-1, keepdim=True)], 1)
    return idx[0].tolist()

print(f"  贪心: {decode(greedy_decode(model, pid, 60))[:80]}")

# ============================================================
# 第2部分：温度采样
# ============================================================
print("\n" + "=" * 60)
print("第2部分：温度采样 (Temperature)")
print("=" * 60)
print("  T→0: 接近贪心 | T=1: 原始分布 | T>1: 更随机")

@torch.no_grad()
def temperature_sample(model, ids, n=50, T=1.0):
    """logits/T → softmax → 采样"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n):
        nxt = torch.multinomial(F.softmax(model.next_logits(idx)/max(T,1e-8), -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx[0].tolist()

for t in [0.1, 0.5, 1.0, 2.0]:
    print(f"  T={t:<3}: {decode(temperature_sample(model, pid, 60, t))[:75]}")

# 温度对分布的影响可视化
dummy = torch.tensor([3., 2., 1.5, 1., .5, .1, -.5, -1., -2., -3.])
fig, axes = plt.subplots(1, 4, figsize=(14, 2.5))
for ax, t in zip(axes, [0.1, 0.5, 1.0, 2.0]):
    ax.bar(range(10), F.softmax(dummy/t, 0).numpy(), color='steelblue')
    ax.set_title(f'T={t}'); ax.set_ylim(0, 1)
plt.suptitle('温度对概率分布的影响'); plt.tight_layout()
plt.savefig('c:/project/dl/08_build_gpt/temperature_effect.png', dpi=100); plt.close()
print("\n  温度可视化已保存")

# ============================================================
# 第3部分：Top-K 采样
# ============================================================
print("\n" + "=" * 60)
print("第3部分：Top-K 采样 — 只从概率最高的K个token中采样")
print("=" * 60)

@torch.no_grad()
def top_k_sample(model, ids, n=50, k=10, T=1.0):
    """Top-K: 只从概率最高的K个token中采样"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n):
        logits = model.next_logits(idx) / max(T, 1e-8)
        topv, _ = torch.topk(logits, min(k, logits.size(-1)))
        logits[logits < topv[:, -1:]] = float('-inf')
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx[0].tolist()

print()
for k in [1, 5, 20, 50]:
    print(f"  K={k:<3}: {decode(top_k_sample(model, pid, 60, min(k, len(chars))))[:75]}")

# ============================================================
# 第4部分：Top-P (Nucleus) 采样
# ============================================================
print("\n" + "=" * 60)
print("第4部分：Top-P (Nucleus) — 累积概率>=P的最小token集合")
print("=" * 60)

@torch.no_grad()
def top_p_sample(model, ids, n=50, p=0.9, T=1.0):
    """Top-P: 从累积概率>=p的最小集合中采样"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(n):
        logits = model.next_logits(idx) / max(T, 1e-8)
        sorted_l, sorted_i = torch.sort(logits, descending=True)
        sorted_p = F.softmax(sorted_l, -1)
        cum_p = torch.cumsum(sorted_p, -1)
        mask = (cum_p - sorted_p) > p
        sorted_l[mask] = float('-inf')
        out = torch.zeros_like(logits)
        out.scatter_(1, sorted_i, sorted_l)
        nxt = torch.multinomial(F.softmax(out, -1), 1)
        idx = torch.cat([idx, nxt], 1)
    return idx[0].tolist()

print()
for p in [0.5, 0.8, 0.9, 0.95]:
    print(f"  P={p:<4}: {decode(top_p_sample(model, pid, 60, p))[:75]}")

# ============================================================
# 第5部分：束搜索
# ============================================================
print("\n" + "=" * 60)
print("第5部分：束搜索 — 维护beam个候选，保留总分最高的")
print("=" * 60)

@torch.no_grad()
def beam_search(model, ids, n=40, beam=3):
    """束搜索: 维护beam个最优候选序列"""
    beams = [(ids[:], 0.0)]
    for _ in range(n):
        cands = []
        for seq, score in beams:
            idx = torch.tensor([seq], dtype=torch.long, device=device)
            lp = F.log_softmax(model.next_logits(idx), -1)[0]
            topv, topi = torch.topk(lp, beam)
            for i in range(beam):
                cands.append((seq + [topi[i].item()], score + topv[i].item()))
        cands.sort(key=lambda x: x[1], reverse=True)
        beams = cands[:beam]
    return beams[0][0], beams[0][1]

res, sc = beam_search(model, pid, 40, 3)
print(f"\n  Beam=3: {decode(res)[:75]} (得分:{sc:.2f})")
print(f"  贪心:  {decode(greedy_decode(model, pid, 40))[:75]}")

# ============================================================
# 第6部分：重复惩罚
# ============================================================
print("\n" + "=" * 60)
print("第6部分：重复惩罚 — logits/penalty 抑制已出现token")
print("=" * 60)

@torch.no_grad()
def rep_penalty_sample(model, ids, n=50, T=1.0, penalty=1.2):
    """对已出现token的logits施加惩罚"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    seen = set(ids)
    for _ in range(n):
        logits = model.next_logits(idx) / max(T, 1e-8)
        for tid in seen:
            if tid < logits.size(-1):
                logits[0,tid] = logits[0,tid]/penalty if logits[0,tid]>0 else logits[0,tid]*penalty
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1); seen.add(nxt.item())
    return idx[0].tolist()

for pen in [1.0, 1.2, 1.5, 2.0]:
    print(f"  pen={pen:<3}: {decode(rep_penalty_sample(model, pid, 60, 0.8, pen))[:75]}")

# ============================================================
# 第7部分：综合 generate 函数
# ============================================================
print("\n" + "=" * 60)
print("第7部分：综合 generate 函数")
print("=" * 60)

@torch.no_grad()
def generate(model, ids, n=50, T=1.0, top_k=0, top_p=1.0, rep_pen=1.0):
    """整合所有策略: T=0→贪心 | top_k>0→Top-K | top_p<1→Top-P | rep_pen>1→惩罚"""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    seen = set(ids)
    for _ in range(n):
        logits = model.next_logits(idx)
        # 重复惩罚
        if rep_pen != 1.0:
            for t in seen:
                if t < logits.size(-1):
                    logits[0,t] = logits[0,t]/rep_pen if logits[0,t]>0 else logits[0,t]*rep_pen
        if T == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / T
            if top_k > 0:
                tv, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < tv[:, -1:]] = float('-inf')
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                sp = F.softmax(sl, -1)
                cp = torch.cumsum(sp, -1)
                sl[(cp - sp) > top_p] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(1, si, sl)
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
        idx = torch.cat([idx, nxt], 1)
        seen.add(nxt.item())
    return idx[0].tolist()

# 综合函数演示
print("\n  策略组合演示:")
for label, kw in [("贪心", dict(T=0)), ("T=0.8", dict(T=0.8)),
                  ("Top-K=10", dict(T=0.8, top_k=10)), ("Top-P=0.9", dict(T=0.8, top_p=0.9)),
                  ("P0.9+惩罚", dict(T=0.8, top_p=0.9, rep_pen=1.3))]:
    print(f"    {label:<12}: {decode(generate(model, pid, 60, **kw))[:75]}")

# ============================================================
# 第8部分：同一prompt全面对比 + 重复率分析
# ============================================================
print("\n" + "=" * 60)
print("第8部分：同一提示不同策略对比")
print("=" * 60)
tp, ti = "神经网络", encode("神经网络")
print(f"  提示: '{tp}'\n")
strats = [("贪心", lambda: greedy_decode(model, ti, 60)),
          ("T=0.5", lambda: temperature_sample(model, ti, 60, 0.5)),
          ("T=2.0", lambda: temperature_sample(model, ti, 60, 2.0)),
          ("Top-K=5", lambda: top_k_sample(model, ti, 60, 5)),
          ("Top-P=0.9", lambda: top_p_sample(model, ti, 60, 0.9)),
          ("Beam=3", lambda: beam_search(model, ti, 40, 3)[0]),
          ("惩罚=1.5", lambda: rep_penalty_sample(model, ti, 60, penalty=1.5))]
results = {}
for name, fn in strats:
    text = decode(fn()); results[name] = text
    ch = list(text)
    bg = [(ch[i], ch[i+1]) for i in range(len(ch)-1)] if len(ch)>1 else []
    rr = f"{1-len(set(bg))/max(len(bg),1):.0%}" if bg else "N/A"
    print(f"  {name:<12}: {text[:65]}  [重复:{rr}]")

print("\n  策略建议: 代码T=0.2+P0.9 | 对话T=0.7+P0.9 | 创意T=0.9+P0.95 | 翻译beam=4")
print("  原则: 精确→低温 | 创意→高温+Top-P | 减重复→惩罚 | Top-P通常优于Top-K")

# ============================================================
# 思考题
# ============================================================
print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. T=0.5 和 T=2.0 的输出为何差异巨大？从概率分布角度解释。
2. Top-K 的 K 固定是缺点——举例说明何时 K=10 太多、何时太少。
3. 束搜索 beam 越大质量越好吗？为何开放式生成中效果不佳？
4. presence_penalty 和 frequency_penalty 有何区别？各适合何场景？
5. 同时设 top_k=10 和 top_p=0.9，两个过滤器如何协同？
""")
print("本节完！")
