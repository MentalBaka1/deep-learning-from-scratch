"""
====================================================================
第7章 · 第5节 · Decoder 模块组装
====================================================================

【一句话总结】
Decoder 比 Encoder 多了"因果掩码自注意力"和"交叉注意力"——
因果掩码确保生成时不能偷看未来，交叉注意力从 Encoder 获取信息。

【为什么深度学习需要这个？】
- GPT 系列就是纯 Decoder 架构（去掉交叉注意力）
- 理解 Decoder 的掩码机制是理解自回归生成的关键
- Encoder-Decoder vs Decoder-Only 是大模型架构的核心选择

【核心概念】
1. Decoder Block 的结构（3个子层）：掩码自注意力 + 交叉注意力 + FFN，每个子层都有 LN + 残差
2. Decoder-Only 架构（GPT风格）：去掉交叉注意力，只保留掩码自注意力+FFN
3. KV Cache：缓存已计算的K,V避免重复计算，加速自回归生成

【前置知识】第7章第1-4节，第6章第4节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

torch.manual_seed(42)

# ====================================================================
# 第1部分：基础组件（支持交叉注意力和 KV Cache）
# ====================================================================
print("=" * 60)
print("第1部分：基础组件")
print("=" * 60)


class MultiHeadAttention(nn.Module):
    """多头注意力（通用版）——支持自注意力、交叉注意力、KV Cache。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, kv_cache=None):
        """
        query/key/value: (batch, seq, d_model)
        kv_cache: {"k":..., "v":...} 或 None（增量推理用）
        返回 output 或 (output, new_cache)
        """
        B = query.size(0)
        Q = self.W_Q(query).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        # KV Cache：拼接历史 K/V
        new_cache = None
        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)
            new_cache = {"k": K, "v": V}
        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        w = self.dropout(F.softmax(scores, dim=-1))
        out = self.W_O(torch.matmul(w, V).transpose(1, 2).contiguous().view(B, -1, self.d_model))
        return (out, new_cache) if new_cache is not None else out


class FeedForward(nn.Module):
    """位置级前馈网络: Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


print("  MultiHeadAttention: 支持自注意力/交叉注意力/KV Cache")
print("  FeedForward: GELU 前馈网络")

# ====================================================================
# 第2部分：因果掩码 —— 不能偷看未来
# ====================================================================
print("\n\n" + "=" * 60)
print("第2部分：因果掩码")
print("=" * 60)


def create_causal_mask(seq_len):
    """上三角=True(遮挡), 对角线及以下=False(可见)。"""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
                      ).unsqueeze(0).unsqueeze(0)


causal = create_causal_mask(5)
print(f"\n  因果掩码 (5x5): 形状 {list(causal.shape)}")
for i in range(5):
    print(f"    pos_{i}: {'  '.join('x' if causal[0,0,i,j] else '.' for j in range(5))}")

# 验证 softmax 后被遮挡位置权重为 0
w = F.softmax(torch.randn(1,1,5,5).masked_fill(causal, float("-inf")), dim=-1)
print(f"\n  掩码后注意力权重示例（某头）:")
for i in range(5):
    print(f"    pos_{i}: [{', '.join(f'{w[0,0,i,j]:.3f}' for j in range(5))}]")
print("  → 未来位置权重全为 0！")

# ====================================================================
# 第3部分：DecoderBlock —— 3个子层
# ====================================================================
#
# Decoder Block 比 Encoder Block 多了一个"交叉注意力"子层：
#
#   输入 x (来自上一层 Decoder)
#    ├─────── 残差 ──┐
#    ↓               │
#   LN → Masked Self-Attn → +    ← 子层1: 只看已生成的token
#    ├─────── 残差 ──┐
#    ↓               │
#   LN → Cross-Attn ────→ +      ← 子层2: 从Encoder获取信息
#    ├─────── 残差 ──┐
#    ↓               │
#   LN → FFN ──────────→ +       ← 子层3: 非线性变换
#    ↓
#   输出

print("\n\n" + "=" * 60)
print("第3部分：DecoderBlock（掩码自注意力 + 交叉注意力 + FFN）")
print("=" * 60)


class DecoderBlock(nn.Module):
    """Decoder Block (Pre-Norm): 掩码自注意力→交叉注意力→FFN，各有LN+残差。"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.m_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.c_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, causal_mask=None, cross_mask=None):
        r=x; n=self.norm1(x); x = r + self.m_attn(n, n, n, causal_mask)       # 掩码自注意力
        r=x; n=self.norm2(x); x = r + self.c_attn(n, enc_out, enc_out, cross_mask)  # 交叉注意力
        r=x; x = r + self.ffn(self.norm3(x))                                   # FFN
        return x


d_model, n_heads, d_ff = 64, 8, 256
db = DecoderBlock(d_model, n_heads, d_ff, dropout=0.1); db.eval()
enc_out = torch.randn(2, 8, d_model)
dec_in  = torch.randn(2, 6, d_model)
with torch.no_grad():
    dec_out = db(dec_in, enc_out, causal_mask=create_causal_mask(6))
print(f"\n  Encoder输出: {list(enc_out.shape)}, Decoder输入: {list(dec_in.shape)}")
print(f"  Decoder输出: {list(dec_out.shape)}, 参数量: {sum(p.numel() for p in db.parameters()):,}")

# ====================================================================
# 第4部分：DecoderOnlyBlock —— GPT 风格
# ====================================================================
print("\n\n" + "=" * 60)
print("第4部分：DecoderOnlyBlock（GPT 风格，无交叉注意力）")
print("=" * 60)


class DecoderOnlyBlock(nn.Module):
    """GPT 风格 Block: 掩码自注意力 + FFN（去掉交叉注意力）。"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, causal_mask=None, kv_cache=None):
        r = x; n = self.norm1(x)
        if kv_cache is not None:
            a, cache = self.attn(n, n, n, causal_mask, kv_cache)
        else:
            a = self.attn(n, n, n, causal_mask); cache = None
        x = r + a
        x = x + self.ffn(self.norm2(x))
        return (x, cache) if cache is not None else x


gb = DecoderOnlyBlock(d_model, n_heads, d_ff); gb.eval()
with torch.no_grad():
    go = gb(torch.randn(2, 10, d_model), create_causal_mask(10))
print(f"\n  输出: {list(go.shape)}, 参数量: {sum(p.numel() for p in gb.parameters()):,}")
print(f"  (比 DecoderBlock 少了交叉注意力子层)")

# ====================================================================
# 第5部分：TransformerDecoder + 自回归生成
# ====================================================================
print("\n\n" + "=" * 60)
print("第5部分：TransformerDecoder（堆叠 N 层）")
print("=" * 60)


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


class TransformerDecoder(nn.Module):
    """Decoder-Only Transformer (GPT风格): Embed→PosEnc→Block×N→LN→Linear"""
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers, max_len=512, drop=0.1):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.emb = nn.Embedding(vocab, d_model)
        self.pe = PositionalEncoding(d_model, max_len, drop)
        self.scale = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [DecoderOnlyBlock(d_model, n_heads, d_ff, drop) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, ids, use_cache=False, past=None):
        x = self.emb(ids) * self.scale
        if past and past[0] and past[0].get("k") is not None:
            pl = past[0]["k"].size(2)
            x = self.pe.dropout(x + self.pe.pe[:, pl:pl+ids.size(1)])
            cm = None
        else:
            x = self.pe(x); cm = create_causal_mask(ids.size(1)).to(x.device)
        caches = []
        for i, layer in enumerate(self.layers):
            c = past[i] if past else None
            if use_cache:
                x, nc = layer(x, cm, kv_cache=c or {})
                caches.append(nc)
            else:
                x = layer(x, cm)
        logits = self.proj(self.ln(x))
        return (logits, caches) if use_cache else logits


vocab, n_layers = 100, 4
dec = TransformerDecoder(vocab, d_model, n_heads, d_ff, n_layers, 128, 0.0); dec.eval()
with torch.no_grad():
    lo = dec(torch.randint(1, vocab, (2, 10)))
print(f"\n  输入: (2, 10), 输出: {list(lo.shape)}, 参数量: {sum(p.numel() for p in dec.parameters()):,}")

# ====================================================================
# 第6部分：KV Cache 加速对比
# ====================================================================
print("\n\n" + "=" * 60)
print("第6部分：KV Cache 加速对比")
print("=" * 60)


def gen_no_cache(m, pf, n):
    ids = pf.clone()
    for _ in range(n):
        ids = torch.cat([ids, m(ids)[:, -1:].argmax(-1)], 1)
    return ids


def gen_cache(m, pf, n):
    ids = pf.clone()
    lo, ca = m(ids, True, [None]*m.n_layers)
    nt = lo[:, -1:].argmax(-1); ids = torch.cat([ids, nt], 1)
    for _ in range(n-1):
        lo, ca = m(nt, True, ca)
        nt = lo[:, -1:].argmax(-1); ids = torch.cat([ids, nt], 1)
    return ids


pf = torch.randint(1, vocab, (1, 5))
print(f"\n  {'长度':<6s} {'无Cache':>9s} {'有Cache':>9s} {'加速':>7s} {'一致':>5s}")
print(f"  {'-'*40}")
for gl in [20, 50, 100]:
    with torch.no_grad():
        t0=time.time(); r1=gen_no_cache(dec,pf,gl); t1=time.time()
        t2=time.time(); r2=gen_cache(dec,pf,gl);    t3=time.time()
    d1, d2 = t1-t0, t3-t2
    print(f"  {gl:<6d} {d1:>8.4f}s {d2:>8.4f}s {d1/max(d2,1e-9):>6.1f}x"
          f"  {'Yes' if torch.equal(r1,r2) else 'No':>4s}")
print("  → 生成越长，Cache 加速越明显。代价：额外显存。")

# ====================================================================
# 第7部分：架构对比表
# ====================================================================
print("\n\n" + "=" * 60)
print("第7部分：Encoder-Decoder vs Decoder-Only")
print("=" * 60)
print("""
  ┌──────────────┬────────────────────┬────────────────────┐
  │     特性     │  Encoder-Decoder   │  Decoder-Only(GPT) │
  ├──────────────┼────────────────────┼────────────────────┤
  │ 代表模型     │ T5, BART           │ GPT, LLaMA, Qwen   │
  │ Decoder子层  │ 3(自+交叉+FFN)     │ 2(自+FFN)           │
  │ 典型任务     │ 翻译、摘要         │ 生成、对话、通用    │
  │ 参数效率     │ 两套参数           │ 一套参数，更高效    │
  │ 扩展性       │ 需扩展两部分       │ 只扩展一个栈        │
  └──────────────┴────────────────────┴────────────────────┘
  Decoder-Only 成为主流：架构简单、参数高效、prompt统一各种任务。
""")

# ====================================================================
# 第8部分：思考题
# ====================================================================
print("=" * 60)
print("思考题")
print("=" * 60)

for i, (q, a) in enumerate([
    ("因果掩码为什么用上三角而非下三角？",
     "scores[i][j] 行=Query列=Key。上三角遮挡=位置i不能看j(j>i)。\n"
     "    反过来则只能看未来不能看过去，对自回归毫无意义。"),
    ("KV Cache 显存: d_model=4096, 32层, seq=2048, float16 要多少？",
     "每层K+V=2*2048*4096元素, 32层≈5.37亿元素, float16≈1GB。"),
    ("交叉注意力中 Q来自Decoder, K/V来自Encoder, 能反过来吗？",
     "Q='提问方', K/V='信息源'。Decoder根据生成内容向Encoder查询信息，\n"
     "    反过来让Encoder向未完成的Decoder查询不合理。"),
    ("GPT 没有交叉注意力，如何做翻译？",
     "把源文本放进prompt续写译文。因果掩码保证能看到前面的完整源文本。"),
], 1):
    print(f"\n思考题 {i}：{q}\n  参考答案：{a}")

# ====================================================================
# 总结
# ====================================================================
print("\n\n" + "=" * 60)
print("本节总结")
print("=" * 60)
print("""
  1. DecoderBlock (Enc-Dec) = 掩码自注意力 + 交叉注意力 + FFN
  2. DecoderOnlyBlock (GPT) = 掩码自注意力 + FFN
  3. 因果掩码: 上三角→softmax后未来位置权重为0→保证因果性
  4. KV Cache: 缓存历史K/V, 每步只算新token, 显著加速推理

  下一节: 第7章·第6节·完整 Transformer 组装与实战
""")
print("=" * 60)
print("第7章 · 第5节 完成！")
print("=" * 60)
