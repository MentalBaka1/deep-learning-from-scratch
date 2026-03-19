"""
====================================================================
第8章 · 第3节 · GPT模型完整实现（本教程的高潮！）
====================================================================
用约200行 PyTorch 代码从零实现完整 GPT 模型，
包含多头因果自注意力、前馈网络、残差连接和层归一化。

【核心架构】
CausalSelfAttention → FeedForward → TransformerBlock → GPT
全部使用 Pre-Norm 结构（GPT-2 风格）

【前置知识】第6章-注意力机制, 第7章-Transformer, 第8章第2节-语言模型
====================================================================
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 第1部分：GPT 配置
# ============================================================
print("=" * 60)
print("第1部分：GPT 配置（GPTConfig）")
print("=" * 60)

@dataclass
class GPTConfig:
    """GPT 超参数集中管理"""
    vocab_size: int = 256
    max_seq_len: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = None       # 默认 4 * d_model
    dropout: float = 0.1
    bias: bool = False      # GPT-2 不使用偏置

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0

def get_config(size='mini', vocab_size=256, max_seq_len=128):
    """
    预定义配置:
    mini:   4层 128维 4头  ~1M参数
    small:  6层 256维 8头  ~10M参数
    medium: 12层 512维 12头 ~85M参数
    """
    cfgs = {
        'mini':   GPTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                            n_layers=4, n_heads=4, d_model=128),
        'small':  GPTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                            n_layers=6, n_heads=8, d_model=256),
        'medium': GPTConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                            n_layers=12, n_heads=12, d_model=512),
    }
    return cfgs[size]

config = get_config('mini')
print(f"  Mini 配置: {config}")

# ============================================================
# 第2部分：因果自注意力
# ============================================================
print("\n" + "=" * 60)
print("第2部分：因果自注意力 (CausalSelfAttention)")
print("=" * 60)

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力
    - Q/K/V 合并为一个线性层（效率更高）
    - 因果掩码确保每个位置只看自己和前面的token
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer('mask', mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        # 分头: (B, T, C) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # 注意力计算 + 因果掩码
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

# 测试
attn = CausalSelfAttention(config)
x_test = torch.randn(2, 10, config.d_model)
print(f"  输入: {x_test.shape} → 输出: {attn(x_test).shape}")
print(f"  参数量: {sum(p.numel() for p in attn.parameters()):,}")

# ============================================================
# 第3部分：前馈网络
# ============================================================
print("\n" + "=" * 60)
print("第3部分：前馈网络 (GELU FFN)")
print("=" * 60)

class FeedForward(nn.Module):
    """位置级前馈网络: Linear→GELU→Dropout→Linear→Dropout"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

ffn = FeedForward(config)
print(f"  输入: {x_test.shape} → 输出: {ffn(x_test).shape}")
print(f"  隐层维度: {config.d_ff} (= 4 × {config.d_model})")
print(f"  参数量: {sum(p.numel() for p in ffn.parameters()):,}")

# ============================================================
# 第4部分：Transformer 块 (Pre-Norm)
# ============================================================
print("\n" + "=" * 60)
print("第4部分：Transformer 块 (Pre-Norm)")
print("=" * 60)

class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer 块
    x → LN → Attn → +x → LN → FFN → +x
    Pre-Norm 比 Post-Norm 训练更稳定，是 GPT-2 以来的主流选择
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

block = TransformerBlock(config)
print(f"  输入: {x_test.shape} → 输出: {block(x_test).shape}")
print(f"  参数量: {sum(p.numel() for p in block.parameters()):,}")

# ============================================================
# 第5部分：完整 GPT 模型
# ============================================================
print("\n" + "=" * 60)
print("第5部分：完整 GPT 模型")
print("=" * 60)

class GPT(nn.Module):
    """
    完整 GPT 模型
    token_emb + pos_emb → Dropout → N×TransformerBlock → LayerNorm → lm_head
    token_emb 与 lm_head 权重绑定（减参数+提性能）
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # 权重绑定
        self.token_emb.weight = self.lm_head.weight
        # 初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('net.3.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) 输入token ID
        targets: (B, T) 目标token ID（训练时提供）
        返回:    logits (B, T, V), loss (标量或None)
        """
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"序列长度{T}超过上限{self.config.max_seq_len}"
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.emb_drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1)
        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return total

# ============================================================
# 第6部分：参数量计算与分析
# ============================================================
print("\n" + "=" * 60)
print("第6部分：参数量计算与分析")
print("=" * 60)

print(f"\n  {'配置':<8} {'层数':<5} {'维度':<5} {'头数':<5} {'参数量':<14} {'显存估计'}")
print("  " + "-" * 55)
for name in ['mini', 'small', 'medium']:
    cfg = get_config(name)
    m = GPT(cfg)
    total = m.count_parameters()
    mem = total * 16 / 1024 / 1024  # 训练时约 16 bytes/param
    print(f"  {name:<8} {cfg.n_layers:<5} {cfg.d_model:<5} {cfg.n_heads:<5} "
          f"{total:>12,} {mem:>8.1f} MB")
    del m

# 详细拆解 mini 配置
print("\n  Mini 配置参数量拆解:")
model_mini = GPT(get_config('mini'))
details = {}
for n, p in model_mini.named_parameters():
    cat = n.split('.')[0]
    if cat == 'blocks':
        cat = 'blocks.' + n.split('.')[2]
    details[cat] = details.get(cat, 0) + p.numel()
total_p = model_mini.count_parameters()
for cat, cnt in details.items():
    print(f"    {cat:<25} {cnt:>10,}  ({cnt/total_p*100:>5.1f}%)")
print(f"    {'总计':<25} {total_p:>10,}")

# ============================================================
# 第7部分：前向传播 shape 追踪
# ============================================================
print("\n" + "=" * 60)
print("第7部分：前向传播 shape 追踪")
print("=" * 60)

cfg_mini = get_config('mini')
model_mini = GPT(cfg_mini)
model_mini.eval()
input_ids = torch.randint(0, cfg_mini.vocab_size, (2, 16))

print(f"  输入: batch=2, seq_len=16, vocab={cfg_mini.vocab_size}\n")
with torch.no_grad():
    B, T = input_ids.shape
    pos = torch.arange(T)
    tok_emb = model_mini.token_emb(input_ids)
    pos_emb = model_mini.pos_emb(pos)
    print(f"  input_ids:           {input_ids.shape}")
    print(f"  token_emb:           {tok_emb.shape}")
    print(f"  pos_emb:             {pos_emb.shape}")
    x = tok_emb + pos_emb
    print(f"  tok + pos:           {x.shape}")
    for i, blk in enumerate(model_mini.blocks):
        x = blk(x)
        if i == 0 or i == cfg_mini.n_layers - 1:
            print(f"  Block[{i}]:            {x.shape}")
    x = model_mini.ln_final(x)
    logits = model_mini.lm_head(x)
    print(f"  ln_final:            {x.shape}")
    print(f"  logits:              {logits.shape}")
    # 完整前向
    logits2, loss = model_mini(input_ids, targets=input_ids)
    print(f"\n  验证 loss: {loss.item():.4f} (理论随机: {math.log(cfg_mini.vocab_size):.4f})")

# ============================================================
# 第8部分：与 GPT-2 结构对比
# ============================================================
print("\n" + "=" * 60)
print("第8部分：与 GPT-2 结构对比")
print("=" * 60)
print("""
  ┌──────────────────┬─────────────┬────────────────┐
  │ 特性             │ 我们的实现  │ GPT-2 (124M)   │
  ├──────────────────┼─────────────┼────────────────┤
  │ 架构             │ Decoder-only│ Decoder-only   │
  │ 归一化           │ Pre-Norm    │ Pre-Norm       │
  │ 激活函数         │ GELU        │ GELU           │
  │ 位置编码         │ 可学习      │ 可学习         │
  │ 权重绑定         │ ✓           │ ✓              │
  ├──────────────────┼─────────────┼────────────────┤
  │ 层/维/头         │ 4/128/4     │ 12/768/12      │
  │ 词汇表           │ 256         │ 50,257         │
  │ 最大序列长度     │ 128         │ 1,024          │
  │ 参数量           │ ~1M         │ 124M           │
  └──────────────────┴─────────────┴────────────────┘
  核心架构完全一致！我们的实现就是一个迷你 GPT-2。
""")

# ============================================================
# 第9部分：简单生成验证
# ============================================================
print("=" * 60)
print("第9部分：用未训练模型生成（验证代码正确性）")
print("=" * 60)

@torch.no_grad()
def simple_generate(model, start_ids, max_new_tokens=20):
    """贪心生成：每步选概率最高的 token"""
    model.eval()
    idx = torch.tensor([start_ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.max_seq_len:]
        logits, _ = model(idx_cond)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

generated = simple_generate(model_mini, [65, 66, 67], max_new_tokens=10)
text = ''.join(chr(c) if 32 <= c < 127 else '?' for c in generated)
print(f"  输入: [65,66,67] ('ABC')")
print(f"  输出: {generated} → '{text}'")
print("  模型未训练，输出随机，这是正常的！")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 60)
print("思考题")
print("=" * 60)
print("""
1. 为什么 GPT 使用 Pre-Norm 而不是 Post-Norm？（提示：梯度流）
2. 权重绑定 (token_emb = lm_head) 为什么有效？二者本质上在做什么？
3. 因果掩码 (max_seq_len × max_seq_len) 在 seq_len=8192 时占多少显存？
   有没有更高效的实现？
4. GELU 换成 ReLU 会怎样？现代 LLM 为何倾向 SwiGLU？
5. 参数量主要集中在哪些模块？如何在参数量不变时提升性能？
""")
print("本节完！")
