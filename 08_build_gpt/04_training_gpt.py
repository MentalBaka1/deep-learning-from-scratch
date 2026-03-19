"""
====================================================================
第8章 · 第4节 · 训练GPT
====================================================================
在一段中文文本上用字符级分词训练一个迷你 GPT，
观察 Loss 下降和模型从"胡说八道"到"像模像样"的过程。

【核心流程】
数据准备（滑动窗口）→ 模型构建 → AdamW + Warmup + Cosine Decay → 生成

【前置知识】第8章第3节 - GPT 模型实现
====================================================================
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 第1部分：训练数据
# ============================================================
print("=" * 60)
print("第1部分：准备训练数据")
print("=" * 60)

TRAINING_TEXT = """
深度学习是人工智能的一个重要分支。它通过构建多层神经网络来学习数据中的复杂模式。
深度学习的核心思想是让计算机从数据中自动学习特征，而不需要人工设计特征。
卷积神经网络是深度学习在计算机视觉领域最重要的模型。它通过卷积操作提取图像的局部特征，
然后通过池化操作降低特征维度，最终通过全连接层完成分类任务。
循环神经网络是深度学习在自然语言处理领域最早的尝试。它通过循环结构处理序列数据，
能够记住历史信息。但是循环神经网络存在梯度消失的问题，难以处理长序列。
注意力机制的出现解决了循环神经网络的长距离依赖问题。注意力机制让模型可以直接关注
序列中任意位置的信息，不再受距离限制。
Transformer模型完全基于注意力机制，摒弃了循环结构。它使用自注意力机制处理序列，
可以并行计算，训练效率大幅提升。Transformer是现代大语言模型的基础架构。
GPT是基于Transformer解码器的语言模型。它通过预测下一个词来学习语言的规律。
GPT的训练目标非常简单：给定前面的词，预测下一个词。这种简单的训练目标
却能让模型涌现出翻译、问答、推理、编程等复杂能力。
大语言模型的成功证明了一个深刻的道理：规模足够大的语言模型，仅通过预测下一个词，
就能学会人类语言的深层结构。这是深度学习最令人震撼的发现之一。
""".strip()

print(f"  训练文本长度: {len(TRAINING_TEXT)} 字符")

# ============================================================
# 第2部分：字符级分词器
# ============================================================
print("\n" + "=" * 60)
print("第2部分：字符级分词器")
print("=" * 60)

class CharTokenizer:
    """字符级分词器"""
    def __init__(self, text):
        chars = sorted(set(text))
        self.c2i = {ch: i for i, ch in enumerate(chars)}
        self.i2c = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.c2i[ch] for ch in text if ch in self.c2i]

    def decode(self, ids):
        return ''.join(self.i2c[i] for i in ids if i in self.i2c)

tokenizer = CharTokenizer(TRAINING_TEXT)
all_ids = tokenizer.encode(TRAINING_TEXT)
print(f"  词汇表大小: {tokenizer.vocab_size}")
print(f"  编码序列长度: {len(all_ids)}")

# ============================================================
# 第3部分：滑动窗口数据集
# ============================================================
print("\n" + "=" * 60)
print("第3部分：滑动窗口数据集")
print("=" * 60)

class TextDataset(Dataset):
    """
    滑动窗口数据集
    输入: tokens[i : i+seq_len]
    标签: tokens[i+1 : i+seq_len+1]（右移一位）
    """
    def __init__(self, token_ids, seq_len, stride=1):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        s = idx * self.stride
        return self.data[s:s+self.seq_len], self.data[s+1:s+self.seq_len+1]

SEQ_LEN = 64
dataset = TextDataset(all_ids, seq_len=SEQ_LEN, stride=16)
x0, y0 = dataset[0]
print(f"  序列长度: {SEQ_LEN}, 样本数: {len(dataset)}")
print(f"  x: '{tokenizer.decode(x0[:20].tolist())}'")
print(f"  y: '{tokenizer.decode(y0[:20].tolist())}'")
print(f"  (y 是 x 右移一位)")

# ============================================================
# 第4部分：GPT 模型（紧凑版）
# ============================================================
print("\n" + "=" * 60)
print("第4部分：构建 GPT 模型")
print("=" * 60)

@dataclass
class GPTConfig:
    vocab_size: int = 256
    max_seq_len: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.hd = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)
        mask = torch.tril(torch.ones(c.max_seq_len, c.max_seq_len))
        self.register_buffer('mask', mask.view(1, 1, c.max_seq_len, c.max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        a = self.drop(F.softmax(a.masked_fill(self.mask[:,:,:T,:T]==0, -1e9), dim=-1))
        return self.drop(self.proj((a @ v).transpose(1, 2).contiguous().view(B, T, C)))

class TransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.ffn = nn.Sequential(nn.Linear(c.d_model, c.d_ff, bias=False), nn.GELU(),
                                 nn.Dropout(c.dropout), nn.Linear(c.d_ff, c.d_model, bias=False),
                                 nn.Dropout(c.dropout))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos_emb = nn.Embedding(c.max_seq_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

config = GPTConfig(vocab_size=tokenizer.vocab_size, max_seq_len=SEQ_LEN+1,
                   n_layers=4, n_heads=4, d_model=128, d_ff=512, dropout=0.1)
model = GPT(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"  参数量: {n_params:,} ({config.n_layers}层 {config.d_model}维 {config.n_heads}头)")

# ============================================================
# 第5部分：学习率调度
# ============================================================
print("\n" + "=" * 60)
print("第5部分：Warmup + Cosine Decay 学习率调度")
print("=" * 60)

def get_lr(step, total, max_lr=3e-4, min_lr=1e-5, warmup=100):
    """
    Warmup: 前 warmup 步线性增长到 max_lr（避免初期梯度不稳）
    Cosine Decay: 之后余弦衰减到 min_lr
    """
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

total_vis = 2000
lrs = [get_lr(s, total_vis, warmup=200) for s in range(total_vis)]
plt.figure(figsize=(7, 2.5))
plt.plot(lrs)
plt.axvline(200, color='r', ls='--', alpha=0.5, label='Warmup结束')
plt.xlabel('步数'); plt.ylabel('学习率'); plt.title('Warmup + Cosine Decay')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('c:/project/dl/08_build_gpt/lr_schedule.png', dpi=100); plt.close()
print("  学习率曲线已保存")

# ============================================================
# 第6部分：训练循环
# ============================================================
print("\n" + "=" * 60)
print("第6部分：训练循环")
print("=" * 60)

@torch.no_grad()
def generate_text(model, tokenizer, prompt="深度学习", max_len=80, temperature=0.8):
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_len):
        idx_c = idx[:, -config.max_seq_len:]
        logits, _ = model(idx_c)
        logits = logits[:, -1, :] / temperature
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())

BATCH_SIZE, NUM_EPOCHS, MAX_LR = 32, 50, 3e-4
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
steps_per_epoch = len(loader)
total_steps = NUM_EPOCHS * steps_per_epoch
warmup_steps = total_steps // 10

print(f"  批大小: {BATCH_SIZE}, 轮数: {NUM_EPOCHS}, 总步数: {total_steps}")

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)
loss_hist, lr_hist = [], []
step = 0

print("\n  训练开始...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    ep_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lr = get_lr(step, total_steps, MAX_LR, 1e-5, warmup_steps)
        for g in optimizer.param_groups:
            g['lr'] = lr
        _, loss = model(x, targets=y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item()
        loss_hist.append(loss.item())
        lr_hist.append(lr)
        step += 1

    avg = ep_loss / steps_per_epoch
    if (epoch + 1) % 10 == 0 or epoch == 0:
        ppl = math.exp(min(avg, 20))
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss:{avg:.4f} | PPL:{ppl:.1f} | LR:{lr:.2e}")
        sample = generate_text(model, tokenizer, "深度学习", max_len=50)
        print(f"    生成: {sample[:70]}...\n")

# ============================================================
# 第7部分：训练曲线可视化
# ============================================================
print("=" * 60)
print("第7部分：训练曲线可视化")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(loss_hist, alpha=0.3, color='blue', label='Loss')
w = min(50, len(loss_hist) // 5)
if w > 1:
    sm = [sum(loss_hist[max(0,i-w):i+1])/min(i+1,w+1) for i in range(len(loss_hist))]
    axes[0].plot(sm, 'r-', lw=2, label=f'平滑({w}步)')
axes[0].set_xlabel('步数'); axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(lr_hist, 'g-')
axes[1].set_xlabel('步数'); axes[1].set_ylabel('学习率')
axes[1].set_title('学习率变化'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('c:/project/dl/08_build_gpt/training_curves.png', dpi=100); plt.close()
print("  训练曲线已保存")

# ============================================================
# 第8部分：训练后生成
# ============================================================
print("\n" + "=" * 60)
print("第8部分：训练完成后的生成效果")
print("=" * 60)

for prompt in ["深度学习", "Transformer", "注意力", "GPT"]:
    try:
        text = generate_text(model, tokenizer, prompt, max_len=80, temperature=0.8)
        print(f"  '{prompt}' → {text[:90]}")
    except Exception:
        print(f"  '{prompt}' → 包含未见字符，跳过")

print("\n  不同温度对比 (提示: '深度学习'):")
for t in [0.3, 0.8, 1.5]:
    text = generate_text(model, tokenizer, "深度学习", max_len=60, temperature=t)
    print(f"    T={t}: {text[:80]}")

# ============================================================
# 第9部分：训练总结
# ============================================================
print("\n" + "=" * 60)
print("第9部分：训练总结")
print("=" * 60)
print(f"""
  模型: {n_params:,} 参数 | 数据: {len(TRAINING_TEXT)} 字符 | 轮数: {NUM_EPOCHS}
  最终 Loss: {loss_hist[-1]:.4f} | 理论随机: {math.log(tokenizer.vocab_size):.2f}

  关键观察:
  - 初始Loss接近随机猜测 → 训练后大幅下降
  - 低温度生成更确定但重复，高温度更多样但混乱
  - 核心训练流程与真实GPT完全一致，仅规模不同
""")

# ============================================================
# 思考题
# ============================================================
print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. AdamW 与 Adam 有何区别？权重衰减和 L2 正则化有何不同？
2. 去掉 Warmup 直接用大学习率会怎样？试试看！
3. 梯度裁剪 (grad_clip=1.0) 起什么作用？不做会怎样？
4. 滑动窗口步长 stride=1 和 stride=seq_len 各有何优缺点？
5. 数据量和模型同时放大10倍，Loss会怎样？这和Scaling Law有何关系？
""")
print("本节完！")
