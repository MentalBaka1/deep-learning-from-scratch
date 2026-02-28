"""
==============================================================
第5章 第4节：BERT —— 双向预训练语言模型
==============================================================

【为什么需要它？】
在 BERT 之前，语言模型（如 GPT-1）只能"看左边"：
  "The cat sat on the [?]"
  → 模型从左到右预测，只知道 "The cat sat on the"

但人类理解语言时是双向的：
  "The [MASK] sat on the mat" → 看左边"The"和右边"mat"才能猜出"cat"
  "Bank account" vs "river bank" → 需要看整个上下文才能消歧

BERT（Bidirectional Encoder Representations from Transformers）的解决方案：
  1. 双向 Transformer Encoder（同时看左边和右边）
  2. 创新的预训练任务（不用标注数据！）
  3. 预训练完成后，用少量标注数据微调到各种任务

【两个预训练任务】
  1. Masked LM（完形填空）：随机遮住 15% 的词，让模型猜
     "The [MASK] sat on the mat" → 预测 "cat"
  2. Next Sentence Prediction（NSP）：判断两句话是否相邻
     "[CLS] 我爱苹果 [SEP] 苹果很好吃 [SEP]" → 是相邻的 ✓
     "[CLS] 我爱苹果 [SEP] 量子力学很难 [SEP]" → 不相邻 ✗

【存在理由】
解决问题：单向语言模型理解上下文不完整，标注数据昂贵
核心思想：双向 Transformer + 大规模无监督预训练 → 通用语言表示
          然后用少量标注数据微调，像"打地基"然后"盖各种房子"
==============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

# ============================================================
# Part 1: BERT 的核心创新
# ============================================================
print("=" * 50)
print("Part 1: BERT 的设计哲学")
print("=" * 50)

"""
BERT vs GPT 的根本区别：

GPT（单向，自回归）：
  训练目标：P(token_t | token_1, ..., token_{t-1})
  只看"已经生成的"内容
  好处：天然适合生成任务（写文章、续写）
  缺陷：理解任务时上下文不完整

BERT（双向，编码器）：
  训练目标：P(token_t | 所有其他 token)
  同时看左边和右边
  好处：理解任务（分类、问答、NER）效果极好
  缺陷：不直接适合生成任务

为什么 BERT 不能"从左到右预测"？
  如果双向 Transformer 做自回归预测，会"看到答案"（信息泄露）
  所以 BERT 用"完形填空"（MLM）代替"从左到右预测"
"""

print("BERT 输入格式：")
print("  词汇表特殊符号：")
print("    [CLS] = 分类 token（放在序列开头，输出用于分类任务）")
print("    [SEP] = 句子分隔符（分开两个句子，或标记句子结尾）")
print("    [MASK] = 被遮盖的 token（预训练时随机遮盖 15% 的词）")
print("    [PAD] = 填充 token（让不同长度的句子对齐）")
print()
print("  输入 = 词嵌入 + 位置嵌入 + 句子段嵌入（哪个句子）")
print()

# ============================================================
# Part 2: Masked LM —— 完形填空预训练
# ============================================================
print("=" * 50)
print("Part 2: Masked LM（MLM）—— 双向完形填空")
print("=" * 50)

"""
MLM 的遮盖策略（精心设计！）：
  对于被选中的 15% token：
    - 80% 的概率替换为 [MASK]
    - 10% 的概率替换为随机词
    - 10% 的概率保持不变

为什么不全部换成 [MASK]？
  微调阶段没有 [MASK] token！
  如果训练时全是 [MASK]，模型会"学到"只有遇到 [MASK] 才需要预测
  混合使用让模型每个位置都要"随时准备预测"

MLM 的损失：
  只计算被遮盖位置的损失（不计算未遮盖的位置）
  loss = CrossEntropy(model_output[mask_positions], true_tokens[mask_positions])
"""

class TinyBERTConfig:
    """简化版 BERT 配置（用于演示）"""
    vocab_size = 30      # 词汇表大小（真实 BERT 约 30,000）
    max_seq_len = 20     # 最大序列长度（真实 BERT 是 512）
    d_model = 32         # 隐藏层维度（真实 BERT-base 是 768）
    n_heads = 4          # 注意力头数（真实 BERT-base 是 12）
    n_layers = 2         # 层数（真实 BERT-base 是 12）
    d_ff = 64            # FFN 维度（真实 BERT-base 是 3072）

    # 特殊 token 的 ID
    CLS_ID = 0
    SEP_ID = 1
    MASK_ID = 2
    PAD_ID = 3

config = TinyBERTConfig()

def create_masked_input(token_ids, mask_prob=0.15):
    """
    对输入 token 序列应用 MLM 遮盖策略
    返回：遮盖后的输入、遮盖位置的 mask、原始 label
    """
    input_ids = token_ids.copy()
    labels = np.full_like(token_ids, -1)  # -1 表示"不计算损失"

    # 不遮盖特殊 token
    special_tokens = {config.CLS_ID, config.SEP_ID, config.PAD_ID}

    for i, token in enumerate(token_ids):
        if token in special_tokens:
            continue
        if np.random.random() < mask_prob:
            labels[i] = token  # 记录原始 token（目标）
            r = np.random.random()
            if r < 0.8:
                input_ids[i] = config.MASK_ID  # 80%：换成 [MASK]
            elif r < 0.9:
                input_ids[i] = np.random.randint(4, config.vocab_size)  # 10%：随机词
            # else: 10%：保持不变

    mask = (labels != -1)  # 哪些位置被选中
    return input_ids, mask, labels

# 演示遮盖过程
example_tokens = np.array([0, 8, 12, 5, 19, 7, 1])  # [CLS] 词... [SEP]
print("MLM 遮盖演示：")
print(f"  原始 token: {example_tokens}")
np.random.seed(7)
masked_input, mask, labels = create_masked_input(example_tokens)
print(f"  遮盖后输入: {masked_input}")
print(f"  遮盖位置:   {mask}")
print(f"  原始标签:   {labels}  （-1=不计算，其他=需要预测的原始词）")
print(f"  被遮盖的词: {example_tokens[mask]}")

# ============================================================
# Part 3: 简化版 BERT 实现
# ============================================================
print("\nPart 3: 简化版 BERT 实现")
print("=" * 50)

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
    PE[:, 0::2] = np.sin(position / div_term)
    PE[:, 1::2] = np.cos(position / div_term)
    return PE

def attention_forward(Q, K, V, mask=None):
    d_k = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores)
    return weights @ V, weights

class TinyBERT:
    """
    简化版 BERT（用于教学演示，非完整实现）
    保留核心结构：词嵌入 + 位置编码 + Transformer Encoder + MLM head
    """
    def __init__(self, config):
        self.config = config
        d = config.d_model

        # 词嵌入矩阵（词汇表 × 隐藏维度）
        self.token_embedding = np.random.randn(config.vocab_size, d) * 0.02
        # 句子段嵌入（0 = 第一句，1 = 第二句）
        self.segment_embedding = np.random.randn(2, d) * 0.02

        # 注意力层参数（简化：只存 W_Q/W_K/W_V/W_O）
        scale = 1.0 / np.sqrt(d)
        self.layers = []
        for _ in range(config.n_layers):
            layer = {
                'W_Q': np.random.randn(d, d) * scale,
                'W_K': np.random.randn(d, d) * scale,
                'W_V': np.random.randn(d, d) * scale,
                'W_O': np.random.randn(d, d) * scale,
                'W1': np.random.randn(d, config.d_ff) * scale,
                'b1': np.zeros(config.d_ff),
                'W2': np.random.randn(config.d_ff, d) * scale,
                'b2': np.zeros(d),
                'ln1_gamma': np.ones(d),
                'ln1_beta': np.zeros(d),
                'ln2_gamma': np.ones(d),
                'ln2_beta': np.zeros(d),
            }
            self.layers.append(layer)

        # MLM 输出头（预测被遮盖的词）
        self.mlm_W = np.random.randn(d, config.vocab_size) * 0.02
        self.mlm_b = np.zeros(config.vocab_size)

        # NSP 输出头（判断两句是否相邻）
        self.nsp_W = np.random.randn(d, 2) * 0.02
        self.nsp_b = np.zeros(2)

    def layer_norm(self, x, gamma, beta, eps=1e-6):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mu) / np.sqrt(var + eps) + beta

    def transformer_layer(self, X, layer):
        """单个 Transformer Encoder 层的前向传播"""
        batch, seq_len, d = X.shape
        H = self.config.n_heads
        d_k = d // H

        # 多头注意力
        Q = X @ layer['W_Q']
        K = X @ layer['W_K']
        V = X @ layer['W_V']

        def split(x):
            return x.reshape(batch, seq_len, H, d_k).transpose(0, 2, 1, 3)

        Q, K, V = split(Q), split(K), split(V)
        Q_f = Q.reshape(batch*H, seq_len, d_k)
        K_f = K.reshape(batch*H, seq_len, d_k)
        V_f = V.reshape(batch*H, seq_len, d_k)

        out, _ = attention_forward(Q_f, K_f, V_f)
        out = out.reshape(batch, H, seq_len, d_k).transpose(0, 2, 1, 3).reshape(batch, seq_len, d)
        out = out @ layer['W_O']

        # 残差 + LayerNorm
        X = self.layer_norm(X + out, layer['ln1_gamma'], layer['ln1_beta'])

        # FFN
        ffn = np.maximum(0, X @ layer['W1'] + layer['b1']) @ layer['W2'] + layer['b2']
        X = self.layer_norm(X + ffn, layer['ln2_gamma'], layer['ln2_beta'])

        return X

    def forward(self, token_ids, segment_ids=None):
        """
        token_ids: (batch, seq_len) 整数数组
        segment_ids: (batch, seq_len) 0或1，表示第几个句子
        """
        batch, seq_len = token_ids.shape

        # 词嵌入 + 位置编码 + 段嵌入
        X = self.token_embedding[token_ids]  # (batch, seq, d)
        X += positional_encoding(seq_len, self.config.d_model)
        if segment_ids is not None:
            X += self.segment_embedding[segment_ids]

        # Transformer Encoder 层
        for layer in self.layers:
            X = self.transformer_layer(X, layer)

        # MLM head：预测每个位置的 token
        mlm_logits = X @ self.mlm_W + self.mlm_b  # (batch, seq, vocab_size)

        # NSP head：用 [CLS] 位置预测句子关系
        cls_output = X[:, 0, :]  # (batch, d)
        nsp_logits = cls_output @ self.nsp_W + self.nsp_b  # (batch, 2)

        return mlm_logits, nsp_logits, X

# 初始化模型
bert = TinyBERT(config)
print(f"Tiny BERT 参数量估算：")
total_params = (config.vocab_size * config.d_model +  # 词嵌入
                config.n_layers * (4 * config.d_model**2 +  # QKV + O
                                   2 * config.d_model * config.d_ff))  # FFN
print(f"  词嵌入：{config.vocab_size * config.d_model:,}")
print(f"  每层 Transformer：{4*config.d_model**2 + 2*config.d_model*config.d_ff:,}")
print(f"  总参数：{total_params:,}")
print(f"  真实 BERT-base 参数：110,000,000（1.1亿）")

# 运行前向传播
batch_size = 2
seq_len = 12
token_ids = np.random.randint(4, config.vocab_size, (batch_size, seq_len))
token_ids[:, 0] = config.CLS_ID
token_ids[:, -1] = config.SEP_ID
segment_ids = np.zeros((batch_size, seq_len), dtype=int)

mlm_logits, nsp_logits, hidden = bert.forward(token_ids, segment_ids)
print(f"\n前向传播输出：")
print(f"  输入形状：{token_ids.shape}")
print(f"  隐藏状态：{hidden.shape}")
print(f"  MLM logits：{mlm_logits.shape}  → 每个位置预测 vocab 大小的分数")
print(f"  NSP logits：{nsp_logits.shape}  → [CLS] 输出，2分类（相邻/不相邻）")

# ============================================================
# Part 4: 预训练的直觉 —— 为什么这样有效？
# ============================================================
print("\nPart 4: 预训练的直觉")
print("=" * 50)

"""
BERT 为什么强大？用"万能地基"来类比：

  工人（模型）在盖楼之前先学习了：
    - 如何搬砖（语法规则）
    - 砖的重量手感（词的语义）
    - 墙的结构（句子结构）
    ← 这就是在大量文本上做 MLM 预训练

  然后要建不同的楼（下游任务）：
    - 情感分析（判断楼是否漂亮）
    - 命名实体识别（找出楼的地标）
    - 问答系统（楼在哪里）
    ← 这就是用少量标注数据微调

  因为地基（预训练权重）已经包含了丰富的语言知识，
  只需要很少的标注数据就能学会新任务！

具体化：预训练后 BERT 的能力
  1. 词向量包含语义（king - man + woman ≈ queen）
  2. 能消歧（bank 在金融 vs 河流语境中表示不同）
  3. 能理解语法（主谓宾关系）
  4. 能处理长距离依赖（代词 it 指向前面的名词）
"""

print("预训练流程（大规模，无监督）：")
print("  1. 收集大量文本（维基百科、书籍...）")
print("  2. 随机遮盖 15% 的词")
print("  3. 让 BERT 预测被遮盖的词")
print("  4. 重复数百万次更新参数")
print()
print("微调流程（小规模，有监督）：")
print("  1. 拿预训练好的 BERT 权重（已经\"懂语言\"）")
print("  2. 加一个任务专用的输出层")
print("  3. 用少量标注数据更新所有参数（学习率很小）")
print("  4. BERT 把通用语言知识迁移到具体任务")
print()
print("BERT 在 NLP benchmark 上的成绩：")
print("  GLUE score：发布时提升约 7%（相对之前最好模型）")
print("  SQuAD 问答：超越人类水平")
print("  情感分析、NER、文本相似等：全面刷新记录")

# ============================================================
# Part 5: 可视化 BERT 的注意力模式
# ============================================================
print("\nPart 5: 可视化 BERT 注意力模式")
print("=" * 50)

"""
研究人员分析训练好的 BERT 注意力头，发现：
  - 某些头专门关注[CLS]和[SEP]（句子边界感知）
  - 某些头关注相邻词（语法关系）
  - 某些头关注代词对应的名词
  - 某些头关注语义相关的词

这证明了多头注意力确实学到了有意义的语言学特征！
"""

# 模拟一个简单的注意力分析
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('BERT 直觉可视化', fontsize=14)

# 1. 预训练 vs 微调流程图
ax = axes[0][0]
ax.axis('off')
pipeline_text = """
BERT 使用流程：

【阶段1：预训练（大规模，无监督）】

大量文本 → BERT训练 → 通用语言表示

训练任务：
  MLM: The [MASK] ate the fish → 猫
  NSP: 句子A，句子B → 是否相邻

【阶段2：微调（小规模，有监督）】

预训练BERT + 少量标注数据 → 任务专用模型

情感分析: [CLS]output → 正/负
命名实体: 每个位置output → 实体类型
问答系统: 找出答案的起止位置
"""
ax.text(0.05, 0.95, pipeline_text, transform=ax.transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax.set_title('预训练 → 微调流程')

# 2. MLM 遮盖策略可视化
ax = axes[0][1]
strategies = ['替换为[MASK]\n(80%)', '替换为随机词\n(10%)', '保持原词\n(10%)']
counts = [80, 10, 10]
colors = ['steelblue', 'orange', 'green']
bars = ax.bar(strategies, counts, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('比例 (%)')
ax.set_title('MLM 遮盖策略\n（15%的词被选中）')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{count}%', ha='center', fontsize=11, fontweight='bold')

# 3. BERT vs GPT 结构对比（文字说明）
ax = axes[0][2]
ax.axis('off')
comparison_text = """
BERT vs GPT 对比：

          BERT          GPT
方向：   双向（←→）   单向（→）
架构：   Encoder      Decoder
预训练：  MLM+NSP      因果语言模型
强项：   理解任务      生成任务

BERT 更好的任务：
  ✓ 文本分类（情感、主题）
  ✓ 命名实体识别
  ✓ 问答（阅读理解）
  ✓ 文本相似度

GPT 更好的任务：
  ✓ 文章续写
  ✓ 对话生成
  ✓ 代码补全
  ✓ few-shot 学习（GPT-3+）
"""
ax.text(0.05, 0.95, comparison_text, transform=ax.transAxes,
        fontsize=8.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.set_title('BERT vs GPT')

# 4. 模拟注意力模式（不同头的特化）
ax = axes[1][0]
words = ['[CLS]', '猫', '追', '了', '狗', '[SEP]']
seq = len(words)
# 模拟：某个头专注于"主谓关系"
attn_head1 = np.eye(seq) * 0.3
attn_head1[2, 1] = 0.6  # "追" 关注 "猫"（主语）
attn_head1[2, 4] = 0.5  # "追" 关注 "狗"（宾语）
attn_head1 = softmax(attn_head1)
im = ax.imshow(attn_head1, cmap='Blues', vmin=0, vmax=attn_head1.max())
ax.set_xticks(range(seq))
ax.set_yticks(range(seq))
ax.set_xticklabels(words, rotation=45)
ax.set_yticklabels(words)
ax.set_title('注意力头示例\n（模拟：动词关注主宾语）')
plt.colorbar(im, ax=ax, fraction=0.046)

# 5. 嵌入空间语义（模拟词向量的语义聚类）
ax = axes[1][1]
np.random.seed(12)
# 模拟不同类型词的嵌入（降到2D展示）
categories = {
    '动物': (['猫', '狗', '鸟', '鱼'], np.array([1, 1]) + np.random.randn(4, 2)*0.3),
    '食物': (['米饭', '面条', '苹果', '香蕉'], np.array([-1, 1]) + np.random.randn(4, 2)*0.3),
    '动词': (['跑', '跳', '吃', '睡'], np.array([0, -1]) + np.random.randn(4, 2)*0.3),
}
colors_cat = ['blue', 'red', 'green']
for (cat, (words_cat, coords)), color in zip(categories.items(), colors_cat):
    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=100, alpha=0.7, label=cat)
    for word, (x, y) in zip(words_cat, coords):
        ax.text(x+0.05, y+0.05, word, fontsize=8)
ax.legend()
ax.set_title('BERT 嵌入空间（示意）\n相似词聚集在一起')
ax.grid(True, alpha=0.3)

# 6. 微调策略对比
ax = axes[1][2]
tasks = ['情感\n分析', '命名实体\n识别', '问答\n系统', '文本\n相似度']
bert_scores = [95.5, 92.3, 91.2, 87.8]  # 模拟 BERT 微调后的分数
baseline_scores = [88.2, 83.1, 82.5, 79.3]  # 模拟基线分数

x = np.arange(len(tasks))
width = 0.35
ax.bar(x - width/2, baseline_scores, width, label='基线（无预训练）', color='orange', alpha=0.7)
ax.bar(x + width/2, bert_scores, width, label='BERT 微调', color='steelblue', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylabel('准确率 (%)')
ax.set_ylim(75, 100)
ax.set_title('BERT 微调效果对比\n（模拟数据，说明趋势）')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('05_transformer_attention/bert_intuition.png', dpi=80, bbox_inches='tight')
print("图片已保存：05_transformer_attention/bert_intuition.png")
plt.show()

# ============================================================
# Part 6: 实际体验 MLM 任务
# ============================================================
print("\nPart 6: 体验 MLM 预训练任务")
print("=" * 50)

"""
用一个玩具数据集演示 MLM 预训练的过程：
  词汇表：数字 0-9（用数字代替真实词）
  任务：给定遮盖后的序列，预测原始数字

  这是极简化的演示，真实 BERT 在几百亿词上训练
"""

# 生成玩具 MLM 数据集
def generate_mlm_batch(batch_size, seq_len, vocab_size, special_size=4):
    """
    生成随机的 MLM 训练样本
    返回：遮盖后的输入、遮盖位置、原始标签
    """
    # 生成随机序列（排除特殊 token）
    tokens = np.random.randint(special_size, vocab_size, (batch_size, seq_len))
    tokens[:, 0] = 0   # [CLS]
    tokens[:, -1] = 1  # [SEP]

    masked_tokens = tokens.copy()
    mask = np.zeros((batch_size, seq_len), dtype=bool)

    for b in range(batch_size):
        for i in range(1, seq_len - 1):  # 不遮盖 [CLS] 和 [SEP]
            if np.random.random() < 0.15:
                mask[b, i] = True
                r = np.random.random()
                if r < 0.8:
                    masked_tokens[b, i] = 2  # [MASK]
                elif r < 0.9:
                    masked_tokens[b, i] = np.random.randint(special_size, vocab_size)

    return masked_tokens, mask, tokens

# 演示 MLM 批次
np.random.seed(42)
masked_input, mask, original = generate_mlm_batch(
    batch_size=3, seq_len=8, vocab_size=config.vocab_size
)

print("MLM 训练批次示例（3个样本）：")
token_names = {0: '[CLS]', 1: '[SEP]', 2: '[MASK]', 3: '[PAD]'}
for b in range(3):
    orig_str = [token_names.get(t, str(t)) for t in original[b]]
    masked_str = [token_names.get(t, str(t)) for t in masked_input[b]]
    print(f"\n  样本 {b+1}：")
    print(f"    原始：  {' '.join(orig_str)}")
    print(f"    遮盖后：{' '.join(masked_str)}")
    mask_pos = np.where(mask[b])[0]
    if len(mask_pos) > 0:
        print(f"    遮盖位置：{mask_pos}，需要预测：{original[b][mask_pos]}")
    else:
        print(f"    本次没有遮盖任何词")

print("\n在真实 BERT 中：")
print("  - 在数十亿词上重复此过程")
print("  - 每步更新参数，让模型越来越会\"猜词\"")
print("  - 猜词迫使模型理解语言的语义和语法")
print("  - 猜完词后，模型拥有了丰富的语言知识")
print("  → 这就是 BERT 强大的秘密：大规模无监督预训练！")

# ============================================================
# 思考题
# ============================================================
print("\n" + "=" * 50)
print("思考题")
print("=" * 50)
print("""
1. 【双向 vs 单向的权衡】
   为什么 GPT 系列（单向）在大模型时代反超了 BERT（双向）？
   提示：单向模型可以做生成任务，双向不行。
   当模型足够大时（GPT-3：1750亿参数），few-shot 能力是什么？
   现代大模型（GPT-4、Claude）采用哪种架构？为什么？

2. 【NSP 的争议】
   后来的研究（RoBERTa，2019）发现：
   去掉 NSP 任务，只用 MLM，效果反而更好！
   你认为 NSP 的问题是什么？
   （提示：NSP 的负样本是随机句子，太容易了；
    下游任务需要的是"细粒度"的句子关系，不是"随机vs相邻"）

3. 【遮盖策略的设计】
   BERT 的 80/10/10 策略为什么比 100%[MASK] 更好？
   如果全部替换为[MASK]：
   - 训练时：模型只需要对[MASK]位置输出有意义的预测
   - 微调时：没有[MASK]，模型不知道"对所有位置都要关注"
   这叫做"预训练-微调差距"（pretrain-finetune discrepancy）。
   80/10/10 如何缓解这个问题？
""")
