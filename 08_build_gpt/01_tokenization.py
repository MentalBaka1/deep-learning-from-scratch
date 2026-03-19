"""
====================================================================
第8章 · 第1节 · 文本分词：从字符到BPE
====================================================================
分词（Tokenization）是将文本切分为模型可处理的最小单元——token，
BPE 算法在字符级和词级之间找到了最佳平衡点，是 GPT 系列的标配。

【核心概念】
1. 字符级分词：每个字符一个 token，词汇表小，序列长
2. 词级分词：每个词一个 token，OOV（未登录词）问题严重
3. BPE 算法：从字符出发，反复合并最高频的相邻 token 对
4. 特殊 token：PAD（填充）、BOS（句首）、EOS（句尾）、UNK（未知）
====================================================================
"""

import collections
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1部分：字符级分词
# ============================================================
print("=" * 60)
print("第1部分：字符级分词")
print("=" * 60)

class CharTokenizer:
    """字符级分词器：每个字符映射为一个整数 ID"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}

    def build_vocab(self, text):
        """从文本构建字符词汇表，预留特殊token位置"""
        chars = sorted(set(text))
        special = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.char2idx = dict(special)
        self.idx2char = {v: k for k, v in special.items()}
        for i, ch in enumerate(chars):
            self.char2idx[ch] = i + 4
            self.idx2char[i + 4] = ch

    def encode(self, text):
        return [self.char2idx.get(ch, 3) for ch in text]

    def decode(self, ids):
        return ''.join(self.idx2char.get(i, '<UNK>') for i in ids)

    @property
    def vocab_size(self):
        return len(self.char2idx)

# 演示
sample_text = "深度学习改变了世界，GPT是其中最耀眼的明星。"
char_tok = CharTokenizer()
char_tok.build_vocab(sample_text)

encoded = char_tok.encode(sample_text)
decoded = char_tok.decode(encoded)
print(f"  原文: {sample_text}")
print(f"  编码: {encoded}")
print(f"  解码: {decoded}")
print(f"  词汇表大小: {char_tok.vocab_size}, 序列长度: {len(encoded)}")

# OOV 测试
oov_text = "AI很强"
oov_decoded = char_tok.decode(char_tok.encode(oov_text))
print(f"  OOV测试: '{oov_text}' → '{oov_decoded}'")

# ============================================================
# 第2部分：词级分词与 OOV 问题
# ============================================================
print("\n" + "=" * 60)
print("第2部分：词级分词与 OOV 问题")
print("=" * 60)

class WordTokenizer:
    """词级分词器（简化版），用于演示 OOV 问题"""
    def __init__(self, text):
        words = sorted(set(text))
        self.w2i = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        for i, w in enumerate(words):
            self.w2i[w] = i + 4
        self.i2w = {v: k for k, v in self.w2i.items()}

    def encode(self, text):
        return [self.w2i.get(ch, 3) for ch in text]

    def decode(self, ids):
        return ''.join(self.i2w.get(i, '<UNK>') for i in ids)

word_tok = WordTokenizer("我爱深度学习")
print("  OOV 问题演示:")
for t in ["我爱深度学习", "我爱机器学习", "强化学习很难"]:
    enc = word_tok.encode(t)
    unk_count = sum(1 for x in enc if x == 3)
    print(f"    '{t}' → '{word_tok.decode(enc)}' (UNK: {unk_count})")

# ============================================================
# 第3部分：BPE 算法完整实现
# ============================================================
print("\n" + "=" * 60)
print("第3部分：BPE (Byte Pair Encoding) 算法")
print("=" * 60)

class BPETokenizer:
    """
    完整 BPE 分词器
    训练: 统计相邻 token 对频率 → 合并最高频对 → 重复
    编码: 按训练时的合并顺序依次应用合并规则
    """
    def __init__(self):
        self.merges = {}          # (a, b) → merged
        self.merge_order = []     # 合并顺序
        self.vocab = {}
        self.inv_vocab = {}

    def _count_pairs(self, seqs):
        counts = collections.Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def _apply_merge(self, seqs, pair, new_tok):
        result = []
        for seq in seqs:
            new_seq, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                    new_seq.append(new_tok)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, text, num_merges=20, verbose=True):
        """训练 BPE：从字符开始，反复合并最高频的相邻对"""
        # 加词首标记
        words = text.replace(' ', ' ▁').split()
        if not text.startswith(' '):
            words[0] = '▁' + words[0]
        freqs = collections.Counter(words)
        seqs = []
        for word, freq in freqs.items():
            for _ in range(freq):
                seqs.append(list(word))
        # 初始词汇表
        all_chars = set(ch for seq in seqs for ch in seq)
        special = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        self.vocab = {t: i for i, t in enumerate(special)}
        for ch in sorted(all_chars):
            self.vocab[ch] = len(self.vocab)
        init_size = len(self.vocab)
        if verbose:
            print(f"  初始词汇表: {init_size}, 目标: {init_size + num_merges}")
        sizes = [init_size]
        for step in range(num_merges):
            pairs = self._count_pairs(seqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_tok = best[0] + best[1]
            self.merges[best] = new_tok
            self.merge_order.append(best)
            self.vocab[new_tok] = len(self.vocab)
            seqs = self._apply_merge(seqs, best, new_tok)
            sizes.append(len(self.vocab))
            if verbose and (step < 8 or (step+1) % 5 == 0):
                print(f"    第{step+1:2d}步: '{best[0]}'+'{best[1]}' → '{new_tok}' (频率:{pairs[best]})")
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"  最终词汇表: {len(self.vocab)}")
        return sizes

    def encode(self, text):
        """按训练顺序依次应用合并规则"""
        t = text.replace(' ', ' ▁')
        if not text.startswith(' '):
            t = '▁' + t
        tokens = list(t)
        for pair in self.merge_order:
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(self.merges[pair])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, 3) for t in tokens]

    def decode(self, ids):
        text = ''.join(self.inv_vocab.get(i, '?') for i in ids)
        text = text.replace('▁', ' ')
        return text.lstrip(' ')

# ============================================================
# 第4部分：BPE 训练演示
# ============================================================
print("\n" + "=" * 60)
print("第4部分：BPE 训练演示")
print("=" * 60)

train_text = ("深度学习 深度学习 机器学习 机器学习 自然语言处理 自然语言处理 "
              "深度学习模型 机器学习算法 人工智能 人工智能 神经网络 神经网络模型")

bpe = BPETokenizer()
vocab_sizes = bpe.train(train_text, num_merges=25, verbose=True)

print("\n  编码解码测试:")
for s in ["深度学习", "机器学习算法", "自然语言处理"]:
    ids = bpe.encode(s)
    tokens = [bpe.inv_vocab[i] for i in ids]
    print(f"    '{s}' → {tokens} → ids:{ids} → '{bpe.decode(ids)}'")

# 词汇表增长可视化
plt.figure(figsize=(7, 3))
plt.plot(range(len(vocab_sizes)), vocab_sizes, 'b-o', markersize=3)
plt.xlabel('合并步骤')
plt.ylabel('词汇表大小')
plt.title('BPE 训练：词汇表增长')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('c:/project/dl/08_build_gpt/bpe_vocab_growth.png', dpi=100)
plt.close()
print("  词汇表增长曲线已保存")

# ============================================================
# 第5部分：特殊 token
# ============================================================
print("\n" + "=" * 60)
print("第5部分：特殊 token")
print("=" * 60)

for tok, desc in [('<PAD>', '填充，对齐不同长度序列'),
                  ('<BOS>', '句首标记，提示模型开始生成'),
                  ('<EOS>', '句尾标记，模型遇到它停止生成'),
                  ('<UNK>', '未知token，词汇表外的兜底')]:
    print(f"  {tok}: {desc}")

print("\n  PAD 示例（对齐序列长度）:")
for s in ["你好", "深度学习很有趣", "AI"]:
    padded = list(s) + ['<PAD>'] * (7 - len(s))
    print(f"    '{s}' → {padded}")

# ============================================================
# 第6部分：分词策略对比
# ============================================================
print("\n" + "=" * 60)
print("第6部分：不同分词策略的影响对比")
print("=" * 60)

cmp_text = "深度学习是人工智能的核心"
char_toks = list(cmp_text)
bpe_ids = bpe.encode(cmp_text)
bpe_toks = [bpe.inv_vocab.get(i, '?') for i in bpe_ids]

print(f"  原文: '{cmp_text}'")
print(f"  字符级: {char_toks} (长度:{len(char_toks)})")
print(f"  BPE:    {bpe_toks} (长度:{len(bpe_toks)})")
print(f"  BPE/字符 长度比: {len(bpe_toks)/len(char_toks):.2f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
for ax, data, title in zip(axes,
    [[len(char_toks), len(bpe_toks)], [char_tok.vocab_size, len(bpe.vocab)]],
    ['序列长度', '词汇表大小']):
    ax.bar(['字符级', 'BPE'], data, color=['steelblue', 'coral'])
    ax.set_title(title)
    for i, v in enumerate(data):
        ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('c:/project/dl/08_build_gpt/tokenization_compare.png', dpi=100)
plt.close()
print("  对比图已保存")

# ============================================================
# 第7部分：真实分词器简介
# ============================================================
print("\n" + "=" * 60)
print("第7部分：真实世界的分词器")
print("=" * 60)
print("""
  GPT-2/3/4 使用字节级 BPE (Byte-level BPE):
  - 在 UTF-8 字节上做 BPE，而非字符上
  - 任何文本都能编码，不需要 <UNK>
  - GPT-2 词汇表: 50,257 | GPT-4: ~100,000

  安装 tiktoken 后可直接使用:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode("深度学习")
""")

# ============================================================
# 思考题
# ============================================================
print("=" * 60)
print("思考题")
print("=" * 60)
print("""
1. BPE 处理中文时效率为何不如英文？（提示：中文字符数量）
2. 合并次数从 30 增到 3000，词汇表和序列长度如何变化？
3. 为什么 GPT-2 选字节级 BPE 而非字符级 BPE？
4. <EOS> 在训练和推理时分别起什么作用？去掉会怎样？
5. 中英混合数据的词汇表大小该设多大？太大太小各有何问题？
""")
print("本节完！")
