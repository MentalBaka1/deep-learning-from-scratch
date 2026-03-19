"""
第11章·第2节·RAG检索增强生成
核心: 为什么需要RAG, Embedding, 向量相似度, Chunking策略, 完整RAG流程

RAG (Retrieval-Augmented Generation) 通过检索外部知识来增强LLM的生成能力，
解决LLM知识截止、幻觉、领域知识不足等问题。
本节从零实现一个完整的RAG Pipeline。
"""

import math
import re
from typing import List, Dict, Tuple
from collections import Counter

# ============================================================
# 第一部分：为什么需要RAG？
# ============================================================
#
# LLM的局限性：
#   1. 知识截止日期 —— 训练数据有时间边界，无法获取最新信息
#   2. 幻觉问题 —— 模型可能编造不存在的事实
#   3. 领域专业性 —— 通用模型对特定领域知识覆盖不足
#   4. 上下文长度限制 —— 无法一次性输入所有文档
#
# RAG的解决思路：
#   Query → 检索相关文档 → 将文档作为上下文拼入Prompt → LLM基于上下文生成回答
#
# 好处：知识可实时更新、回答有据可查、无需重新训练模型

# ============================================================
# 第二部分：简单Embedding实现
# ============================================================

class BagOfWordsEmbedding:
    """
    词袋模型Embedding：将文本表示为词频向量。
    生产环境中应使用预训练模型（如text-embedding-ada-002、BGE等）。
    """
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """简单分词：英文按单词、中文逐字切分"""
        tokens = re.findall(r'[a-zA-Z]+', text.lower())   # 英文单词
        tokens += re.findall(r'[\u4e00-\u9fff]', text)     # 中文逐字
        return tokens

    def fit(self, documents: List[str]):
        """从文档集合中构建词表"""
        all_tokens = set()
        for doc in documents:
            all_tokens.update(self._tokenize(doc))
        self.vocabulary = {w: i for i, w in enumerate(sorted(all_tokens))}
        self._fitted = True
        print(f"  [Embedding] 词表大小：{len(self.vocabulary)}")

    def encode(self, text: str) -> List[float]:
        """将文本编码为L2归一化的词袋向量"""
        assert self._fitted, "请先调用fit()构建词表"
        vec = [0.0] * len(self.vocabulary)
        for word, count in Counter(self._tokenize(text)).items():
            if word in self.vocabulary:
                vec[self.vocabulary[word]] = float(count)
        norm = math.sqrt(sum(v * v for v in vec)) + 1e-8
        return [v / norm for v in vec]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量编码"""
        return [self.encode(t) for t in texts]


class RandomProjectionEmbedding:
    """随机投影Embedding：将词袋向量投影到低维空间（用于对比演示）"""
    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.bow = BagOfWordsEmbedding()
        self._proj = None
        import random
        self.rng = random.Random(seed)

    def fit(self, documents: List[str]):
        self.bow.fit(documents)
        vs = len(self.bow.vocabulary)
        self._proj = [[self.rng.gauss(0, 1/math.sqrt(self.dim))
                        for _ in range(self.dim)] for _ in range(vs)]

    def encode(self, text: str) -> List[float]:
        bow_vec = self.bow.encode(text)
        result = [0.0] * self.dim
        for i, val in enumerate(bow_vec):
            if val != 0:
                for j in range(self.dim):
                    result[j] += val * self._proj[i][j]
        norm = math.sqrt(sum(v*v for v in result)) + 1e-8
        return [v / norm for v in result]

# ============================================================
# 第三部分：向量相似度计算
# ============================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    余弦相似度: cos(a,b) = (a·b) / (|a|×|b|)
    值域[-1,1]，1=完全相同方向，0=正交，-1=完全相反
    """
    assert len(vec_a) == len(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a*a for a in vec_a)) + 1e-8
    nb = math.sqrt(sum(b*b for b in vec_b)) + 1e-8
    return dot / (na * nb)

def top_k_search(query_vec: List[float], doc_vecs: List[List[float]],
                 k: int = 3) -> List[Tuple[int, float]]:
    """在文档向量库中搜索与query最相似的top-k文档"""
    sims = [(i, cosine_similarity(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]

# ============================================================
# 第四部分：文档Chunking策略
# ============================================================

def chunk_by_fixed_size(text: str, chunk_size: int = 100,
                        overlap: int = 20) -> List[str]:
    """
    固定大小分块（带重叠）。
    优点：实现简单，块大小均匀。缺点：可能在语句中间截断。
    """
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def chunk_by_sentence(text: str, max_sentences: int = 3,
                      overlap_sentences: int = 1) -> List[str]:
    """
    按句子分块（带重叠）。
    优点：保持语义完整性。缺点：块大小不均匀。
    """
    sentences = [s.strip() for s in re.split(r'(?<=[。！？.!?])', text) if s.strip()]
    chunks, start = [], 0
    step = max_sentences - overlap_sentences
    while start < len(sentences):
        chunks.append("".join(sentences[start:start + max_sentences]))
        start += step
    return chunks

def chunk_by_paragraph(text: str) -> List[str]:
    """按段落分块（以双换行符分隔）"""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

# ============================================================
# 第五部分：完整RAG Pipeline
# ============================================================

class SimpleRAGPipeline:
    """
    完整的RAG流水线：
      1. 文档加载与分块  2. 向量化并建立索引
      3. 查询时检索相关块  4. 将检索结果拼入Prompt  5. 调用LLM生成回答
    """
    def __init__(self, chunk_method: str = "sentence", top_k: int = 2):
        self.embedding = BagOfWordsEmbedding()
        self.chunks: List[str] = []
        self.chunk_vectors: List[List[float]] = []
        self.top_k = top_k
        self.chunk_method = chunk_method

    def ingest(self, documents: List[str]):
        """文档摄入：分块 → 向量化 → 存储"""
        print("\n  [RAG] 开始文档摄入...")
        all_chunks = []
        for doc in documents:
            if self.chunk_method == "fixed":
                all_chunks += chunk_by_fixed_size(doc, 80, 15)
            elif self.chunk_method == "sentence":
                all_chunks += chunk_by_sentence(doc, 2, 1)
            else:
                all_chunks += chunk_by_paragraph(doc)
        self.chunks = all_chunks
        print(f"  [RAG] 共分为 {len(self.chunks)} 个块")
        self.embedding.fit(self.chunks)
        self.chunk_vectors = self.embedding.encode_batch(self.chunks)
        print(f"  [RAG] 向量化完成，维度：{len(self.chunk_vectors[0])}")

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """检索与查询最相关的文档块"""
        qv = self.embedding.encode(query)
        results = top_k_search(qv, self.chunk_vectors, k=self.top_k)
        return [(self.chunks[idx], score) for idx, score in results]

    def build_prompt(self, query: str, contexts: List[Tuple[str, float]]) -> str:
        """将检索结果拼接为RAG Prompt"""
        ctx_str = "\n---\n".join(f"[相关度:{s:.3f}] {t}" for t, s in contexts)
        return (f"请根据以下参考资料回答问题。无相关信息则说明无法回答。\n\n"
                f"【参考资料】\n{ctx_str}\n\n【问题】{query}\n\n【回答】")

    def query(self, question: str) -> str:
        """完整RAG查询：检索 → 构建Prompt → 生成回答"""
        print(f"\n  [RAG] 查询：{question}")
        contexts = self.retrieve(question)
        for text, score in contexts:
            print(f"  [RAG] 检索到（{score:.3f}）：{text[:50]}...")
        prompt = self.build_prompt(question, contexts)
        answer = f"根据参考资料：{contexts[0][0][:40]}..."  # 模拟生成
        print(f"  [RAG] 回答：{answer[:60]}...")
        return answer

# ============================================================
# 第六部分：运行演示
# ============================================================

print("=" * 60)
print("【Embedding & 相似度演示】")
sample_docs = [
    "Python是一种广泛使用的高级编程语言。Python支持多种编程范式。",
    "深度学习是机器学习的一个子领域。深度学习使用多层神经网络。",
    "向量数据库用于存储和检索高维向量。常见的有Milvus和Pinecone。",
    "大语言模型通过海量文本数据训练而成。GPT和BERT是著名的大语言模型。",
]
embedder = BagOfWordsEmbedding()
embedder.fit(sample_docs)
vecs = embedder.encode_batch(sample_docs)

print("\n  【向量相似度矩阵】")
print("        ", "  ".join(f"Doc{i}" for i in range(len(sample_docs))))
for i in range(len(sample_docs)):
    row = [f"{cosine_similarity(vecs[i], vecs[j]):.3f}" for j in range(len(sample_docs))]
    print(f"  Doc{i}  " + "  ".join(row))

print("\n" + "=" * 60)
print("【Chunking策略对比】")
long_text = ("自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。"
             "近年来，大语言模型推动了NLP的快速发展。Transformer架构是大语言模型的基础。"
             "注意力机制是Transformer的核心创新。")
fixed_chunks = chunk_by_fixed_size(long_text, 30, 5)
sent_chunks = chunk_by_sentence(long_text, 2, 1)
print(f"  固定大小分块(size=30) → {len(fixed_chunks)} 块")
for i, c in enumerate(fixed_chunks):
    print(f"    [{i}] ({len(c)}字符) {c}")
print(f"  句子级分块(max=2) → {len(sent_chunks)} 块")
for i, c in enumerate(sent_chunks):
    print(f"    [{i}] ({len(c)}字符) {c}")

print("\n" + "=" * 60)
print("【完整RAG Pipeline演示】")
knowledge_base = [
    "PyTorch是由Facebook开发的开源深度学习框架。它以动态计算图著称，支持GPU加速。"
    "PyTorch提供了autograd自动微分引擎。",
    "TensorFlow是Google开发的深度学习框架。TensorFlow有丰富的生态系统。",
    "Transformer模型由Attention Is All You Need论文提出。它完全基于注意力机制。"
    "Transformer的核心是多头自注意力和位置编码。",
    "RAG是检索增强生成的缩写。它结合了信息检索和文本生成技术。"
    "RAG先从知识库检索相关文档，再将文档作为上下文传给LLM生成回答。",
]
rag = SimpleRAGPipeline(chunk_method="sentence", top_k=2)
rag.ingest(knowledge_base)
for q in ["PyTorch有什么特点？", "什么是Transformer？", "RAG是什么？"]:
    rag.query(q)
    print()

# ============================================================
# 第七部分：评估指标 & 思考题
# ============================================================

print("=" * 60)
print("""
【RAG评估指标】
  检索质量：Recall@K, Precision@K, MRR, nDCG
  生成质量：忠实度(Faithfulness), 相关性(Relevance), 有害内容检测
  常用框架：RAGAS, TruLens, LangSmith

【思考题】
  1. 词袋模型Embedding有什么局限？为什么生产环境需要预训练Embedding模型？

  2. Chunk大小如何影响RAG效果？太大或太小各有什么问题？

  3. 除了余弦相似度，还有哪些向量相似度度量方式？各适合什么场景？

  4. RAG和微调(Fine-tuning)各自的优缺点是什么？什么场景适合用RAG？

  5. 如何解决RAG中检索结果与问题不相关的问题？
     有哪些改进策略（如Reranking、HyDE等）？
""")

if __name__ == "__main__":
    print("本节演示完毕。完整RAG Pipeline已从零实现。")
    print("生产环境建议：sentence-transformers + FAISS/Milvus + LLM API")
