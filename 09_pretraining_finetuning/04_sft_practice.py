"""
第9章·第4节·SFT指令微调实践

核心概念:
    - 指令微调 (Supervised Fine-Tuning, SFT): 用高质量指令数据微调基座模型
    - 指令数据格式: instruction / input / output 三元组
    - ChatML格式: 多轮对话的标准化模板 (<|im_start|>role\ncontent<|im_end|>)
    - 数据质量: 质量 > 数量，精选数据往往效果更好
    - 训练技巧: 只对回答部分计算损失，忽略指令部分

依赖: pip install torch numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# ============================================================
# 第一部分: 指令微调数据格式
# ============================================================

# Alpaca格式: 最经典的指令微调数据格式
ALPACA_EXAMPLES = [
    {
        "instruction": "将以下英文句子翻译成中文。",
        "input": "The weather is beautiful today.",
        "output": "今天天气很好。",
    },
    {
        "instruction": "解释什么是机器学习。",
        "input": "",  # 有些指令不需要额外输入
        "output": "机器学习是人工智能的一个分支，它使计算机系统能够通过数据和经验自动改进性能，"
                  "而无需进行明确的编程。核心思想是从数据中学习模式和规律。",
    },
    {
        "instruction": "根据给定的关键词写一首短诗。",
        "input": "月亮, 思乡, 秋风",
        "output": "秋风起兮月如钩，\n万里关山思旧游。\n何处笛声催客泪，\n一轮明月照乡愁。",
    },
    {
        "instruction": "对以下文本进行情感分析。",
        "input": "这家餐厅的食物真的太难吃了，服务态度也很差，再也不会来了。",
        "output": "负面情感。用户对餐厅的食物质量和服务态度均表示不满。",
    },
    {
        "instruction": "计算下面表达式的值。",
        "input": "(3 + 5) * 2 - 4 / 2",
        "output": "(3 + 5) * 2 - 4 / 2 = 8 * 2 - 2 = 16 - 2 = 14",
    },
]


def format_alpaca_prompt(example: dict) -> str:
    """
    将Alpaca格式的数据转换为完整的提示文本

    Alpaca提示模板:
        Below is an instruction... (系统说明)
        ### Instruction: (指令)
        ### Input: (可选输入)
        ### Response: (期望回答)
    """
    if example["input"].strip():
        prompt = (
            f"Below is an instruction that describes a task, paired with an input.\n"
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task.\n"
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n"
        )
    return prompt


def demo_alpaca_format():
    """展示Alpaca格式"""
    print("=" * 60)
    print("Alpaca 指令微调数据格式")
    print("=" * 60)

    for i, ex in enumerate(ALPACA_EXAMPLES[:2]):
        prompt = format_alpaca_prompt(ex)
        full_text = prompt + ex["output"]
        print(f"\n--- 样本 {i + 1} ---")
        print(full_text)
        print()


# ============================================================
# 第二部分: ChatML 格式
# ============================================================

"""
ChatML格式说明:
    这是OpenAI提出并被广泛采用的多轮对话格式。
    很多开源模型 (如Qwen, Yi等) 都使用此格式或其变体。

    格式:
        <|im_start|>system
        你是一个有用的助手。<|im_end|>
        <|im_start|>user
        你好！<|im_end|>
        <|im_start|>assistant
        你好！有什么我可以帮助你的吗？<|im_end|>
"""

# 特殊token定义
SPECIAL_TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


def format_chatml(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """
    将消息列表格式化为ChatML格式

    参数:
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        add_generation_prompt: 是否在末尾添加assistant的开头标记 (用于推理)
    """
    formatted = ""
    for msg in messages:
        formatted += f"{SPECIAL_TOKENS['im_start']}{msg['role']}\n{msg['content']}{SPECIAL_TOKENS['im_end']}\n"

    if add_generation_prompt:
        formatted += f"{SPECIAL_TOKENS['im_start']}assistant\n"

    return formatted


def create_sft_labels(
    messages: List[Dict[str, str]],
    tokenize_fn=None,
) -> Dict[str, List[int]]:
    """
    为SFT创建标签: 只对assistant的回复计算损失

    关键点:
        - system和user部分的标签设为 -100 (忽略)
        - 只有assistant部分参与损失计算
        - 这确保模型学习"如何回答"而不是"记忆问题"
    """
    # 简化版: 用字符级别模拟token级别
    full_text = format_chatml(messages)
    # 生成label_mask: 1表示需要计算损失 (assistant部分), 0表示忽略
    label_mask = []
    in_assistant = False
    i = 0

    while i < len(full_text):
        # 检查是否进入assistant段
        marker_start = f"{SPECIAL_TOKENS['im_start']}assistant\n"
        marker_end = SPECIAL_TOKENS["im_end"]

        if full_text[i:].startswith(marker_start):
            # 跳过 "<|im_start|>assistant\n" 标记本身
            for _ in marker_start:
                label_mask.append(0)
                i += 1
            in_assistant = True
            continue

        if in_assistant and full_text[i:].startswith(marker_end):
            # assistant内容结束
            # im_end标记也计入loss (模型需要学会停止生成)
            for _ in marker_end:
                label_mask.append(1)
                i += 1
            in_assistant = False
            continue

        label_mask.append(1 if in_assistant else 0)
        i += 1

    return {
        "text": full_text,
        "label_mask": label_mask,
    }


def demo_chatml_format():
    """展示ChatML格式和标签构造"""
    print("=" * 60)
    print("ChatML 格式与SFT标签")
    print("=" * 60)

    # 单轮对话
    single_turn = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "什么是深度学习？"},
        {"role": "assistant", "content": "深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表征。"},
    ]

    print("\n--- 单轮对话 (ChatML格式) ---")
    print(format_chatml(single_turn))

    # 多轮对话
    multi_turn = [
        {"role": "system", "content": "你是一个Python编程助手。"},
        {"role": "user", "content": "如何反转一个列表？"},
        {"role": "assistant", "content": "可以使用 list[::-1] 或 list.reverse() 方法。"},
        {"role": "user", "content": "这两种方法有什么区别？"},
        {"role": "assistant", "content": "list[::-1] 返回新列表，原列表不变；list.reverse() 原地修改，不返回新列表。"},
    ]

    print("--- 多轮对话 (ChatML格式) ---")
    print(format_chatml(multi_turn))

    # 展示标签掩码
    result = create_sft_labels(single_turn)
    print("--- SFT标签掩码可视化 ---")
    print("(1=计算损失的部分, 0=忽略的部分)\n")

    text = result["text"]
    mask = result["label_mask"]
    # 逐行展示
    line = ""
    mask_line = ""
    for ch, m in zip(text, mask):
        if ch == "\n":
            print(f"  文本: {line}")
            print(f"  掩码: {mask_line}")
            print()
            line = ""
            mask_line = ""
        else:
            line += ch
            mask_line += str(m)
    if line:
        print(f"  文本: {line}")
        print(f"  掩码: {mask_line}")
    print()


# ============================================================
# 第三部分: 简单SFT训练循环
# ============================================================

class TinyLM(nn.Module):
    """极简语言模型，用于演示SFT训练流程"""

    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_emb = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        前向传播

        参数:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len), -100表示忽略的位置

        返回:
            loss (如果提供labels), logits
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device), diagonal=1
        )

        h = self.embedding(input_ids) + self.position_emb(positions)
        h = self.transformer(h, mask=causal_mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # 移位: 用位置t的logits预测位置t+1的token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # 忽略非assistant部分
            )

        return loss, logits


def prepare_sft_batch(
    texts: List[str],
    label_masks: List[List[int]],
    max_len: int = 64,
) -> Dict[str, torch.Tensor]:
    """
    准备SFT训练batch

    简化版: 使用字符的ASCII码作为token ID
    实际中应使用tokenizer (如SentencePiece, BPE等)
    """
    input_ids_batch = []
    labels_batch = []

    for text, mask in zip(texts, label_masks):
        # 字符级别tokenization (简化)
        token_ids = [ord(c) % 256 for c in text[:max_len]]
        label_mask = mask[:max_len]

        # 构造labels: 非assistant部分设为-100
        labels = []
        for tid, m in zip(token_ids, label_mask):
            labels.append(tid if m == 1 else -100)

        # 填充到max_len
        pad_len = max_len - len(token_ids)
        token_ids += [0] * pad_len
        labels += [-100] * pad_len

        input_ids_batch.append(token_ids)
        labels_batch.append(labels)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


def demo_sft_training():
    """SFT训练循环演示"""
    print("=" * 60)
    print("SFT训练循环演示")
    print("=" * 60)

    torch.manual_seed(42)

    # 构造训练数据
    conversations = [
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Thank you!"},
            {"role": "assistant", "content": "You are welcome!"},
        ],
    ]

    # 格式化并创建标签
    texts = []
    masks = []
    for conv in conversations:
        result = create_sft_labels(conv)
        texts.append(result["text"])
        masks.append(result["label_mask"])

    batch = prepare_sft_batch(texts, masks, max_len=128)

    # 初始化模型
    model = TinyLM(vocab_size=256, d_model=64, n_heads=4, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # 学习率调度: 带warmup的余弦退火
    total_steps = 100
    warmup_steps = 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # 线性warmup
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))  # 余弦退火

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    print("\n开始SFT训练...")
    losses = []
    for step in range(total_steps):
        loss, _ = model(batch["input_ids"], batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 20 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step + 1}/{total_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

    print(f"\n训练完成! 最终损失: {losses[-1]:.4f}")
    print(f"损失下降: {losses[0]:.4f} → {losses[-1]:.4f}")
    print()


# ============================================================
# 第四部分: 数据质量讨论
# ============================================================

def discuss_data_quality():
    """数据质量对SFT效果的影响"""
    print("=" * 60)
    print("SFT数据质量讨论")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────────┐
│  核心原则: 数据质量 >> 数据数量                               │
│                                                              │
│  LIMA论文 (Zhou et al., 2023):                               │
│    仅用1000条精选数据就能微调出高质量对话模型！               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  高质量数据的特征:                                    │     │
│  │    1. 指令多样性 — 覆盖不同任务类型和难度              │     │
│  │    2. 回答准确性 — 事实正确，逻辑清晰                  │     │
│  │    3. 格式规范性 — 结构化输出，标点正确                │     │
│  │    4. 安全性     — 不包含有害或偏见内容                │     │
│  │    5. 复杂度适中 — 包含简单和复杂问题的合理分布        │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  常见数据问题:                                                │
│    x 回答质量不一 → 模型学到"平庸"的回复风格               │
│    x 指令同质化   → 模型在某些任务上过拟合，其他任务欠拟合  │
│    x 存在错误标注 → 模型学到错误知识 (比脏数据更危险)       │
│    x 数据过多     → 边际收益递减，甚至引入噪声              │
│                                                              │
│  最佳实践:                                                    │
│    1. 先用少量高质量数据微调，评估效果                        │
│    2. 根据评估结果有针对性地补充数据                          │
│    3. 使用GPT-4等强模型生成种子数据，再人工审核              │
│    4. 数据去重 (语义级别去重，不只是精确匹配)                │
│    5. 建立数据质量评分体系                                    │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  SFT vs 后续阶段:                                            │
│                                                              │
│  预训练 → SFT → RLHF/DPO                                    │
│    │       │       │                                          │
│    │       │       └─ 对齐人类偏好 (有用性/安全性)            │
│    │       └─ 教会模型"对话格式"和"遵循指令"               │
│    └─ 学习语言知识和世界知识                                  │
│                                                              │
│  SFT的目标不是注入新知识，而是激活预训练中已有的能力！        │
└──────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    demo_alpaca_format()
    demo_chatml_format()
    demo_sft_training()
    discuss_data_quality()

    # ===========================================================
    # 思考题
    # ===========================================================
    print("=" * 60)
    print("思考题")
    print("=" * 60)
    print("""
1. SFT训练时为什么只对assistant部分计算损失？
   如果对整个序列 (包括system和user) 都计算损失会怎样？

2. LIMA论文证明1000条数据就够了。这是否意味着数据量不重要？
   在什么情况下需要更多数据？

3. ChatML格式中的特殊标记 (<|im_start|>, <|im_end|>) 起什么作用？
   如果去掉这些标记会有什么影响？

4. 为什么说"SFT是激活能力而不是注入知识"？
   如果基座模型不具备某种能力，SFT能教会它吗？

5. 在构建SFT数据集时，指令的多样性和回答的质量哪个更重要？
   如何在两者之间取得平衡？
""")
