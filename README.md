# 深度学习完整教程：从数学基础到手写 GPT

> **前半程 NumPy 手写**，理解每一行代码背后的数学。
> **后半程 PyTorch 实现**，构建真实可训练的 Transformer 和 GPT。

---

## 你将学到什么

完成本教程后，你将能够：
- 从零推导反向传播，理解神经网络的数学本质
- 独立手写一个完整的 Transformer（Encoder + Decoder）
- 独立手写一个 GPT 模型，并训练它生成文本
- 理解 LoRA、RLHF、DPO、RAG 等现代 LLM 核心技术的原理
- 看懂 Qwen、LLaMA、DeepSeek 等模型的架构设计

---

## 适合谁

- 有编程基础（Python/Java），想系统学习深度学习的开发者
- 用过 LLM API / Agent 框架，想理解底层原理的工程师
- 对"Transformer 到底是什么"感到好奇的任何人

**不适合：** 完全零编程基础的初学者

---

## 学习路线

```
第一阶段：基础构建（纯 NumPy，理解原理）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第 0 章  数学基础          ← 向量、梯度、概率、链式法则
   ↓
第 1 章  机器学习基础      ← 回归、分类、正则化、评估
   ↓
第 2 章  神经网络核心      ← 感知机 → MLP → 反向传播 → 优化器
   ↓
第 3 章  卷积神经网络      ← 卷积、池化、ResNet
   ↓
第 4 章  序列建模          ← RNN → LSTM → Seq2Seq + Attention

第二阶段：过渡到 PyTorch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第 5 章  PyTorch 快速上手  ← Tensor、autograd、nn.Module

第三阶段：手写 Transformer 与 GPT（核心）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第 6 章  注意力机制深度解析 ← QKV、多头注意力、MQA/GQA
   ↓
第 7 章  一步步手写 Transformer ← 位置编码 → FFN → Encoder → Decoder
   ↓
第 8 章  手写 GPT          ← BPE分词 → 语言模型 → GPT → 文本生成

第四阶段：现代 LLM 技术全景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
第 9 章  预训练与微调      ← Scaling Laws、LoRA、SFT
   ↓
第10章  RLHF 与对齐       ← 奖励模型、PPO、DPO、GRPO
   ↓
第11章  LLM 应用技术      ← Prompt工程、RAG、Agent、推理优化
```

---

## 架构演进：为什么需要这些技术？

| 架构/技术 | 解决了什么问题 | 核心思想 |
|-----------|---------------|---------|
| 线性回归 | 如何用数学描述变量关系？ | 最小化预测误差 |
| 逻辑回归 | 如何输出概率而非任意数值？ | sigmoid 压缩到 [0,1] |
| MLP | 单层网络解不了 XOR | 多层 + 非线性激活 |
| 反向传播 | 多层网络如何高效求梯度？ | 链式法则 + 计算图 |
| CNN | 全连接处理图像参数爆炸 | 卷积核局部连接 + 参数共享 |
| ResNet | 网络越深效果越差（退化问题） | 跳跃连接 = 梯度高速公路 |
| RNN | 传统网络无法处理变长序列 | 循环结构传递隐状态 |
| LSTM | RNN 记不住长距离信息 | 门控机制管理记忆 |
| Seq2Seq | 输入输出长度不同怎么办？ | 编码器-解码器框架 |
| Attention | 长序列中如何找到相关信息？ | QKV 加权聚合 |
| Transformer | RNN 太慢，注意力不够 | 纯注意力 + 并行计算 |
| GPT | 如何让模型生成文本？ | 自回归 + Decoder-Only |
| BERT | 单向理解不完整 | 双向 + 完形填空预训练 |
| LoRA | 大模型微调太贵 | 低秩矩阵近似参数更新 |
| RLHF | 模型不听话/不安全 | 人类反馈 + 强化学习 |
| RAG | 模型知识过时/幻觉 | 检索外部知识 + 生成 |

---

## 目录结构

```
dl/
├── README.md
│
├── 00_math_foundations/           # 第0章：数学基础（NumPy）
│   ├── 01_vectors_matrices.py     #   向量与矩阵运算
│   ├── 02_calculus_gradient.py    #   微积分与梯度
│   ├── 03_probability_info.py     #   概率论与信息论
│   └── 04_chain_rule_backprop.py  #   链式法则与计算图
│
├── 01_classical_ml/               # 第1章：机器学习基础（NumPy）
│   ├── 01_linear_regression.py    #   线性回归
│   ├── 02_logistic_regression.py  #   逻辑回归与交叉熵
│   ├── 03_regularization.py       #   正则化与过拟合
│   └── 04_evaluation.py           #   模型评估方法
│
├── 02_neural_networks/            # 第2章：神经网络核心（NumPy）
│   ├── 01_perceptron.py           #   感知机与线性可分
│   ├── 02_activation_functions.py #   激活函数全览
│   ├── 03_mlp_backprop.py         #   MLP 与反向传播（核心！）
│   ├── 04_optimizers.py           #   SGD/Momentum/Adam
│   └── 05_training_tricks.py      #   BatchNorm/Dropout/初始化
│
├── 03_cnn/                        # 第3章：卷积神经网络（NumPy）
│   ├── 01_convolution_pooling.py  #   卷积与池化操作
│   ├── 02_cnn_architectures.py    #   LeNet→ResNet 架构演进
│   └── 03_cnn_practice.py         #   实战：手写数字识别
│
├── 04_sequence_models/            # 第4章：序列建模（NumPy）
│   ├── 01_rnn_fundamentals.py     #   RNN 基础与 BPTT
│   ├── 02_lstm_gru.py             #   LSTM/GRU 门控机制
│   └── 03_seq2seq_attention.py    #   Seq2Seq + Attention（通向Transformer）
│
├── 05_pytorch_basics/             # 第5章：PyTorch 快速上手
│   ├── 01_tensor_autograd.py      #   Tensor 与自动微分
│   ├── 02_nn_module.py            #   nn.Module 与训练流程
│   └── 03_numpy_to_pytorch.py     #   从 NumPy 到 PyTorch 对照
│
├── 06_attention_deep_dive/        # 第6章：注意力机制（PyTorch）
│   ├── 01_attention_intuition.py  #   注意力的直觉与本质
│   ├── 02_scaled_dot_product.py   #   缩放点积注意力
│   ├── 03_multi_head_attention.py #   多头注意力
│   └── 04_attention_variants.py   #   MQA/GQA/因果注意力
│
├── 07_transformer_step_by_step/   # 第7章：手写 Transformer（PyTorch）
│   ├── 01_positional_encoding.py  #   位置编码（正弦 + RoPE）
│   ├── 02_feed_forward.py         #   前馈网络（GELU/SwiGLU）
│   ├── 03_layer_norm_residual.py  #   层归一化与残差连接
│   ├── 04_encoder_block.py        #   Encoder 模块组装
│   ├── 05_decoder_block.py        #   Decoder 模块组装
│   └── 06_full_transformer.py     #   完整 Transformer
│
├── 08_build_gpt/                  # 第8章：手写 GPT（PyTorch）
│   ├── 01_tokenization.py         #   BPE 分词器手写
│   ├── 02_language_model.py       #   语言模型原理
│   ├── 03_gpt_model.py            #   GPT 完整实现
│   ├── 04_training_gpt.py         #   训练 GPT
│   └── 05_generate_text.py        #   文本生成与采样策略
│
├── 09_pretraining_finetuning/     # 第9章：预训练与微调
│   ├── 01_pretraining_paradigm.py #   预训练范式与 Scaling Laws
│   ├── 02_transfer_learning.py    #   迁移学习
│   ├── 03_peft_lora.py            #   LoRA 原理与实现
│   └── 04_sft_practice.py         #   SFT 指令微调实践
│
├── 10_alignment/                  # 第10章：RLHF 与对齐
│   ├── 01_reward_model.py         #   奖励模型
│   ├── 02_ppo_rlhf.py             #   PPO 与 RLHF
│   ├── 03_dpo.py                  #   DPO 直接偏好优化
│   └── 04_alignment_overview.py   #   对齐技术全景
│
└── 11_llm_applications/           # 第11章：LLM 应用技术
    ├── 01_prompt_engineering.py    #   Prompt 工程
    ├── 02_rag_pipeline.py         #   RAG 检索增强生成
    ├── 03_agent_framework.py      #   Agent 框架
    └── 04_inference_optimization.py #  推理优化（KV Cache/量化）
```

---

## 运行环境

```bash
# 第一阶段（第0-4章）：只需 NumPy
pip install numpy matplotlib

# 第二阶段起（第5-11章）：需要 PyTorch
pip install torch torchvision
pip install transformers datasets  # 第9章起需要

# 运行任意文件
python 00_math_foundations/01_vectors_matrices.py
```

---

## 每个文件的阅读方式

每个 .py 文件都遵循统一结构：

1. **【概念讲解】** — 先用大段文字解释核心概念，建立直觉
2. **【为什么需要它？】** — 这个技术解决了什么具体问题
3. **【数学 → 代码】** — 从公式直接翻译为可运行的 Python
4. **【从朴素到高效】** — 先写清晰的 for 循环，再写向量化版本
5. **【实验与可视化】** — 改参数，看效果，培养直觉
6. **【思考题】** — 3-5 道练习，确保真正理解

---

## 学习建议

- **不要跳章**。每章都为下一章铺路，尤其是第4章的 Seq2Seq+Attention 是理解 Transformer 的关键桥梁
- **先读概念，再看代码**。每个文件头部的概念讲解比代码更重要
- **动手改参数**。改学习率、改层数、改注意力头数，观察效果
- **第7-8章是核心**。手写 Transformer 和 GPT 是本教程的灵魂
- 遇到不懂的数学，回到第0章复习
