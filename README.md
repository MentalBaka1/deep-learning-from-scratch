# 深度学习从零开始 🧠

> 仅用 **numpy + matplotlib**，从数学基础到现代架构，手写每一行代码。

---

## 学习路径

```
第0章  数学基础 (00_math_python/)
  ↓
第1章  经典机器学习 (01_classical_ml/)
  ↓
第2章  神经网络基础 (02_neural_networks/)
  ↓
第3章  卷积神经网络 (03_cnn/)
  ↓
第4章  序列模型 RNN/LSTM (04_rnn_sequence/)
  ↓
第5章  注意力与Transformer (05_transformer_attention/)
  ↓
第6章  生成模型 [选修] (06_generative/)
第7章  强化学习 (07_rl/)
```

---

## 为什么这些架构被发明出来？

| 架构 | 面临的问题 | 解决方案 |
|------|-----------|---------|
| 线性回归 | 如何用数学描述变量关系？ | 最小化误差平方和 |
| 逻辑回归 | 线性输出不能直接当概率 | sigmoid 压缩到 [0,1] |
| 决策树 | 线性模型捕捉不了非线性边界 | 递归信息增益分裂 |
| MLP | 单层感知机解不了 XOR 等问题 | 加隐层 + 非线性激活 |
| CNN | 全连接处理图像参数爆炸 | 卷积核共享参数 |
| ResNet | 深层网络梯度消失，越深越差 | 跳跃连接提供梯度高速公路 |
| RNN | 传统网络无法处理可变长序列 | 循环隐状态传递历史信息 |
| LSTM/GRU | RNN 记不住长距离依赖 | 门控机制主动管理记忆 |
| Transformer | RNN 串行慢，长距离依赖仍难 | 全注意力机制，任意位置直接交互 |
| BERT | 语言模型只看左边，理解不完整 | 双向 Transformer + 完形填空预训练 |
| Autoencoder | 如何无监督学习数据表示？ | 编码压缩 + 解码重建 |
| VAE | AE 潜空间不连续无法生成 | 学习潜空间概率分布 |
| Q-Learning | 规则未知时如何试错学习？ | Q 表 + 贝尔曼方程迭代 |

---

## 目录结构

```
dl/
├── 00_math_python/
│   ├── 01_vectors_matrices.py    # 向量/矩阵：空间中的箭头和变换
│   ├── 02_derivatives.py         # 导数/梯度：从斜率到方向
│   ├── 03_probability_stats.py   # 概率统计：不确定性的语言
│   └── 04_chain_rule.py          # 链式法则：反向传播的数学心脏
│
├── 01_classical_ml/
│   ├── 01_linear_regression.py   # 用直线拟合数据
│   ├── 02_logistic_regression.py # 概率分类器
│   ├── 03_decision_tree.py       # 20个问题游戏
│   └── 04_svm_intuition.py       # 找最宽的走廊
│
├── 02_neural_networks/
│   ├── 01_perceptron.py          # 最简单的神经元
│   ├── 02_mlp_backprop.py        # 手写反向传播（核心！）
│   ├── 03_activation_functions.py # 为什么需要非线性
│   └── 04_optimizers.py          # SGD/Momentum/Adam 动画对比
│
├── 03_cnn/
│   ├── 01_convolution_viz.py     # 卷积：手电筒扫描图像
│   ├── 02_cnn_from_scratch.py    # 手写 CNN 完整实现
│   ├── 03_digit_recognizer.py    # 实战：识别数字 1-6
│   └── 04_resnet_residual.py     # 残差：梯度高速公路
│
├── 04_rnn_sequence/
│   ├── 01_rnn_basics.py          # 有记忆的神经网络
│   ├── 02_lstm_gru.py            # 门控记忆管理
│   └── 03_sequence_prediction.py # 实战：时间序列预测
│
├── 05_transformer_attention/
│   ├── 01_attention_qkv.py       # QKV：英文阅读理解类比
│   ├── 02_self_attention.py      # 句子的自我审视
│   ├── 03_transformer_encoder.py # 完整 Transformer Encoder
│   └── 04_bert_intuition.py      # 完形填空预训练
│
├── 06_generative/
│   ├── 01_autoencoder.py         # 压缩与重建
│   └── 02_vae_intuition.py       # 学会"生成"
│
└── 07_rl/
    ├── 01_rl_concepts.py         # 训狗：奖励驱动学习
    ├── 02_qlearning.py           # 贝尔曼方程到代码
    └── 03_gridworld.py           # 实战：格世界寻路
```

---

## 运行要求

```bash
pip install numpy matplotlib
python 00_math_python/01_vectors_matrices.py
```

**没有 PyTorch，没有 TensorFlow，只有 numpy。**

---

## 每个文件的阅读方式

每个文件都遵循以下结构：

1. **【为什么需要它？】** — 这个技术解决了什么具体问题
2. **【生活类比】** — 用日常例子建立直觉
3. **【数学公式 → Python 代码】** — 公式不是纸面上的，是可以运行的
4. **【从朴素到高效】** — 先写 for 循环（清晰），再写向量化（快速）
5. **【实验与可视化】** — 改参数，看效果变化
6. **【思考题】** — 3 道练习，加深理解

---

## 学习建议

- 每个文件先**读注释**，再**运行代码**，再**改参数实验**
- 遇到不懂的数学，回到 `00_math_python/` 复习
- `02_neural_networks/02_mlp_backprop.py` 是整个课程的数学核心，务必搞懂
- 带 `gradient_check` 的文件：验证你修改的代码是否正确
