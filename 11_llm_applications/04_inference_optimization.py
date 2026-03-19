"""
第11章·第4节·推理优化
核心: KV Cache详解, 量化(INT8/INT4), Flash Attention, 投机采样, 持续批处理/vLLM

LLM推理面临的挑战：模型参数量大、自回归生成速度慢、显存占用高。
本节通过PyTorch代码演示各种推理优化技术的原理和效果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import random
from typing import Optional, Tuple, List

torch.manual_seed(42)

# ============================================================
# 第一部分：KV Cache 详解与对比
# ============================================================

class SimpleAttention(nn.Module):
    """简单多头注意力层，用于演示KV Cache"""
    def __init__(self, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        """
        KV Cache原理：
          自回归生成中，每次只新增一个token。之前token的K、V不变可缓存复用。
          无缓存：每步重新计算所有K、V → O(n²)
          有缓存：只计算新token的K、V并拼接 → O(n)
        """
        B, T, _ = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=1)
            V = torch.cat([kv_cache[1], V], dim=1)
        new_cache = (K, V) if use_cache else None

        def reshape(t, s):
            return t.view(B, s, self.n_heads, self.head_dim).transpose(1, 2)
        Q, K, V = reshape(Q, T), reshape(K, K.shape[1]), reshape(V, V.shape[1])

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), new_cache


def benchmark_kv_cache():
    """对比有无KV Cache的推理速度"""
    print("=" * 60)
    print("【KV Cache 速度对比】")
    d_model, seq_len, gen_steps = 128, 64, 32
    attn = SimpleAttention(d_model=d_model)
    attn.eval()

    # 无KV Cache
    with torch.no_grad():
        tokens = torch.randn(1, seq_len, d_model)
        start = time.perf_counter()
        for _ in range(gen_steps):
            tokens = torch.cat([tokens, torch.randn(1, 1, d_model)], dim=1)
            out, _ = attn(tokens, use_cache=False)
        t_no = time.perf_counter() - start

    # 有KV Cache
    with torch.no_grad():
        tokens = torch.randn(1, seq_len, d_model)
        start = time.perf_counter()
        out, cache = attn(tokens, use_cache=True)  # Prefill
        for _ in range(gen_steps):
            out, cache = attn(torch.randn(1, 1, d_model), kv_cache=cache, use_cache=True)
        t_yes = time.perf_counter() - start

    print(f"  序列长度={seq_len}，生成步数={gen_steps}")
    print(f"  无Cache：{t_no*1000:.2f}ms | 有Cache：{t_yes*1000:.2f}ms | 加速：{t_no/max(t_yes,1e-8):.1f}x")
    ck, cv = cache
    mem = (ck.nelement() + cv.nelement()) * 4
    print(f"  KV Cache显存：{mem/1024:.1f}KB，形状：K={list(ck.shape)}\n")

benchmark_kv_cache()

# ============================================================
# 第二部分：量化 (Quantization) —— Float32 → INT8/INT4
# ============================================================

def symmetric_quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    对称量化：float32 → int8
    scale = max(|tensor|) / 127，零点始终为0
    """
    abs_max = tensor.abs().max().item()
    scale = abs_max / 127.0 if abs_max > 0 else 1.0
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return quantized, scale

def dequantize_int8(quantized: torch.Tensor, scale: float) -> torch.Tensor:
    """反量化：int8 → float32"""
    return quantized.float() * scale

print("=" * 60)
print("【量化演示：Float32 → INT8 / INT4】")
weight = torch.randn(64, 64)
print(f"  原始权重：{weight.dtype}，形状={list(weight.shape)}")
print(f"  原始大小：{weight.nelement()*4/1024:.1f}KB (float32)")

# INT8量化
q8, s8 = symmetric_quantize_int8(weight)
dq8 = dequantize_int8(q8, s8)
err8 = (weight - dq8).abs()
print(f"\n  INT8量化：scale={s8:.6f}，大小={q8.nelement()/1024:.1f}KB，压缩比=4x")
print(f"    MAE={err8.mean():.6f}，最大误差={err8.max():.6f}，"
      f"相对误差={err8.mean()/weight.abs().mean()*100:.2f}%")

# INT4量化（概念演示）
s4 = weight.abs().max().item() / 7.0
q4 = torch.clamp(torch.round(weight / s4), -8, 7)
dq4 = q4 * s4
err4 = (weight - dq4).abs()
print(f"\n  INT4量化：压缩比=8x")
print(f"    MAE={err4.mean():.6f}，相对误差={err4.mean()/weight.abs().mean()*100:.2f}%")

# 量化矩阵乘法精度对比
print(f"\n  【量化矩阵乘法精度】")
x = torch.randn(16, 64)
r_fp32 = x @ weight.T
r_int8 = x @ dq8.T
r_int4 = x @ dq4.T
print(f"    FP32均值={r_fp32.mean():.4f}")
print(f"    INT8均值={r_int8.mean():.4f}（误差={(r_fp32-r_int8).abs().mean():.6f}）")
print(f"    INT4均值={r_int4.mean():.4f}（误差={(r_fp32-r_int4).abs().mean():.6f}）")
print()

# ============================================================
# 第三部分：Flash Attention 概念与伪代码
# ============================================================

print("=" * 60)
print("【Flash Attention 原理】")
print("""
  标准Attention问题：存储N×N注意力矩阵→O(N²)显存，大量HBM读写→IO瓶颈

  Flash Attention核心思想：
    1. 分块计算(Tiling)：Q/K/V分成小块，逐块在SRAM中计算
    2. 内核融合(Kernel Fusion)：softmax+matmul融合为一个CUDA内核
    3. 重算代替存储：前向不存储注意力矩阵，反向重新计算
    4. 在线Softmax：不需要全局max即可分块计算softmax

  复杂度：计算O(N²d)不变，显存从O(N²)降为O(N)
  效果：训练加速2-4x，支持更长序列，数学等价无精度损失
""")

def flash_attention_pseudocode(Q: torch.Tensor, K: torch.Tensor,
                                V: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """
    Flash Attention的分块计算伪代码（Python模拟）。
    展示在线softmax的分块计算逻辑，真正实现需要CUDA内核。
    """
    N, d = Q.shape
    output = torch.zeros(N, d)
    for i in range(0, N, block_size):
        q_blk = Q[i:i+block_size]
        blen = q_blk.shape[0]
        row_max = torch.full((blen,), float('-inf'))
        row_sum = torch.zeros(blen)
        blk_out = torch.zeros(blen, d)
        for j in range(0, N, block_size):
            k_blk, v_blk = K[j:j+block_size], V[j:j+block_size]
            scores = q_blk @ k_blk.T / math.sqrt(d)
            blk_max = scores.max(dim=-1).values
            new_max = torch.maximum(row_max, blk_max)
            old_scale = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            blk_out = blk_out * old_scale.unsqueeze(-1) + exp_scores @ v_blk
            row_sum = row_sum * old_scale + exp_scores.sum(dim=-1)
            row_max = new_max
        output[i:i+blen] = blk_out / row_sum.unsqueeze(-1)
    return output

# 验证正确性
N, d = 32, 16
Q, K, V = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
standard = F.softmax(Q @ K.T / math.sqrt(d), dim=-1) @ V
flash = flash_attention_pseudocode(Q, K, V, block_size=8)
diff = (standard - flash).abs().max().item()
print(f"  标准 vs 分块Attention 最大误差：{diff:.8f} → {'通过' if diff < 1e-5 else '未通过'}\n")

# ============================================================
# 第四部分：投机采样 (Speculative Decoding)
# ============================================================

print("=" * 60)
print("【投机采样 (Speculative Decoding)】")

class MockDraftModel:
    """草稿模型（小模型，快但精度低）"""
    def predict_probs(self, ctx: List[int]) -> torch.Tensor:
        torch.manual_seed(sum(ctx) % 100)
        return F.softmax(torch.randn(100), dim=-1)

class MockTargetModel:
    """目标模型（大模型，慢但精度高）"""
    def predict_probs(self, ctx: List[int]) -> torch.Tensor:
        torch.manual_seed(sum(ctx) % 100 + 1)
        return F.softmax(torch.randn(100) * 1.5, dim=-1)

def speculative_decoding_step(context: List[int], draft: MockDraftModel,
                               target: MockTargetModel, gamma: int = 4) -> List[int]:
    """
    投机采样一步：
      1. 小模型快速生成gamma个候选token
      2. 大模型一次验证所有候选
      3. 接受与大模型分布一致的token，拒绝后从修正分布重采样
    保证最终输出分布与纯大模型采样完全一致。
    """
    # 小模型生成候选
    draft_tokens, draft_probs = [], []
    ctx = list(context)
    for _ in range(gamma):
        p = draft.predict_probs(ctx)
        t = torch.multinomial(p, 1).item()
        draft_tokens.append(t)
        draft_probs.append(p)
        ctx.append(t)

    # 大模型验证
    target_probs = []
    for i in range(gamma):
        target_probs.append(target.predict_probs(context + draft_tokens[:i]))

    # 逐个接受/拒绝
    accepted = []
    for i in range(gamma):
        tok = draft_tokens[i]
        p_d = draft_probs[i][tok].item()
        p_t = target_probs[i][tok].item()
        if random.random() < min(1.0, p_t / max(p_d, 1e-10)):
            accepted.append(tok)
        else:
            # 从修正分布重采样
            correction = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            correction = correction / (correction.sum() + 1e-10)
            accepted.append(torch.multinomial(correction, 1).item())
            break
    return accepted

draft_m, target_m = MockDraftModel(), MockTargetModel()
context = [1, 5, 10, 20]
print(f"  初始上下文：{context}，gamma=4\n")
total_accepted = 0
for step in range(5):
    acc = speculative_decoding_step(context, draft_m, target_m, gamma=4)
    total_accepted += len(acc)
    context.extend(acc)
    print(f"  第{step+1}步：草稿4个 → 接受{len(acc)}个 {acc}")
avg = total_accepted / 5
print(f"\n  平均每步接受：{avg:.1f}个token，理论加速≈{avg:.1f}x\n")

# ============================================================
# 第五部分：优化技术对比 & 持续批处理
# ============================================================

print("=" * 60)
print("""
【LLM推理优化技术对比】
  技术          优化维度      加速效果    代价
  ─────────────────────────────────────────────────
  KV Cache      计算量        2-10x       额外显存
  INT8量化      显存+带宽     1.5-2x      微小精度损失
  INT4量化      显存+带宽     2-3x        可感知精度损失
  Flash Attn    IO+显存       2-4x        需CUDA支持
  投机采样      延迟          2-3x        需草稿模型
  持续批处理    吞吐量        5-20x       系统复杂度
  PagedAttn     显存碎片      显存利用+    实现复杂

【持续批处理 & vLLM】
  传统批处理：短序列必须等长序列完成，GPU利用率低
  持续批处理：序列完成后立即替换新请求，不等待整batch
  PagedAttention：将KV Cache分成固定大小"页"管理，
    类似操作系统虚拟内存，支持共享前缀，减少显存浪费
""")

# ============================================================
# 第六部分：综合Benchmark
# ============================================================

print("=" * 60)
print("【综合性能测试 — 矩阵乘法 FP32 vs 模拟INT8】")
print(f"  {'大小':>8} | {'FP32(ms)':>10} | {'INT8(ms)':>10} | {'误差':>10}")
print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")
for size in [128, 256, 512]:
    w = torch.randn(size, size)
    x = torch.randn(32, size)
    q_w, s = symmetric_quantize_int8(w)
    dq_w = dequantize_int8(q_w, s)

    start = time.perf_counter()
    for _ in range(100):
        _ = x @ w.T
    t_fp = (time.perf_counter() - start) * 10

    start = time.perf_counter()
    for _ in range(100):
        _ = x @ dq_w.T
    t_q = (time.perf_counter() - start) * 10

    err = (x @ w.T - x @ dq_w.T).abs().mean().item()
    print(f"  {size:>8} | {t_fp:>10.3f} | {t_q:>10.3f} | {err:>10.6f}")
print("  注：模拟INT8（反量化后FP32计算）。真实INT8需TensorRT等硬件支持。\n")

# ============================================================
# 思考题
# ============================================================

print("""
【思考题】
  1. KV Cache显存占用与哪些因素有关？
     128层、8192上下文的模型，KV Cache有多大？

  2. INT8和INT4量化各适用于什么场景？
     GPTQ、AWQ、GGUF这些量化方法有什么区别？

  3. Flash Attention为什么能在不改变计算量的情况下大幅提速？
     IO复杂度和计算复杂度的区别是什么？

  4. 投机采样的接受率取决于什么？如何选择合适的草稿模型？

  5. vLLM的PagedAttention如何解决KV Cache显存碎片问题？
     它与操作系统虚拟内存机制有什么相似之处？
""")

if __name__ == "__main__":
    print("本节演示完毕。所有推理优化技术均通过简化代码展示核心原理。")
    print("生产环境推荐：vLLM / TensorRT-LLM / llama.cpp / ONNX Runtime")
