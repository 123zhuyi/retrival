# 需求背景

ZJ客户需求实现DeepseekV3.2模型中Lighting Indexer的triton前向算子。DeepseekV3.2 中的 indexer 被称为 Lightning Indexer，是DSA (DeepSeek Sparse Attention) 稀疏注意力机制的核心组件，其主要作用是大幅提升长文本处理效率，通过 "轻量级筛选替代全局计算" 的策略，显著降低模型训练和推理成本。

# 需求分析

客户面对于该算子只要求功能正常，没有额外的要求。算子的计算流程、标杆、典型场景客户均无法提供，这带来一定的实现难度，需要结合模型结构、模块功能和官方实现展开分析。

# 1、DS3.2 indexer源码分析

既然客户无法给出有效参考信息，只能从官方实现入手。

官方实现链接为：

[DeepSeek-V3.2-Exp/inference/model.py at main · deepseek-ai/DeepSeek-V3.2-Exp (github.com)](#L435)

indexer的forward实现：

```python
def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
    bsz, seqlen, _ = x.size()  # [b, s, 7168] 
    end_pos = start_pos + seqlen
    # qr矩阵乘计算得到q， Input Tensor Shape:[b, s, 1536] * [1536, 64*128] -> Output Tensor Shape:[b, s, 64*128] 
    q = self.wq_b(qr)
    # 将Hidden维度拆成Head Num, Head Dim两个维度，Input Tensor Shape:[b, s, 64*128] -> Output Tensor Shape:[b, s, 64, 128] 
    q = rearrange(q, 'b s (h d) -> b s h d', d=self.head_dim)
    # Head Dim维度拆成两个64, Input Tensor Shape:[b, s, 64, 128] -> Output Tensor Shape:[b, s, 64, 64], [b, s, 64, 64]
    q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # 旋转位置编码， Input Tensor Shape:[b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 64] 
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    # 合并q， Input Tensor Shape:[b, s, 64, 64], [b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 128]
    q = torch.cat([q_pe, q_nope], dim=-1)
    # 矩阵乘计算得到k,k只有1个头，Input Tensor Shape:[b, s, 7168] * [7168, 128] -> Output Tensor Shape:[b, s, 128]
    k = self.wk(x)
    # LayerNorm归一化，Input Tensor Shape:[b, s, 128] -> Output Tensor Shape:[b, s, 128]
    k = self.k_norm(k)
    # Head Dim维度拆成两个64，Input Tensor Shape:[b, s, 128] -> Output Tensor Shape:[b, s, 64], [b, s, 64]
    k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
    # 旋转位置编码， Input Tensor Shape:[b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 64] 
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
    # 合并k， Input Tensor Shape:[b, s, 64, 64], [b, s, 64, 64] -> Output Tensor Shape:[b, s, 64, 128]
    k = torch.cat([k_pe, k_nope], dim=-1)
    # Hadamard变换 -> [b, s, 64, 128]
    q = rotate_activation(q)
    # Hadamard变换 -> [b, s, 128]
    k = rotate_activation(k)
    # perGroup量化，q_fp8 -> [b, s, 64, 128]，q_scale -> [b, s, 64]
    q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
    # perGroup量化，k_fp8 -> [b, s, 128]，k_scale -> [b, s]
    k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
    self.k_cache[:bsz, start_pos:end_pos] = k_fp8
    self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale
    # 对x矩阵乘，[b, s, 7168] -> [b, s, 64] / sqrt(128)
    weights = self.weights_proj(x) * self.n_heads ** -0.5
    # Weight与q_scale相乘，省一次点乘，并扩维，Input Tensor Shape:[b, s, 64] -> Output Tensor Shape:[b, s, 64, 1]
    weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    # Input Tensor Shape : [b, s, 64, 128]， [b, s, 64，1], [b, s, 128], [b, s]
    index_score = fp8_index(q_fp8.contiguous(), weights, self.k_cache[:bsz, :end_pos].contiguous(), self.k_scale_cache[:bsz, :end_pos].contiguous())
    # Output Tensor Shape: [b, s, s] Token之间的相关性的分数
    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
    # [b, s, s] -> [b, s, 2048] 选出分数最高的2048个索引
    topk_indices_ = topk_indices.clone()
    dist.broadcast(topk_indices_, src=0)
    assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    return topk_indices

```

其整体的计算流程大概可以分为

* Query和Key的构建和预处理
* FP8量化
* **相似度计算**，对当前查询 token 与所有历史 token 计算索引分数
* **掩码计算和Top-K筛选**，根据分数为每个查询选出前 k 个最相关的 token
  通过以上操作，后续的MLA模块只需要对topk个token进行稀疏注意力计算，将 `O(n²)` 的全局注意力计算转化为`O(nk)` 的高效稀疏计算
  其中旋转位置编码、归一化、旋转激活（hadamard变换）、FP8量化和TopK索引都是一些通用的计算操作，可以在torch下直接执行计算，先不作过多关注。
  最特别的是fp8_index这个注意力分数计算操作。这是个基于tilelang语法实现的自定义算子，当前昇腾平台对于tielang的支持还尚不完善，无法直接调用这个算子。因此想要在昇腾上将其快速转为高效实现，需要分析其功能然后编写对应的triton算子。
  其官方实现为：
  
  ```python
  @tilelang.jit(out_idx=[4], pass_configs=pass_configs)
   def fp8_index_kernel(h: int, d: int):
       b = T.symbolic("b")
       m = T.symbolic("m")
       n = T.symbolic("n")
  
       blk_n1 = 512
       blk_n2 = 128
  
       @T.prim_func
       def fp8_index_kernel_(
           q: T.Tensor[(b, m, h, d), FP8],
           q_s: T.Tensor[(b, m, h), FP32],
           k: T.Tensor[(b, n, d), FP8],
           k_s: T.Tensor[(b, n), FP32],
           o: T.Tensor[(b, m, n), FP32],
       ) -> None:
           # 分核逻辑，其中b,m是Query的前两维，对应batch_size和seqlen
           # n是Key的seqlen，一般会很大，所以用blk_n1的大小对其分块处理
           with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
               # 将一个完整的量化多头query载入共享内存
               q_smem = T.alloc_shared((h, d), FP8)
               T.copy(q[i_b, i_m, 0, 0], q_smem)
               # 加载Query对应的权重与量化系数
               q_s_frag = T.alloc_fragment(h, FP32)
               T.copy(q_s[i_b, i_m, 0], q_s_frag)
  
               for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                   # 将对应量化key载入共享内存
                   k_smem = T.alloc_shared((blk_n2, d), FP8)
                   T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)
                   # 加载对应key量化系数
                   k_s_frag = T.alloc_fragment(blk_n2, FP32)
                   T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)
  
                   logits = T.alloc_fragment((blk_n2, h), FP32)
                   # K × Q^T直接得到[blk_n2, h]对应shape
                   T.gemm(
                       k_smem,
                       q_smem,
                       logits,
                       transpose_A=False,
                       transpose_B=True,
                       clear_accum=True,
                   )
  
                   for i_h, i3_n in T.Parallel(h, blk_n2):
                       logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]
  
                   logits_sum = T.alloc_fragment(blk_n2, FP32)
                   # 对head维度求和
                   T.reduce_sum(logits, logits_sum, dim=1)
  
                   for i3_n in T.Parallel(blk_n2):
                       logits_sum[i3_n] *= k_s_frag[i3_n]
  
                   T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])
  
       return fp8_index_kernel_
  
  
   def fp8_index(
       q: torch.Tensor,
       q_s: torch.Tensor,
       k: torch.Tensor,
       k_s: torch.Tensor,
   ) -> torch.Tensor:
       """
       Perform index score using FP8 precision.
  
       Args:
           q (torch.Tensor): The Q tensor, must be contiguous.
           q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
           k (torch.Tensor): The K tensor, must be contiguous.
           k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.
  
           fp8 q @ fp8 k -> fp32 logits
           relu(fp32 logits) * q_s (weights) -> fp32 logits
           fp32 logits -> fp32 logits_sum
           fp32 logits_sum * k_s (e8m0) -> fp32 index_score
       """
       return fp8_index_kernel(q.shape[2], q.shape[3])(q, q_s, k, k_s)
  ```

算子内部的逻辑并不复杂，主要逻辑为对QK进行相关度计算后在head维度上进行求和。但值得注意的是，对于昇腾硬件来说，A2A3并不支持FP8的数据格式（软实现也只能数据补位到fp16，无法实现计算加速），只能以fp16实现相同的功能，也就是量化相关的运算都需要摒弃。为此要将tilelang算子中的FP8更改为BF16（默认数据格式），把k_s相关逻辑删除。
为了确认算子修改的正确性，在GPU上构建torch标杆用于互相验证：

```python
def torch_forward(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    b, m, h, d = q.shape
    n = k_cache.shape[1]
    index_score = torch.zeros((b, m, n), dtype=torch.float32, device=q.device)
    for batch_id in range(b):
        for m_id in range(m):
            q_slice = q[batch_id, m_id].to(torch.float32)
            k_slice = k_cache[batch_id].to(torch.float32)
            relu_out = torch.maximum(
                torch.matmul(
                    q_slice.to(torch.float32), k_slice.to(torch.float32).transpose(0, 1)
                ),
                torch.tensor(0),
            )
            weight_out = relu_out * weights[batch_id, m_id].unsqueeze(-1)
            reduce_out = torch.sum(weight_out, dim=0)
            index_score[batch_id, m_id] = reduce_out
    # print(index_score)
    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(2048, dim=-1)[1]
    topk_indices_ = topk_indices.clone()

    assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    return topk_indices
```

其中torch标杆中的运算提升到FP32，保证较高的精度。

# 构建测试用例

在编写完torch标杆和tilelang标杆后，期望对功能正确性进行测试。测试场景为了贴合实际场景，需要结合DeepSeek-V3.2-exp模型结构分析。
首先从官方模型config来看，V3.2模型与V3最大的区别就是MLA的稀疏化配置，其中新增了三个有关参数：

```python
# index
index_n_heads: int = 64
index_head_dim: int = 128
index_topk: int = 2048
```

对应到用例中`H(head num)`默认为64，`K(head dim)`默认为128，`topk索引数`默认为2048。剩余参数为`B(batch size)、M(qseqlen)、和N(kseqlen)`，模型配置中也给出了其范围

```python
max_batch_size: int = 8
max_seq_len: int = 4096 * 4
```

其中`M`比较特殊，分为prefill和decode场景，prefill阶段计算的是自注意力，故`M=N`，decode阶段输入为上轮自回归生成的token，通常`M=1`。由此设计测试场景组合为4K和16K长序列场景：

```python
shape_list = [
    # (batch, q_seq, k_seq, q_headnum)
    (8, 1, 4096, 64, 128),  # decode
    (8, 1, 4096 * 4, 64, 128),
    (8, 4096, 4096, 64, 128),  # prefill
    (8, 4096 * 4, 4096 * 4, 64, 128),
]
```

通过以上分析和设计，已经构建了参考的标杆和测试用例场景，接下来在此基础上构建triton算子。

# 构建triton算子

## 分核逻辑

triton算子的分核逻辑比较固定，以用满所有的aicore为最基本的要求，CV融合算子的分核数直接固定为AIC的数量。

## tiling逻辑

首先确定算子输入，由于没有fp8量化的过程，triton算子的输入tensor只需要q，q_s(weights)、k和对应的shape参数。
根据之前forward中的shape分析，其中`q->[B,M,64,128]`，`q_s->[B,M,64]`，`k->[B,N,128]`
`B、M、N`为可变量，分块也按这三个维度进行。从矩阵乘看，`B、M`轴属于高纬轴，可以直接合轴作为分块索引，最小计算单位为q的最后两维和k的最后两维`[64,128],[128,N]^T->[64,N]`，经过reducesum后为`[1,N]`。由于`N`最大可以达到16K，因此在长序列下一次性肯定不能载入整块数据，`N`轴上需要用子块`BLOCK_SIZE_N`切分。
由此整体的任务分块设计为`B轴×M轴×N轴上子块数`，每个子块任务处理数据量为`qslice[1,1,64,128]`和`kslice[1,BLOCK_SIZE_N,128]`，输出为`[1,1,BLOCK_SIZE_N]`，预期算子输出`[B,M,N]`。
对应的triton算子实现为：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_N": 512}),
    ],
    key=["H", "N", "K"],
)
@triton.jit
def triton_index_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    q_s_ptr,
    B,
    H: tl.constexpr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        q_ptr (tl.tensor): Pointer to the Q.
        k_ptr (tl.tensor): Pointer to the K.
        o_ptr (tl.tensor): Pointer to the index score.
        q_s_ptr (tl.tensor): Pointer to scaling factors for Q (float), or weights.
        B : Batch size.
        H (tl.constexpr): Number of Q heads.
        M : Q sequence length.
        N : K sequence length.
        K (tl.constexpr): dim length.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.

    Returns:
        None
    """

    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = B * M * NUM_BLOCKS_N

    # Current M-dimension block index
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        batch_idx = block_idx // (M * NUM_BLOCKS_N)
        bm_idx = block_idx // NUM_BLOCKS_N
        n_idx = block_idx % NUM_BLOCKS_N

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        q_ptrs = q_ptr + bm_idx * H * K + offs_h[:, None] * K + offs_k[None, :]
        q_s_ptrs = q_s_ptr + bm_idx * H + offs_h
        q = tl.load(q_ptrs)
        q_s = tl.load(q_s_ptrs)

        k_ptrs = (
            k_ptr
            + batch_idx * N * K
            + n_idx * BLOCK_SIZE_N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )

        mask = n_idx * BLOCK_SIZE_N + offs_n < N
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

        o = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * q_s[:, None], axis=0)

        o_ptrs = o_ptr + bm_idx * N + n_idx * BLOCK_SIZE_N + offs_n
        tl.store(o_ptrs, o, mask=mask)

def triton_index(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor):
    # due to unquant, there is no need for k_s
    assert q.is_contiguous() and k.is_contiguous(), "Input tensors must be contiguous"

    B, M, H, K = q.size()
    N = k.size(1)

    o = torch.zeros((B, M, N), dtype=torch.float32, device=q.device)

    grid = (NUM_CORE // 2,)

    triton_index_kernel[grid](q, k, o, q_s, B, H, M, N, K)
    return o
```

在GPU上测试tilelang、torch、triton三者最终输出的topk索引二进制一致，在昇腾平台上由于矩阵乘的精度误差，部分索引之间存在位置互换，但topk索引集合元素一致可以认为精度达到要求。

# 参考脚本

```python
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

sys.path.append(str(Path(__file__).parent))

try:
    import triton.runtime.driver as driver

    NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())[
        "num_vectorcore"
    ]
except AttributeError:
    NUM_CORE = 48

try:
    import tilelang
    import tilelang.language as T

    tilelang.set_log_level("WARNING")

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: True,
    }

    BF16 = "bfloat16"
    FP32 = "float32"

    @tilelang.jit(out_idx=[3], pass_configs=pass_configs)
    def tilelang_index_kernel(h: int, d: int):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")

        blk_n1 = 512
        blk_n2 = 128

        @T.prim_func
        def tilelang_index_kernel_(
            q: T.Tensor[(b, m, h, d), BF16],
            q_s: T.Tensor[(b, m, h), FP32],
            k: T.Tensor[(b, n, d), BF16],
            # k_s: T.Tensor[(b, n), FP32],
            o: T.Tensor[(b, m, n), FP32],
        ) -> None:
            with T.Kernel(b, m, T.ceildiv(n, blk_n1)) as (i_b, i_m, i1_n):
                # 共享内存分配改为BF16类型
                q_smem = T.alloc_shared((h, d), BF16)
                T.copy(q[i_b, i_m, 0, 0], q_smem)

                q_s_frag = T.alloc_fragment(h, FP32)
                T.copy(q_s[i_b, i_m, 0], q_s_frag)

                for i2_n in T.Pipelined(blk_n1 // blk_n2, num_stages=2):
                    # 共享内存分配改为BF16类型
                    k_smem = T.alloc_shared((blk_n2, d), BF16)
                    T.copy(k[i_b, i1_n * blk_n1 + i2_n * blk_n2, 0], k_smem)

                    # k_s_frag = T.alloc_fragment(blk_n2, FP32)
                    # T.copy(k_s[i_b, i1_n * blk_n1 + i2_n * blk_n2], k_s_frag)

                    logits = T.alloc_fragment((blk_n2, h), FP32)
                    # GEMM操作保持不变，tilelang会自动处理BF16输入
                    T.gemm(
                        k_smem,
                        q_smem,
                        logits,
                        transpose_A=False,
                        transpose_B=True,
                        clear_accum=True,
                    )

                    for i_h, i3_n in T.Parallel(h, blk_n2):
                        logits[i3_n, i_h] = T.max(logits[i3_n, i_h], 0) * q_s_frag[i_h]

                    logits_sum = T.alloc_fragment(blk_n2, FP32)
                    T.reduce_sum(logits, logits_sum, dim=1)

                    # for i3_n in T.Parallel(blk_n2):
                    # logits_sum[i3_n] *= k_s_frag[i3_n]

                    T.copy(logits_sum, o[i_b, i_m, i1_n * blk_n1 + i2_n * blk_n2])

        return tilelang_index_kernel_

    def tilelang_index(
        q: torch.Tensor,
        q_s: torch.Tensor,
        k: torch.Tensor,
        # k_s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform index score using FP8 precision.

        Args:
            q (torch.Tensor): The Q tensor, must be contiguous.
            q_s (torch.Tensor): The scaling factor for Q (float), must be contiguous.
            k (torch.Tensor): The K tensor, must be contiguous.
            k_s (torch.Tensor): The scaling factor for K (e8m0 here), must be contiguous.

            fp8 q @ fp8 k -> fp32 logits
            relu(fp32 logits) * q_s (weights) -> fp32 logits
            fp32 logits -> fp32 logits_sum
            fp32 logits_sum * k_s (e8m0) -> fp32 index_score
        """

        return tilelang_index_kernel(q.shape[2], q.shape[3])(q, q_s, k)

except ImportError:
    None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128}),
        triton.Config({"BLOCK_SIZE_N": 256}),
        triton.Config({"BLOCK_SIZE_N": 512}),
    ],
    key=["H", "N", "K"],
)
@triton.jit
def triton_index_kernel(
    q_ptr,
    k_ptr,
    o_ptr,
    q_s_ptr,
    B,
    H: tl.constexpr,
    M,
    N,
    K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        q_ptr (tl.tensor): Pointer to the Q.
        k_ptr (tl.tensor): Pointer to the K.
        o_ptr (tl.tensor): Pointer to the index score.
        q_s_ptr (tl.tensor): Pointer to scaling factors for Q (float), or weights.
        B : Batch size.
        H (tl.constexpr): Number of Q heads.
        M : Q sequence length.
        N : K sequence length.
        K (tl.constexpr): dim length.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.

    Returns:
        None
    """

    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = B * M * NUM_BLOCKS_N

    # Current M-dimension block index
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        batch_idx = block_idx // (M * NUM_BLOCKS_N)
        bm_idx = block_idx // NUM_BLOCKS_N
        n_idx = block_idx % NUM_BLOCKS_N

        offs_h = tl.arange(0, H)
        offs_k = tl.arange(0, K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        q_ptrs = q_ptr + bm_idx * H * K + offs_h[:, None] * K + offs_k[None, :]
        q_s_ptrs = q_s_ptr + bm_idx * H + offs_h
        q = tl.load(q_ptrs)
        q_s = tl.load(q_s_ptrs)

        k_ptrs = (
            k_ptr
            + batch_idx * N * K
            + n_idx * BLOCK_SIZE_N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )

        mask = n_idx * BLOCK_SIZE_N + offs_n < N
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

        o = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * q_s[:, None], axis=0)

        o_ptrs = o_ptr + bm_idx * N + n_idx * BLOCK_SIZE_N + offs_n
        tl.store(o_ptrs, o, mask=mask)


def triton_index(q: torch.Tensor, q_s: torch.Tensor, k: torch.Tensor):
    # due to unquant, there is no need for k_s
    assert q.is_contiguous() and k.is_contiguous(), "Input tensors must be contiguous"

    B, M, H, K = q.size()
    N = k.size(1)

    o = torch.zeros((B, M, N), dtype=torch.float32, device=q.device)

    grid = (NUM_CORE // 2,)

    triton_index_kernel[grid](q, k, o, q_s, B, H, M, N, K)
    return o


def tilelang_forward(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    print("tilelang==================================>")
    index_score = tilelang_index(q, weights, k_cache)
    # print(index_score)

    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(2048, dim=-1)[1]
    topk_indices_ = topk_indices.clone()

    assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    return topk_indices


def torch_forward(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    print("torch==================================>")
    b, m, h, d = q.shape
    n = k_cache.shape[1]
    index_score = torch.zeros((b, m, n), dtype=torch.float32, device=q.device)
    for batch_id in range(b):
        for m_id in range(m):
            q_slice = q[batch_id, m_id].to(torch.float32)
            k_slice = k_cache[batch_id].to(torch.float32)
            relu_out = torch.maximum(
                torch.matmul(
                    q_slice.to(torch.float32), k_slice.to(torch.float32).transpose(0, 1)
                ),
                torch.tensor(0),
            )
            weight_out = relu_out * weights[batch_id, m_id].unsqueeze(-1)
            reduce_out = torch.sum(weight_out, dim=0)
            index_score[batch_id, m_id] = reduce_out
    # print(index_score)
    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(2048, dim=-1)[1]
    topk_indices_ = topk_indices.clone()

    assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    return topk_indices


def triton_forward(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_cache: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    print("triton==================================>")
    index_score = triton_index(q, weights, k_cache)
    # print(index_score)
    if mask is not None:
        index_score += mask
    topk_indices = index_score.topk(2048, dim=-1)[1]
    topk_indices_ = topk_indices.clone()

    assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
    return topk_indices


def compare_result(pred_out: torch.Tensor, target_out: torch.Tensor, topk=2048):
    """
    比较两个topk索引输出结果，只需topk个索引元素集合相等（不考虑位置顺序）

    Args:
        pred_out: 预测输出（如tilelang_out、triton_out），shape: (B, M, topk)
        target_out: 目标输出（如torch_out），shape: (B, M, topk)
        topk: 需要比较的topk大小，默认与代码中一致为2048

    Returns:
        bool: 所有位置的topk索引集合是否都相等
    """
    pred_out = pred_out.cpu()
    target_out = target_out.cpu()

    # 检查形状是否一致
    if pred_out.shape != target_out.shape:
        print(f"形状不一致！预测形状: {pred_out.shape}, 目标形状: {target_out.shape}")
        return False

    B, M, _ = pred_out.shape
    all_match = True

    for i in range(B):
        for j in range(M):
            # 获取当前位置的topk索引（整数类型）
            pred_indices = pred_out[i, j]
            target_indices = target_out[i, j]

            # 排序后比较（消除位置顺序影响）
            pred_sorted = torch.sort(pred_indices)[0]
            target_sorted = torch.sort(target_indices)[0]

            # 整数直接比较是否完全一致
            match = torch.equal(pred_sorted, target_sorted)

            if not match:
                # 找出不一致的元素
                diff_mask = pred_sorted != target_sorted
                diff_positions = torch.where(diff_mask)[0]

                print(f"\n批次 {i}, 序列位置 {j} 的topk索引集合不一致")
                print(f"不一致的位置索引（前10个）: {diff_positions[:10]}")
                print(f"预测索引（对应位置）: {pred_sorted[diff_positions[:10]]}")
                print(f"目标索引（对应位置）: {target_sorted[diff_positions[:10]]}")
                print(f"预测所有索引: {pred_sorted.tolist()[:20]}...")  # 显示前20个
                print(f"目标所有索引: {target_sorted.tolist()[:20]}...")

                all_match = False

    if all_match:
        print(f"\n 所有 {B} 个批次、{M} 个序列位置的topk={topk}索引集合均一致")
    else:
        print(f"\n 存在批次/序列位置的topk={topk}索引集合不一致")

    return all_match


def test_indexer():
    np.random.seed(0)
    shape_list = [
        # (batch, q_seq, k_seq, q_headnum)
        (8, 1, 4096, 64, 128),  # decoder
        (8, 1, 4096 * 4, 64, 128),
        (8, 4096, 4096, 64, 128),  # prefill
        (8, 4096 * 4, 4096 * 4, 64, 128),
    ]
    for shape in shape_list:
        batch, q_seq, k_seq, q_headnum, dim = shape
        device = "npu"
        np.random.seed(0)
        query = torch.tensor(
            np.random.uniform(-10, 10, (batch, q_seq, q_headnum, dim)),
            device=device,
            dtype=torch.bfloat16,
        )
        weights = torch.tensor(
            np.random.uniform(-1, 1, (batch, q_seq, q_headnum)),
            device=device,
            dtype=torch.float32,
        )
        k_cache = torch.tensor(
            np.random.uniform(-10, 10, (batch, k_seq, dim)),
            device=device,
            dtype=torch.bfloat16,
        )
        mask = (
            torch.full((q_seq, q_seq), float("-inf"), device=device).triu_(1)
            if q_seq > 1
            else None
        )

        # if device == "cuda":
        #     tilelang_out = tilelang_forward(query, weights, k_cache, mask)
        #     print(tilelang_out)

        # torch_out = torch_forward(query, weights, k_cache, mask)
        # print(torch_out)

        triton_out = triton_forward(query, weights, k_cache, mask)
        print(triton_out)

        # if device == "cuda":
        #     print("tilelang_out:", compare_result(tilelang_out, torch_out))
        # print("triton_out:", compare_result(triton_out, torch_out))


if __name__ == "__main__":
    test_indexer()

```


