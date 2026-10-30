# lightning_indexer - Triton-Ascend skeleton (PSEUDOCODE)
#
# 知识制品，非可编译实现：未经编译、未经 NPU 验证、未 benchmark。
# reproducibility: pseudocode
# 稀疏注意力索引选择器：根据 Q·K^T 和 weights 计算每个 token 的 topK 重要位置。
# 来源：attention/lightning_indexer/ (commit d28d8f9...), README L20-24.

import triton
import triton.language as tl


@triton.jit
def lightning_indexer_kernel(
    q_ptr, k_ptr, weights_ptr, indices_ptr, values_ptr,
    block_table_ptr,
    actual_seq_q_ptr, actual_seq_k_ptr,
    sparse_count: tl.constexpr,
    D: tl.constexpr,  # 128
    BLOCK_M: tl.constexpr,  # 512
    BLOCK_N: tl.constexpr,  # 512
    PAGE_ATTENTION: tl.constexpr,
    CAUSAL: tl.constexpr,
    RETURN_VALUES: tl.constexpr,
):
    pid = tl.program_id(0)
    # 任务 = (batch, n2Head, gS1, s2Block)
    # ...

    # 1. Q·K^T（通过 blockTable 映射，PA 场景）
    # 2. ReLU(Q·K^T) = max(s, 0)
    # 3. weights 缩放
    # 4. topK 选择最重要的 sparse_count 个位置
    # 5. 输出 sparse_indices（INT32）
    # 6. 可选输出 sparse_values

    # 注意：
    # - layout 支持 BSND / TND / PA_BSND
    # - sparse_count 默认 2048，支持最高 8192
    # - sparse_mode=3 为 rightDownCausal mask
    # - N2（KV heads）仅支持 1
    # - 输出 shape [B,S1,N2,sparse_count] 或 [T1,N2,sparse_count]
    pass