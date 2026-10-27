@triton.jit
def matmul_qn_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
):
    # q = matmul_qc_qr_out.view(b * seq_len, num_heads, q_head_dim)
    # q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # q_nope = q_nope.view(b * seq_len, num_heads, qk_nope_head_dim)
    # matmul_qn_out = torch.bmm(q_nope.transpose(0, 1), weight_uk)
    # 左矩阵q的nope部分，右矩阵weight_uk: [num_heads, qk_nope_head_dim, kv_lora_rank]，实现bmm逻辑
    # mat_c: [b * seq_len, num_heads, kv_lora_rank]
    # K == qk_nope_head_dim

    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for head_idx in range(NUM_HEADS):
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
            )
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            # begin k
            mat_a_offset = (
                ((m_start + tl.arange(0, BLOCK_M)) * (NUM_HEADS * Q_HEAD_DIM))[:, None]
                + (head_idx * Q_HEAD_DIM)
                + tl.arange(0, BLOCK_K)[None, :]
            )
            mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (tl.arange(0, BLOCK_K)) < K
            )[None, :]
            mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
            dl.compile_hint(mat_a_block, "dot_pad_only_k")
            mat_b_offset = (
                (head_idx * K * N)
                + (tl.arange(0, BLOCK_K) * N)[:, None]
                + (n_start + tl.arange(0, BLOCK_N))[None, :]
            )
            mat_b_mask = (tl.arange(0, BLOCK_K) < K)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
            dl.compile_hint(mat_b_block, "dot_pad_only_k")
            mat_c_block = tl.dot(mat_a_block, mat_b_block)
            # end k
            mat_c_offset = (
                ((m_start + tl.arange(0, BLOCK_M)) * NUM_HEADS * N)[:, None]
                + (head_idx * N)
                + (n_start + tl.arange(0, BLOCK_N))[None, :]
            )
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(
                mat_c + mat_c_offset,
                mat_c_block.to(mat_c.dtype.element_ty),
                mask=mat_c_mask,
            )