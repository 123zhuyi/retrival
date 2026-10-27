@triton.jit
def _single_rope_matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for head_idx in range(HEADS):
        for batch_idx in range(BATCH):
            for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
                task_m_idx, task_n_idx = grouped_launch_diagonal(
                    block_idx, NUM_BLOCKS_M, NUM_BLOCKS_N, BLOCK_TRESHHOLD
                )
                m_start = task_m_idx * BLOCK_M
                n_start = task_n_idx * BLOCK_N
                mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_start in range(0, K, BLOCK_K):
                    mat_a_offset = (
                        (batch_idx * HEADS * M * K)
                        + (head_idx * M * K)
                        + ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None]
                        + (k_start + tl.arange(0, BLOCK_K))[None, :]
                    )
                    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                        (k_start + tl.arange(0, BLOCK_K)) < K
                    )[None, :]
                    mat_a_block = tl.load(
                        mat_a + mat_a_offset, mask=mat_a_mask, other=0.0
                    )
                    dl.compile_hint(mat_a_block, "dot_pad_only_k")
                    mat_b_offset = (
                        (head_idx * K * N)
                        + ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None]
                        + (n_start + tl.arange(0, BLOCK_N))[None, :]
                    )
                    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                        (n_start + tl.arange(0, BLOCK_N)) < N
                    )[None, :]
                    mat_b_block = tl.load(
                        mat_b + mat_b_offset, mask=mat_b_mask, other=0.0
                    )
                    dl.compile_hint(mat_b_block, "dot_pad_only_k")
                    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
                mat_c_offset = (
                    (batch_idx * HEADS * M * N)
                    + (head_idx * M * N)
                    + ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None]
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