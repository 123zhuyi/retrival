@triton.autotune(configs=k_grouped_mm_autotune_config(), key=["M", "N"])
@triton.jit
def _k_grouped_matmul_kernel(
    A,
    B,
    C,
    group_size_ptr,
    num_groups: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
):
    total_cores = tl.num_programs(axis=0)
    core_idx = tl.program_id(axis=0)
    last_count = 0
    group_start = 0
    group_end = 0
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)
    blocks_per_group = num_block_m * num_block_n
    # group_size_k = tl.load(group_size_ptr + tl.arange(0, num_groups)).to(tl.int32)
    for group_idx in range(num_groups):
        # k = tl.extract_slice(group_size_k, [group_idx], [1], [1])
        tokens = tl.load(group_size_ptr + group_idx).to(tl.int32)
        group_end = group_start + tokens
        cur_count = last_count + blocks_per_group
        cur_block = core_idx if core_idx >= last_count else (core_idx + total_cores)
        for cur_block in range(cur_block, cur_count, total_cores):
            task_m_idx, task_n_idx = grouped_launch_diagonal(
                cur_block - last_count, num_block_m, num_block_n, BLOCK_TRESHHOLD
            )
            # matmul begin
            offs_am = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_bn = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = group_start + tl.arange(0, BLOCK_K)
            a_ptrs_base = A + offs_k[:, None] * M + offs_am[None, :]
            b_ptrs_base = B + offs_k[:, None] * N + offs_bn[None, :]
            msk_m = offs_am < M
            msk_n = offs_bn < N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kk in tl.range(0, tl.cdiv(tokens, BLOCK_K)):
                a_ptrs = a_ptrs_base + kk * BLOCK_K * M
                b_ptrs = b_ptrs_base + kk * BLOCK_K * N
                a = tl.load(a_ptrs, mask=(offs_k[:, None] < group_end - kk * BLOCK_K) and msk_m[None, :], other=0.0)
                aa = tl.trans(a)
                tl.compile_hint(aa, "dot_pad_only_k")
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < group_end - kk * BLOCK_K) and msk_n[None, :], other=0.0)
                tl.compile_hint(b, "dot_pad_only_k")
                accumulator = tl.dot(aa, b, acc=accumulator)

            c = accumulator.to(C.dtype.element_ty)
            offs_cm = group_idx * M + task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_cn = task_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
            c_ptrs = C + offs_cm[:, None] * N + offs_cn[None, :]
            c_mask = (offs_cm[:, None] < (group_idx + 1) * M) and (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask=c_mask)
            # matmul_end
            # cur_block = cur_block + total_cores
        last_count = cur_count % total_cores
        group_start = group_end