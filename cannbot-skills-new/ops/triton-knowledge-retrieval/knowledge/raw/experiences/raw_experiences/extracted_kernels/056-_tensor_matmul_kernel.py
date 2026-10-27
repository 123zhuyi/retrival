@triton.autotune(configs=[triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16,
    'BLOCK_K': 16}, num_warps=4, num_stages=3)], key=['M',
    'N', 'K'])
@triton.jit
def _tensor_matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am,
    stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.
    constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=k_mask[None, :] & (offs_m[:, None] < M),
            other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None,
        :] < N))