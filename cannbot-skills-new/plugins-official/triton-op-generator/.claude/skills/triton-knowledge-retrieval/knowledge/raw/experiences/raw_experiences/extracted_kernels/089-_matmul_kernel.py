@triton.autotune(configs=[triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N':
    16, 'BLOCK_SIZE_K': 8}, num_warps=4, num_stages=4)], key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak,
    stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    A_ptr += offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptr += offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(A_ptr, mask=(offs_m[:, None] < M) & (offs_k[None, :] < 
            K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(B_ptr, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) &
            (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        A_ptr += BLOCK_SIZE_K * stride_ak
        B_ptr += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    C_ptr += offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptr, acc, mask=c_mask)