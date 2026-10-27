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
        B : batch size.
        H (tl.constexpr): Number of Q heads.
        M : Q sequence length.
        N : K sequence length.
        K (tl.constexpr): dim length.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    batch_id = pid_m // M
    offs_h = tl.arange(0, H)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    q_ptrs = q_ptr + pid_m * H * K + offs_h[:, None] * K + offs_k[None, :]
    q_s_ptrs = q_s_ptr + pid_m * H + offs_h
    q = tl.load(q_ptrs)
    q_s = tl.load(q_s_ptrs)

    for i in range(num_blocks):
        k_ptrs = (
            k_ptr
            + batch_id * N * K
            + i * BLOCK_SIZE_N * K
            + offs_n[:, None] * K
            + offs_k[None, :]
        )

        mask = i * BLOCK_SIZE_N + offs_n < N
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)

        o = tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * q_s[:, None], axis=0)

        o_ptrs = o_ptr + pid_m * N + i * BLOCK_SIZE_N + offs_n
        tl.store(o_ptrs, o, mask=mask)