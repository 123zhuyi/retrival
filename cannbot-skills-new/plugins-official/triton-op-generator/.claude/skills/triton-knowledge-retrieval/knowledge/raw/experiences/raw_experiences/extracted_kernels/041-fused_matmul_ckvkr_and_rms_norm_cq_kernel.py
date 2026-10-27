@triton.jit
def fused_matmul_ckvkr_and_rms_norm_cq_kernel(
    # matmul params
    mat_a,
    mat_b,
    mat_c,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # rms norm params
    input,
    weight,
    output,
    n_rows,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_NORM: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.cube):
        matmul_kernel(
            mat_a=mat_a,
            mat_b=mat_b,
            mat_c=mat_c,
            M=M,
            N=N,
            K=K,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        rms_norm_block_kernel(
            input=input,
            weight=weight,
            output=output,
            n_rows=n_rows,
            input_row_stride=input_row_stride,
            eps=eps,
            N_COLS=N_COLS,
            BLOCK=BLOCK_NORM,
            NUM_CORES=NUM_CORES,
        )