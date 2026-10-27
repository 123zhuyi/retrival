@triton.jit
def fused_sigmoid_single_norm_rope_matmul_kernel(
    # partial_matmul
    MAT_A,
    MAT_B,
    MAT_C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_STRIDE_K: tl.constexpr,
    B_N_START: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # rmsnorm
    QQ,
    NORM_WEIGHT,
    seq_len,
    qq_stride_s: tl.constexpr,
    qq_stride_h: tl.constexpr,
    eps: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    TOTAL_HEAD_DIM: tl.constexpr,
    # sigmoid
    O_SIGMOID,
    BLOCK_SIGMOID: tl.constexpr,
    # rotary_pos_emb
    COS,
    SIN,
    OUTPUT,
    ROPE_HEAD_DIM: tl.constexpr,
):
    # matmul + norm + rotary_pos_emb
    with dl.async_task(scope=dl.async_task.cube):
        partial_matmul_kernel(
            mat_a=MAT_A,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=M,
            N=N,
            K=K,
            B_STRIDE_K=B_STRIDE_K,
            B_N_START=B_N_START,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        sigmoid_kernel(
            QQ=QQ,
            O_SIGMOID=O_SIGMOID,
            n_rows=seq_len * NUM_HEADS,
            in_stride_row=qq_stride_h,
            SIGMOID_DIM_START=TOTAL_HEAD_DIM,
            SIGMOID_DIM=TOTAL_HEAD_DIM,
            BLOCK=BLOCK_SIGMOID,
            NUM_CORES=NUM_CORES,
        )
        fused_single_norm_and_partial_rope_kernel(
            # rmsnorm
            INPUT=QQ,
            NORM_WEIGHT=NORM_WEIGHT,
            seq_len=seq_len,
            eps=eps,
            input_stride_s=qq_stride_s,
            input_stride_h=qq_stride_h,
            NUM_HEADS=NUM_HEADS,
            TOTAL_HEAD_DIM=TOTAL_HEAD_DIM,
            # rotary_pos_emb
            COS=COS,
            SIN=SIN,
            OUTPUT=OUTPUT,
            ROPE_HEAD_DIM=ROPE_HEAD_DIM,
            NUM_CORES=NUM_CORES,
        )