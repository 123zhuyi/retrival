@triton.jit
def fused_matmul_qb_and_rmsnorm_kv_kernel(
    MAT_A,
    MAT_B,
    MAT_C,
    total_seq_len,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    # rms_norm ckv
    KV,
    W,
    RMS_NORM_O,
    input_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_NORM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.cube):
        matmul_kernel(
            mat_a=MAT_A,
            mat_b=MAT_B,
            mat_c=MAT_C,
            M=total_seq_len,
            N=N,
            K=K,
            NUM_CORES=NUM_CORES,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            BLOCK_TRESHHOLD=BLOCK_TRESHHOLD,
        )
    with dl.async_task(scope=dl.async_task.vector):
        rms_norm_ckv_kernel(
            X=KV,
            W=W,
            O=RMS_NORM_O,
            n_rows=total_seq_len,
            input_row_stride=input_row_stride,
            output_row_stride=output_row_stride,
            eps=eps,
            KV_LORA_RANK=NOPE_DIM,
            BLOCK=BLOCK_NORM,
            NUM_CORES=NUM_CORES,
        )