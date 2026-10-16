@triton.jit
def fused_matmul_qn_and_rotary_pos_emb_qr_kernel(
    # matmul
    Q,
    MAT_B,
    MAT_C,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_TRESHHOLD: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    # rotary pos emb
    COS,
    SIN,
    O,
    total_seq_len,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.vector):
        rotary_pos_emb_qr_kernel(
            X=Q,
            COS=COS,
            SIN=SIN,
            O=O,
            total_seq_len=total_seq_len,
            NOPE_DIM=NOPE_DIM,
            ROPE_DIM=ROPE_DIM,
            BLOCK=BLOCK,
            NUM_HEADS=NUM_HEADS,
            NUM_CORES=NUM_CORES,
        )
    with dl.async_task(scope=dl.async_task.cube):
        matmul_qn_kernel(
            mat_a=Q,
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
            Q_HEAD_DIM=NOPE_DIM + ROPE_DIM,
            NUM_HEADS=NUM_HEADS,
        )