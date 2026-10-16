@triton.jit
def fused_matmul_qn_and_rotary_pos_emb_qk_kernel(
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
    NUM_HEADS_Q: tl.constexpr,
    # rotary pos emb
    Q_INPUT,
    K_INPUT,
    COS,
    SIN,
    Q_EMBED,
    K_EMBED,
    total_seq_len,
    NOPE_DIM_Q: tl.constexpr,
    NOPE_DIM_K: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    NUM_HEADS_K: tl.constexpr,
):
    with dl.async_task(scope=dl.async_task.vector):
        # rope_qk
        partial_rope_qk_kernel(
            Q=Q_INPUT,
            K=K_INPUT,
            COS=COS,
            SIN=SIN,
            Q_EMBED=Q_EMBED,
            K_EMBED=K_EMBED,
            total_seq_len=total_seq_len,
            NOPE_DIM_Q=NOPE_DIM_Q,
            NOPE_DIM_K=NOPE_DIM_K,
            ROPE_DIM=ROPE_DIM,
            BLOCK=BLOCK_ROPE,
            NUM_Q_HEADS=NUM_HEADS_Q,
            NUM_K_HEADS=NUM_HEADS_K,
            NUM_CORES=NUM_CORES,
        )
    with dl.async_task(scope=dl.async_task.cube):
        # batch matmul
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
            Q_HEAD_DIM=NOPE_DIM_Q + ROPE_DIM,
            NUM_HEADS=NUM_HEADS_Q,
        )