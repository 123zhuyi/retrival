@triton.jit
def pad_topk_idx_kernel(
    t_ptr,
    p_ptr,
    cu_seqlens,
    topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_pk,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    # get q start and len after rmpad
    q_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - q_start
    if BLOCK_SIZE_N * pid_n >= q_len:
        return
    # init prts
    t_ptrs = tl.make_block_ptr(
        base=t_ptr + pid_h * stride_th + q_start * stride_tn,
        shape=(q_len, topk),
        strides=(stride_tn, stride_tk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_T),
        order=(1, 0),
    )
    p_ptrs = tl.make_block_ptr(
        base=p_ptr + pid_b * stride_pb + pid_h * stride_ph,
        shape=(q_len, topk),
        strides=(stride_pn, stride_pk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_T),
        order=(1, 0),
    )
    # load and save
    idxs = tl.load(t_ptrs, boundary_check=(0, 1))
    tl.store(p_ptrs, idxs, boundary_check=(0, 1))