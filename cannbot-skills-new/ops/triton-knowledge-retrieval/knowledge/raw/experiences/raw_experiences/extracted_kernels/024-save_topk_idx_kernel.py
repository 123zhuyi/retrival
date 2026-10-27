@triton.jit
def save_topk_idx_kernel(
    p_ptr,
    t_ptr,
    cu_seqblocks,
    cu_topk_q_count,
    n_len,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_th,
    stride_tn,
    stride_ch,
    stride_cn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    # get q start and len after rmpad
    q_block_start = tl.load(cu_seqblocks + pid_b)
    q_block_end = tl.load(cu_seqblocks + pid_b + 1)
    c_start = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_start * stride_cn)
    c_end = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_end * stride_cn)
    c_len = c_end - c_start
    if c_len <= 0:
        return
    if pid_n * BLOCK_SIZE_N >= c_len:
        return
    # init ptrs
    p_ptrs = tl.make_block_ptr(
        base=p_ptr
        + pid_b * stride_pb
        + pid_h * stride_ph
        + (n_len - c_len) * stride_pn,
        shape=(c_len,),
        strides=(stride_pn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    t_ptrs = tl.make_block_ptr(
        base=t_ptr + pid_h * stride_th + c_start * stride_tn,
        shape=(c_len,),
        strides=(stride_tn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    # load and save
    idxs = tl.load(p_ptrs, boundary_check=(0,))
    tl.store(t_ptrs, idxs, boundary_check=(0,))