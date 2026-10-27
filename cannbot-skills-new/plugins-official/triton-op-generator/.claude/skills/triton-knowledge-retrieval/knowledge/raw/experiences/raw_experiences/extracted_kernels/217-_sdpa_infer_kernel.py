@triton.jit
def _sdpa_infer_kernel(
    Q,
    K,
    V,
    mask,
    M,
    Out,
    scale,
    stride_qz: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qk: tl.constexpr,
    stride_kz: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kk: tl.constexpr,
    stride_vz: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vk: tl.constexpr,
    stride_oz: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BSZ: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    KV_HEAD_NUM: tl.constexpr,
    SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = SEQ // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (BSZ) × number of attention heads (Q_HEAD_NUM)
    NUM_BLOCKS = NUM_BLOCKS_M * BSZ * Q_HEAD_NUM

    # Current M-dimension block index
    pid = tl.program_id(0)
    core_step = tl.num_programs(0)
    for block_idx in range(pid, NUM_BLOCKS, core_step):
        task_bn_idx = block_idx // NUM_BLOCKS_M
        task_seq_idx = block_idx % NUM_BLOCKS_M

        bsz_offset = task_bn_idx // Q_HEAD_NUM
        q_head_num_offset = task_bn_idx % Q_HEAD_NUM
        kv_head_num_offset = q_head_num_offset // (Q_HEAD_NUM // KV_HEAD_NUM)
        q_bn_offset = bsz_offset.to(tl.int64) * stride_qz + q_head_num_offset.to(tl.int64) * stride_qh
        kv_bn_offset = bsz_offset.to(tl.int64) * stride_kz + kv_head_num_offset.to(tl.int64) * stride_kh
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_seq_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + q_bn_offset,
            shape=(SEQ, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_seq_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Initialize offsets
        offs_m = task_seq_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        acc_ptr, l_i, m_i = _sdpa_infer_inner(
            acc_ptr,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask,
            task_seq_idx,
            scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            SEQ,
            V.dtype.element_ty == tl.float8e5,
        )

        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * SEQ + offs_m
        # tl.store(m_ptrs, m_i)

        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))