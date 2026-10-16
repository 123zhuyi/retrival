@triton.jit
def paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    o_ptr,
    aux_mask_ptr,
    cu_seqlens_q_ptr,
    q_chunk_indices_ptr,
    num_q_chunks,
    seqlens_kv_ptr,
    block_tables_ptr,
    batch_size,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_k_block,
    stride_k_head,
    stride_k_blksz,
    stride_k_dim,
    stride_v_block,
    stride_v_head,
    stride_v_blksz,
    stride_v_dim,
    stride_ot,
    stride_oh,
    stride_od,
    stride_bt_batch,
    stride_bt_block,
    stride_mask_m,
    stride_mask_n,
    sm_scale,
    AUX_MASK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_INTERLEAVE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    tl.static_assert(CHUNK_SIZE == BLOCK_SIZE_M, "Currently only support CHUNK_SIZE == BLOCK_SIZE_M")
    pid = tl.program_id(0)
    n_progs = tl.num_programs(0)

    num_tasks = num_q_chunks * NUM_Q_HEADS

    for task_id in range(pid, num_tasks, n_progs):
        chunk_id = task_id // NUM_Q_HEADS
        q_head_id = task_id % NUM_Q_HEADS
        if GQA_INTERLEAVE:
            kv_head_id = q_head_id % NUM_KV_HEADS
        else:
            kv_head_id = q_head_id // (NUM_Q_HEADS // NUM_KV_HEADS)

        b_id = tl.load(q_chunk_indices_ptr + chunk_id * 2)
        q_block_id = tl.load(q_chunk_indices_ptr + chunk_id * 2 + 1)

        q_start_loc = tl.load(cu_seqlens_q_ptr + b_id)
        q_end_loc = tl.load(cu_seqlens_q_ptr + b_id + 1)
        q_seq_len = q_end_loc - q_start_loc

        if seqlens_kv_ptr is None:
            kv_seq_len = q_seq_len
        else:
            kv_seq_len = tl.load(seqlens_kv_ptr + b_id)
        kv_cache_len = kv_seq_len - q_seq_len

        q_block_start_in_seq = q_block_id * BLOCK_SIZE_M
        q_block_end_in_seq = min(q_block_start_in_seq + BLOCK_SIZE_M, q_seq_len)
        q_block_len = q_block_end_in_seq - q_block_start_in_seq

        Q_block_ptr = tl.make_block_ptr(
            base=q_ptr + (q_start_loc + q_block_start_in_seq) * stride_qt + q_head_id * stride_qh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_qt, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=o_ptr + (q_start_loc + q_block_start_in_seq) * stride_ot + q_head_id * stride_oh,
            shape=(q_block_len, HEAD_DIM),
            strides=(stride_ot, stride_od),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_D),
            order=(1, 0),
        )

        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float("inf")
        l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

        num_logical_blocks = tl.cdiv(kv_cache_len + q_block_end_in_seq, PAGE_SIZE)

        tl.static_assert(PAGE_SIZE == BLOCK_SIZE_N, "Currently only BLOCK_SIZE_N==PAGE_SIZE supported")

        for logical_block_idx in range(0, num_logical_blocks):
            physical_block_id = tl.load(block_tables_ptr + b_id * stride_bt_batch + logical_block_idx * stride_bt_block)

            kv_block_start_in_seq = logical_block_idx * PAGE_SIZE
            kv_block_end_in_seq = min(kv_block_start_in_seq + PAGE_SIZE, kv_seq_len)
            kv_block_len = kv_block_end_in_seq - kv_block_start_in_seq
            K_T_block_ptr = tl.make_block_ptr(
                base=k_cache_ptr + physical_block_id * stride_k_block + kv_head_id * stride_k_head,
                shape=(HEAD_DIM, kv_block_len),
                strides=(stride_k_dim, stride_k_blksz),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=v_cache_ptr + physical_block_id * stride_v_block + kv_head_id * stride_v_head,
                shape=(kv_block_len, HEAD_DIM),
                strides=(stride_v_blksz, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
                order=(1, 0),
            )

            acc, l_i, m_i = _sdpa_infer_single_block(
                acc,
                l_i,
                m_i,
                q,
                K_T_block_ptr,
                V_block_ptr,
                aux_mask_ptr,
                stride_mask_m,
                stride_mask_n,
                AUX_MASK_SIZE,
                sm_scale,
                kv_cache_len + q_block_start_in_seq,
                kv_block_start_in_seq,
                kv_cache_len + q_block_end_in_seq,
                kv_block_end_in_seq,
                HEAD_DIM,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_D,
                v_cache_ptr.dtype.element_ty == tl.float8e5,
            )

        m_i += tl.math.log(l_i)
        accumulator = acc / l_i[:, None]

        # NOTE(zhangjihang): for training
        # m_ptrs = M + task_bn_idx * sub_kv_len + offs_m
        # tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(o_ptr.type.element_ty), boundary_check=(0, 1))