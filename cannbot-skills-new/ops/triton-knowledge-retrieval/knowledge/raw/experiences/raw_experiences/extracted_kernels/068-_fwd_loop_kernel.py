@triton.jit
def _fwd_loop_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    workspace_q_decay_ptr,
    workspace_k_trans_decay_ptr,
    workspace_block_decay_ptr,
    workspace_diag_decay_ptr,
    output_ptr,
    slope_rate,
    kv_cache_ptr,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    """
    Kernel for lightning attention prefill with KV cache.
    """
    # get offset
    off_c = tl.program_id(0)

    for off_bh in range(off_c, b * h, NUM_CORES):
        for off_e in range((e + BLOCK_MODEL - 1) // BLOCK_MODEL):
            off_h = off_bh % h
            qk_offset = off_bh * n * d
            v_offset = off_bh * n * e
            o_offset = off_bh * n * e
            e_offset = off_e * BLOCK_MODEL
            kv_offset = off_bh * d * e

            # get block ptr
            Q_block_ptr = q_ptr + qk_offset + tl.arange(0, d)[None, :]
            K_trans_block_ptr = k_ptr + qk_offset + tl.arange(0, d)[:, None]
            V_block_ptr = (
                v_ptr + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
            )
            O_block_ptr = (
                output_ptr + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
            )
            S_block_ptr = slope_rate + off_h

            with dl.async_task(scope=dl.async_task.vector):
                # init decay
                s = tl.load(S_block_ptr).to(tl.float32)
                off_block = tl.arange(0, BLOCK)
                q_decay = tl.exp(-s.to(tl.float32) * (off_block[:, None] + 1))
                k_trans_decay = tl.exp(
                    -s.to(tl.float32) * (BLOCK - 1 - off_block[None, :])
                )
                block_decay = tl.exp(-s.to(tl.float32) * BLOCK)

                tl.store(workspace_q_decay_ptr + off_block[:, None], q_decay)
                tl.store(
                    workspace_k_trans_decay_ptr + off_block[None, :], k_trans_decay
                )
                tl.store(workspace_block_decay_ptr, block_decay)

                index = off_block[:, None] - off_block[None, :]
                s_index = s * index
                s_index = tl.where(index >= 0, -s_index, float("-inf"))
                diag_decay = tl.exp(s_index)
                tl.store(
                    workspace_diag_decay_ptr + off_block[:, None] * BLOCK + off_block,
                    diag_decay,
                )

                dl.set_cross_flag(dl.SyncFlag.V2C, 0)

            with dl.async_task(scope=dl.async_task.cube):
                dl.set_cross_flag(dl.SyncFlag.V2C, 0)

                kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)
                q_decay = tl.load(workspace_q_decay_ptr + off_block[:, None])
                k_trans_decay = tl.load(
                    workspace_k_trans_decay_ptr + off_block[None, :]
                )
                block_decay = tl.load(workspace_block_decay_ptr)
                diag_decay = tl.load(
                    workspace_diag_decay_ptr + off_block[:, None] * BLOCK + off_block
                )

                # loop compute
                for i in range(NUM_BLOCK):
                    if n < BLOCK * (i + 1):
                        block_decay = tl.exp(-s.to(tl.float32) * (n - BLOCK * i))
                        # (BLOCK - 1 - off_block[None, :] + n - BLOCK)
                        k_trans_decay = tl.exp(
                            -s.to(tl.float32) * (n - 1 - off_block[None, :])
                        )
                    # load
                    q = tl.load(
                        Q_block_ptr + off_block[:, None] * d,
                        mask=off_block[:, None] < n,
                        other=0.0,
                    ).to(tl.float32)
                    k_trans = tl.load(
                        K_trans_block_ptr + off_block[None, :] * d,
                        mask=off_block[None, :] < n,
                        other=0.0,
                    ).to(tl.float32)
                    v = tl.load(
                        V_block_ptr + off_block[:, None] * e,
                        mask=off_block[:, None] < n,
                        other=0.0,
                    ).to(tl.float32)

                    # compute
                    qk = tl.dot(q, k_trans) * diag_decay
                    o_intra = tl.dot(qk, v)
                    o_inter = tl.dot(q * q_decay, kv)
                    o = o_intra + o_inter

                    # save and update
                    tl.store(
                        O_block_ptr + off_block[:, None] * e,
                        o.to(O_block_ptr.dtype.element_ty),
                        mask=off_block[:, None] < n,
                    )
                    kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
                    off_block += BLOCK

            KV_block_ptr = (
                kv_cache_ptr
                + kv_offset
                + e_offset
                + tl.arange(0, d)[:, None] * e
                + tl.arange(0, BLOCK_MODEL)[None, :]
            )
            tl.store(
                KV_block_ptr,
                kv.to(KV_block_ptr.dtype.element_ty),
            )