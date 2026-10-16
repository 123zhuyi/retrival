@triton.heuristics(
    {
        "HAS_WEIGHT": lambda args: args["dw"] is not None,
        "HAS_BIAS": lambda args: args["db"] is not None,
        "USE_INITIAL_STATE": lambda args: args["dh0"] is not None,
        "USE_FINAL_STATE": lambda args: args["dht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_Blk_D: tl.int32,
    NUM_CHKS: tl.int32,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_tasks = NUM_CHKS * NUM_Blk_D

    for task_id in range(pid, total_tasks, num_programs):
        i_d = task_id % NUM_Blk_D
        i_chk = task_id // NUM_Blk_D

        if IS_VARLEN:
            i_t = i_chk

            idx_chk = i_chk

            i_tg = idx_chk

            ptr = chunk_indices + idx_chk * 2
            i_n = tl.load(ptr).to(tl.int32)
            i_t_offset = tl.load(ptr + 1).to(tl.int32)

            i_t = i_t_offset

            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            T_len = eos - bos
        else:
            NT_per_seq = tl.cdiv(T, BT)

            i_b = i_chk // NT_per_seq
            i_t = i_chk % NT_per_seq

            i_tg = i_chk

            i_n = i_b
            bos = (i_b * T).to(tl.int64)
            eos = (i_b * T + T).to(tl.int64)
            T_len = T

        o_d = i_d * BD + tl.arange(0, BD)
        o_w = tl.arange(0, W)
        m_d = o_d < D
        m_w = o_w >= 0

        if HAS_WEIGHT:
            p_x = tl.make_block_ptr(x + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_x = tl.load(p_x, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(weight, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1), padding_option="zero")

        b_dx = tl.zeros((BT, BD), dtype=tl.float32)
        if HAS_BIAS:
            b_db = tl.zeros((BD,), dtype=tl.float32)

        if not USE_FINAL_STATE:
            b_dw = tl.zeros((W, BD), dtype=tl.float32)

            p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT * W, BD), (1, 0))
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

            if ACTIVATION == "swish" or ACTIVATION == "silu":
                p_y = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT * W, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)

            for i_w in tl.static_range(0, W):
                b_dy_sub = tl.extract_slice(b_dy, [i_w, 0], [BT, BD], [1, 1])

                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    b_y_sub = tl.extract_slice(b_y, [i_w, 0], [BT, BD], [1, 1])
                    b_ys = tl.sigmoid(b_y_sub)
                    b_dy_sub = b_dy_sub * b_ys * (1 + b_y_sub * (1 - b_ys))

                b_wdy = b_dy_sub
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw_sub = tl.sum(b_dy_sub * b_x, 0)  # [BT, BD] * [BT, BD] --> sum(0) = [BD]
                    b_dw = tl.insert_slice(b_dw, b_dw_sub[None, :], [W - i_w - 1, 0], [1, BD], [1, 1])

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_sub, 0)
                b_dx += b_wdy

            p_dw = tl.make_block_ptr(dw + i_tg * W * D, (W, D), (D, 1), (0, i_d * BD), (W, BD), (1, 0))
            tl.store(p_dw, b_dw.to(dw.dtype.element_ty))
        elif i_t * BT >= W:
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))

                b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
                b_wdy = b_dy
                if HAS_WEIGHT:
                    b_wdy = b_wdy * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1])

                    b_dw = tl.sum(b_dy * b_x, 0)
                    tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy, 0)
                b_dx += b_wdy
        else:
            o_t = i_t * BT + tl.arange(0, BT)
            for i_w in tl.static_range(0, W):
                p_dy = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y = tl.make_block_ptr(
                        y + bos * D, (T_len, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0)
                    )
                    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                    b_ys = tl.sigmoid(b_y)
                    b_dy_shift = b_dy_shift * b_ys * (1 + b_y * (1 - b_ys))
                if HAS_WEIGHT:
                    b_dw = tl.sum(b_dy_shift * b_x, 0)

                    if USE_INITIAL_STATE:
                        mask_head_rows = o_t < i_w

                        b_dy_head = tl.load(
                            dy + bos * D + o_t[:, None] * D + o_d,
                            mask=(mask_head_rows[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)
                        if ACTIVATION == "swish" or ACTIVATION == "silu":
                            b_y_head = tl.load(
                                y + bos * D + o_t[:, None] * D + o_d,
                                mask=(mask_head_rows[:, None] & m_d[None, :]),
                                other=0.0,
                            ).to(tl.float32)
                            b_ys_head = tl.sigmoid(b_y_head)
                            b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                        o_c = W - i_w + o_t

                        mask_c = mask_head_rows & (o_c >= 1) & (o_c < W)
                        b_xc = tl.load(
                            initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                            mask=(mask_c[:, None] & m_d[None, :]),
                            other=0.0,
                        ).to(tl.float32)

                        b_dw += tl.sum(b_dy_head * b_xc, 0)
                    tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

                if HAS_BIAS and i_w == 0:
                    b_db += tl.sum(b_dy_shift, 0)
                b_wdy = (
                    b_dy_shift
                    if not HAS_WEIGHT
                    else (b_dy_shift * tl.extract_slice(b_w, [W - i_w - 1, 0], [1, BD], [1, 1]))
                )
                b_dx += b_wdy

            if USE_INITIAL_STATE:
                p_dy0 = tl.make_block_ptr(dy + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_dy0 = tl.load(p_dy0, boundary_check=(0, 1)).to(tl.float32)
                if ACTIVATION == "swish" or ACTIVATION == "silu":
                    p_y0 = tl.make_block_ptr(y + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                    b_y0 = tl.load(p_y0, boundary_check=(0, 1)).to(tl.float32)
                    b_ys0 = tl.sigmoid(b_y0)
                    b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))

                for i_w in tl.static_range(1, W):
                    m_rows = o_t < i_w
                    if HAS_WEIGHT:
                        w_idx_rows = i_w - 1 - o_t

                        w_mask = o_w[None, :] == w_idx_rows[:, None]
                        w_pick = tl.sum(tl.trans(b_w)[None, :, :] * w_mask[:, None, :], 2)
                    else:
                        w_pick = 1.0
                    contrib = (b_dy0 * w_pick).to(tl.float32)
                    contrib = tl.where(m_rows[:, None] & m_d[None, :], contrib, 0.0)

                    b_dh0_s = tl.sum(contrib, 0)

                    tl.store(
                        dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                        b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding="rtne"),
                        mask=m_d,
                    )

        if HAS_BIAS:
            b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding="rtne")
            tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

        if USE_FINAL_STATE:
            if i_t * BT + BT >= T_len - W:
                start_tok = max(0, T_len - (W - 1))
                offset = i_t * BT + tl.arange(0, BT)
                tok_idx = offset - start_tok
                mask = (offset >= start_tok) & (offset < T_len)
                w_idx = 1 + tok_idx
                dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
                b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.0).to(tl.float32)
                b_dx += b_dht

        p_dx = tl.make_block_ptr(dx + bos * D, (T_len, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))