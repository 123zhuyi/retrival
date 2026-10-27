@triton.jit()
def causal_conv1d_update_kernel_bdt_fwd(
    x_ptr,  # [B, D, T]
    conv_state_ptr,  # [B, D, ST]
    conv_state_update_ptr,
    weight_ptr,  # [D, W]
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,  # [B, D, OT]
    batch: tl.constexpr,
    dim: tl.constexpr,
    state_len: tl.constexpr,  # ST
    seq_len: tl.constexpr,  # T
    width: tl.constexpr,  # W
    out_len: tl.constexpr,  # OT
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    HAS_BIAS: tl.constexpr,  # TODO
    SILU_ACTIVATION: tl.constexpr,
    T_CHK_SIZE: tl.constexpr,
    D_CHK_SIZE: tl.constexpr,
    NUM_T_CHK: tl.constexpr,
    NUM_D_CHK: tl.constexpr,
    ST_STORE_HEAD_TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    pnum = tl.num_programs(0)

    total_task = batch * NUM_D_CHK * NUM_T_CHK

    for task_id in tl.range(pid, total_task, pnum):
        di = task_id % NUM_D_CHK
        bti = task_id // NUM_D_CHK
        bi = bti // NUM_T_CHK
        ti = bti % NUM_T_CHK

        w = tl.load(
            tl.make_block_ptr(
                weight_ptr,
                shape=(dim, width),
                strides=(width, 1),
                offsets=(di * D_CHK_SIZE, 0),
                block_shape=(D_CHK_SIZE, width),
                order=(1, 0),
            ),
            boundary_check=(0, 1),
            padding_option="zero",
        )

        if ti == 0:
            st_b = tl.load(
                tl.make_block_ptr(
                    conv_state_ptr + bi * state_len * dim,
                    shape=(dim, state_len),
                    strides=(state_len, 1),
                    offsets=(di * D_CHK_SIZE, state_len - (width - 1)),
                    block_shape=(D_CHK_SIZE, (width - 1) + T_CHK_SIZE),
                    order=(1, 0),
                ),
                boundary_check=(0, 1),
                padding_option="zero",
            )
            offset0_x = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1_x = ti * T_CHK_SIZE + tl.arange(0, T_CHK_SIZE)
            mask_x = (offset0_x < dim)[:, None] & ((offset1_x >= 0) & (offset1_x < seq_len))[None, :]
            block_off_x = bi * dim * seq_len + offset0_x[:, None] * seq_len + offset1_x[None, :]
            x_b_tmp = tl.load(x_ptr + block_off_x, mask=mask_x, other=0)
            x_b = tl.insert_slice(st_b, x_b_tmp, (0, width - 1), (D_CHK_SIZE, T_CHK_SIZE), (1, 1))
        else:
            offset0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)
            offset1 = ti * T_CHK_SIZE - (width - 1) + tl.arange(0, T_CHK_SIZE + width - 1)
            mask = (offset0 < dim)[:, None] & ((offset1 >= 0) & (offset1 < seq_len))[None, :]
            block_off = bi * dim * seq_len + offset0[:, None] * seq_len + offset1[None, :]
            x_b = tl.load(x_ptr + block_off, mask=mask, other=0)

        out_block = tl.zeros((T_CHK_SIZE, D_CHK_SIZE), dtype=x_ptr.dtype.element_ty)
        x_b = tl.trans(x_b, (1, 0))
        w = tl.trans(w, (1, 0))

        new_state_start_off = seq_len - state_len
        t_start_off = ti * T_CHK_SIZE - (width - 1)
        t_end_off = (ti + 1) * T_CHK_SIZE
        if t_end_off >= new_state_start_off:
            t_off = t_start_off - new_state_start_off
            if t_off < -(width - 1):
                # NOTE: In order to avoid use tl.maximum for negative offset,
                #       we pre-compute a fix head tile size (ST_STORE_HEAD_TILE_SIZE)
                #       to store the scene of negative address
                x_new_h = tl.extract_slice(x_b, (-t_off, 0), (ST_STORE_HEAD_TILE_SIZE, D_CHK_SIZE), (1, 1))
                x_new_h = tl.trans(x_new_h, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1_h = tl.arange(0, ST_STORE_HEAD_TILE_SIZE)[None, :]
                nst_mask_h = (nst_off_y0 < dim) & (nst_off_y1_h >= 0) & (nst_off_y1_h < state_len)
                block_ptr_h = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1_h
                tl.store(conv_state_update_ptr + block_ptr_h, x_new_h, mask=nst_mask_h)
            else:
                x_new_s = tl.extract_slice(x_b, (width - 1, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
                x_new_s = tl.trans(x_new_s, (1, 0))
                nst_off_y0 = di * D_CHK_SIZE + tl.arange(0, D_CHK_SIZE)[:, None]
                nst_off_y1 = width - 1 + t_off + tl.arange(0, T_CHK_SIZE)[None, :]
                nst_mask = (nst_off_y0 < dim) & (nst_off_y1 >= 0) & (nst_off_y1 < state_len)
                block_ptr = bi * dim * state_len + nst_off_y0 * state_len + nst_off_y1
                tl.store(conv_state_update_ptr + block_ptr, x_new_s, mask=nst_mask)

        for owi in tl.range(0, width):
            new_x = tl.extract_slice(x_b, (owi, 0), (T_CHK_SIZE, D_CHK_SIZE), (1, 1))
            w_chl_wi = tl.extract_slice(w, (owi, 0), (1, D_CHK_SIZE), (1, 1))
            x_mul_chl_wi = new_x * w_chl_wi
            out_block += x_mul_chl_wi
        out_block = tl.trans(out_block, (1, 0))

        if SILU_ACTIVATION:
            out_block = out_block * tl.sigmoid(out_block)
        tl.store(
            tl.make_block_ptr(
                out_ptr,
                shape=(batch, dim, out_len),
                strides=(dim * out_len, out_len, 1),
                offsets=(bi, di * D_CHK_SIZE, ti * T_CHK_SIZE),
                block_shape=(1, D_CHK_SIZE, T_CHK_SIZE),
                order=(2, 1, 0),
            ),
            out_block[None, :, :],
            boundary_check=(0, 1, 2),
        )