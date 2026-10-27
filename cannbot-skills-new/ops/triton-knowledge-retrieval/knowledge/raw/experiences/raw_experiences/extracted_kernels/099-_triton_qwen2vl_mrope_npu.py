@triton.jit
def _triton_qwen2vl_mrope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    sin,
    sl,
    bs: tl.constexpr,
    total_rows: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BACKWARD_PASS: tl.constexpr = False,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)

    rows_per_program = (total_rows + num_programs - 1) // num_programs
    start_row = program_id * rows_per_program
    actual_rows = tl.minimum(rows_per_program, total_rows - start_row)

    for row_offset in tl.range(0, actual_rows, num_stages=NUM_STAGES):
        pid = start_row + row_offset

        t_end = mrope_section_t
        h_end = t_end + mrope_section_h

        t_cos = cos + pid * hd
        h_cos = t_cos + bs * sl * hd
        w_cos = h_cos + bs * sl * hd
        t_sin = sin + pid * hd
        h_sin = t_sin + bs * sl * hd
        w_sin = h_sin + bs * sl * hd

        q_base = q_ptr + pid * q_row_stride
        k_base = k_ptr + pid * k_row_stride

        d_idx = tl.arange(0, hd // 2)
        d_mask = d_idx < (hd // 2)

        pos_mask_t = d_idx < t_end
        pos_mask_h = (d_idx >= t_end) & (d_idx < h_end)

        text_cos_vals = tl.load(t_cos + d_idx, mask=d_mask, other=0)
        text_sin_vals = tl.load(t_sin + d_idx, mask=d_mask, other=0)
        height_cos_vals = tl.load(h_cos + d_idx, mask=d_mask, other=0)
        height_sin_vals = tl.load(h_sin + d_idx, mask=d_mask, other=0)
        width_cos_vals = tl.load(w_cos + d_idx, mask=d_mask, other=0)
        width_sin_vals = tl.load(w_sin + d_idx, mask=d_mask, other=0)

        cos_vals = tl.where(pos_mask_t, text_cos_vals, tl.where(pos_mask_h, height_cos_vals, width_cos_vals))
        sin_vals = tl.where(pos_mask_t, text_sin_vals, tl.where(pos_mask_h, height_sin_vals, width_sin_vals))

        # Process q heads in chunks to prevent UB overflow
        for qh_block in range(0, n_qh, BLOCK_Q):
            qh_idx = tl.arange(0, BLOCK_Q) + qh_block
            qh_mask = qh_idx < n_qh

            block_mask = qh_mask[:, None] & d_mask[None, :]
            offsets = qh_idx[:, None] * hd + d_idx[None, :]

            q_left = tl.load(q_base + offsets, mask=block_mask, other=0)
            q_right = tl.load(q_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = q_left * cos_vals - q_right * sin_vals
                new_right = q_right * cos_vals + q_left * sin_vals
            else:
                new_left = q_left * cos_vals + q_right * sin_vals
                new_right = q_right * cos_vals - q_left * sin_vals

            tl.store(q_base + offsets, new_left, mask=block_mask)
            tl.store(q_base + offsets + (hd // 2), new_right, mask=block_mask)

        # Process k heads in chunks to prevent UB overflow
        for kh_block in range(0, n_kh, BLOCK_K):
            kh_idx = tl.arange(0, BLOCK_K) + kh_block
            kh_mask = kh_idx < n_kh

            block_mask = kh_mask[:, None] & d_mask[None, :]
            offsets = kh_idx[:, None] * hd + d_idx[None, :]

            k_left = tl.load(k_base + offsets, mask=block_mask, other=0)
            k_right = tl.load(k_base + offsets + (hd // 2), mask=block_mask, other=0)

            if not BACKWARD_PASS:
                new_left = k_left * cos_vals - k_right * sin_vals
                new_right = k_right * cos_vals + k_left * sin_vals
            else:
                new_left = k_left * cos_vals + k_right * sin_vals
                new_right = k_right * cos_vals - k_left * sin_vals

            tl.store(k_base + offsets, new_left, mask=block_mask)
            tl.store(k_base + offsets + (hd // 2), new_right, mask=block_mask)