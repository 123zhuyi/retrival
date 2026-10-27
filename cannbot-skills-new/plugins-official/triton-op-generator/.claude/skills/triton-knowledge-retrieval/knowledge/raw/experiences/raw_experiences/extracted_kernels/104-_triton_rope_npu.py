@triton.jit
def _triton_rope_npu(
    q_ptr,
    q_row_stride,
    k_ptr,
    k_row_stride,
    cos,
    cos_row_stride,
    sin,
    sin_row_stride,
    sl,
    total_rows: tl.constexpr,
    cos_bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
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

        row_idx = pid % sl
        cos_ptr = cos + tl.where(cos_bs == 1, row_idx * cos_row_stride, pid * cos_row_stride)
        sin_ptr = sin + tl.where(cos_bs == 1, row_idx * sin_row_stride, pid * sin_row_stride)

        # Pre-compute d_idx and cos/sin values outside loops (they don't depend on heads)
        d_idx = tl.arange(0, hd // 2)
        d_mask = d_idx < (hd // 2)  # Always True, but kept for clarity
        cos_vals = tl.load(cos_ptr + d_idx, mask=d_mask, other=0)
        sin_vals = tl.load(sin_ptr + d_idx, mask=d_mask, other=0)

        # Process q heads in chunks to prevent UB overflow
        for qh_block in range(0, n_qh, BLOCK_Q):
            qh_idx = tl.arange(0, BLOCK_Q) + qh_block
            qh_mask = qh_idx < n_qh

            # block_mask: qh_mask broadcasted over d_idx dimension
            block_mask = qh_mask[:, None]

            offsets = qh_idx[:, None] * hd + d_idx[None, :]
            q_base = q_ptr + pid * q_row_stride

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

            # block_mask: kh_mask broadcasted over d_idx dimension
            block_mask = kh_mask[:, None]

            offsets = kh_idx[:, None] * hd + d_idx[None, :]
            k_base = k_ptr + pid * k_row_stride

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