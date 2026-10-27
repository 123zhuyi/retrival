@triton.jit
def _tv_distance_kernel(
    p_ptr,
    p_stride,
    q_ptr,
    q_stride,
    loss_ptr,
    loss_stride,
    grads_ptr,
    grads_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,  # V
    total_rows: tl.constexpr,  # BT
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    reduction: tl.constexpr = "batchmean",
):
    thread_id = tl.program_id(0)
    num_threads = tl.num_programs(0)

    for pid in tl.range(thread_id, total_rows, num_threads, num_stages=NUM_STAGES):
        p_row_ptr = p_ptr + pid * p_stride
        q_row_ptr = q_ptr + pid * q_stride
        loss_row_ptr = loss_ptr + pid * loss_stride
        grads_row_ptr = grads_ptr + pid * grads_stride
        label_row_ptr = label_ptr + pid

        base_offsets = tl.arange(0, BLOCK_SIZE)

        should_skip = False
        if HAS_LABEL:
            label = tl.load(label_row_ptr)
            if label == ignore_index:
                should_skip = True

        if should_skip:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols
                tl.store(grads_row_ptr + offsets, 0.0, mask=mask)
                if reduction == "none":
                    tl.store(loss_row_ptr + offsets, 0.0, mask=mask)
        else:
            loss_sum = 0.0
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols

                p = tl.load(p_row_ptr + offsets, mask=mask, other=0.0)
                q = tl.load(q_row_ptr + offsets, mask=mask, other=0.0)

                # TVD(P || Q) = 0.5 * |P - Q|
                tv_loss = 0.5 * tl.abs(p - q)
                grad_res = tl.where(p > q, 0.5, -0.5)

                tl.store(grads_row_ptr + offsets, grad_res, mask=mask)

                if reduction == "none":
                    tl.store(loss_row_ptr + offsets, tv_loss, mask=mask)
                else:
                    loss_sum += tl.sum(tv_loss, axis=0)

            if reduction != "none":
                tl.store(loss_row_ptr, loss_sum)