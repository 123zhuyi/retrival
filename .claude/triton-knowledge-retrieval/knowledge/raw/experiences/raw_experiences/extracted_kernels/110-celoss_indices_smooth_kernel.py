@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["ignore_index", "label_smoothing"])
def celoss_indices_smooth_kernel(
    inp_ptr,
    tgt_ptr,
    w_ptr,
    out_ptr,
    w_tgt_ptr,
    ignore_index,
    label_smoothing,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    tgt_ptrs = tgt_ptr + pid_n * D + offset_d
    tgt_mask = offset_d < D
    tgt = tl.load(tgt_ptrs, mask=tgt_mask, other=0)

    ignore_mask = not (tgt == ignore_index) and tgt_mask

    if w_ptr is None:
        w_tgt = ignore_mask
    else:
        w_tgt = tl.load(w_ptr + tgt, mask=tgt_mask, other=0)
        w_tgt = tl.where(ignore_mask, w_tgt, 0)
    w_tgt_ptrs = w_tgt_ptr + pid_n * D + offset_d
    tl.store(w_tgt_ptrs, w_tgt, mask=tgt_mask)

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)
        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.log(tl.sum(tmp_sum, axis=0))[None, :]
    final_sum_max = final_sum + final_max

    _sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w = tl.load(w_ptr + offset_c, w_mask, other=0).to(tl.float32)

        smooth = tl.where(
            offset_c[:, None] == tgt[None, :],
            1 - label_smoothing + label_smoothing / C,
            label_smoothing / C,
        ).to(tl.float32)

        log = final_sum_max - inp
        _sum += log * smooth * w[:, None]

    out = tl.sum(_sum, axis=0)
    out = tl.where(ignore_mask, out, 0)
    out_ptrs = out_ptr + pid_n * D + offset_d
    tl.store(out_ptrs, out, mask=tgt_mask)