@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("cross_entropy_loss"),
    key=["C", "D"],
)
@triton.jit(do_not_specialize=["label_smoothing", "mean_num"])
def celoss_probability_bwd(
    out_grad_ptr,
    inp_ptr,
    tgt_ptr,
    w_ptr,
    inp_grad_ptr,
    label_smoothing,
    mean_num,
    C,
    D,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tle.program_id(0)
    pid_n = tle.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    out_grad_ptrs = out_grad_ptr + pid_n * D + offset_d
    out_grad = tl.load(out_grad_ptrs, mask=offset_d < D, other=0).to(tl.float32)[
        None, :
    ]

    tmp_max = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    tmp_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)
    w_tgt_sum = tl.zeros([BLOCK_C, BLOCK_D], dtype=tl.float32)

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp_ptrs = inp_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp = tl.load(inp_ptrs, mask, other=-float("inf")).to(tl.float32)

        tgt_ptrs = tgt_ptr + pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        w_tgt_sum += tgt * w[:, None]

        cur_max = tl.maximum(tmp_max, inp)
        cur_exp = tl.exp(inp - cur_max)
        tmp_sum = tmp_sum * tl.exp(tmp_max - cur_max) + cur_exp
        tmp_max = cur_max
    final_max = tl.max(tmp_max, axis=0)[None, :]
    tmp_sum = tmp_sum * tl.exp(tmp_max - final_max)
    final_sum = tl.sum(tmp_sum, axis=0)[None, :]
    w_tgt_sum = tl.sum(w_tgt_sum, axis=0)[None, :]

    for off in range(0, C, BLOCK_C):
        offset_c = off + tl.arange(0, BLOCK_C)
        offset = pid_n * C * D + offset_c[:, None] * D + offset_d[None, :]
        inp_ptrs = inp_ptr + offset
        mask = offset_c[:, None] < C and offset_d[None, :] < D
        inp = tl.load(inp_ptrs, mask, other=0).to(tl.float32)

        tgt_ptrs = tgt_ptr + offset
        tgt = tl.load(tgt_ptrs, mask, other=0).to(tl.float32)
        tgt = tgt * (1 - label_smoothing) + label_smoothing / C

        w_mask = offset_c < C
        if w_ptr is None:
            w = w_mask
        else:
            w_ptrs = w_ptr + offset_c
            w = tl.load(w_ptrs, w_mask, other=0).to(tl.float32)

        grad = w_tgt_sum / final_sum * tl.exp(inp - final_max) - tgt * w[:, None]
        inp_grad = grad * out_grad * mean_num

        inp_grad_ptrs = inp_grad_ptr + offset
        tl.store(inp_grad_ptrs, inp_grad, mask)