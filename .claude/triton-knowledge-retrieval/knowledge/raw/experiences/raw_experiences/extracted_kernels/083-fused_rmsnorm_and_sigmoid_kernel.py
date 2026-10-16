@triton.jit
def fused_rmsnorm_and_sigmoid_kernel(
    QQ,
    W,
    O_NORM,
    O_SIGMOID,
    n_rows,
    eps: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # 拆分QQ，一半做rmsnorm，另一半做sigmoid
    pid = tl.program_id(0)
    HALF_DIM: tl.constexpr = DIM // 2
    offsets = tl.arange(0, HALF_DIM)
    w = tl.load(W + offsets)
    w = tl.expand_dims(w, 0)
    w = tl.broadcast_to(w, (BLOCK, HALF_DIM))
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        # rms_norm
        q = tl.load(QQ + pos_offset[:, None] * DIM + offsets[None, :], mask=pos_mask)
        qf = q.to(tl.float32)
        var = tl.sum(qf * qf, 1) / HALF_DIM

        qrt = tl.expand_dims(tl.math.rsqrt(var + eps), 1)
        out = qf * tl.broadcast_to(qrt, (BLOCK, HALF_DIM))
        out = w * out.to(q.dtype)
        tl.store(
            O_NORM + pos_offset[:, None] * HALF_DIM + offsets[None, :],
            out,
            mask=pos_mask,
        )
        # sigmoid
        q = tl.load(
            QQ + pos_offset[:, None] * DIM + offsets[None, :] + HALF_DIM, mask=pos_mask
        )
        out = tl.sigmoid(q.to(tl.float32))
        tl.store(
            O_SIGMOID + pos_offset[:, None] * HALF_DIM + offsets[None, :],
            out.to(q.dtype),
            mask=pos_mask,
        )