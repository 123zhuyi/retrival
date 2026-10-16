@triton.jit
def sigmoid_kernel(
    QQ,
    O_SIGMOID,
    n_rows,
    in_stride_row: tl.constexpr,
    SIGMOID_DIM_START: tl.constexpr,
    SIGMOID_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    # 最低维拆分QQ，一半做sigmoid
    pid = tl.program_id(0)
    offsets = tl.arange(0, SIGMOID_DIM)
    NUM_BLOCKS = tl.cdiv(n_rows, BLOCK)
    for row_block_id in range(pid, NUM_BLOCKS, NUM_CORES):
        pos_offset = row_block_id * BLOCK + tl.arange(0, BLOCK)
        pos_mask = (pos_offset < n_rows)[:, None]
        # sigmoid
        q = tl.load(
            QQ
            + pos_offset[:, None] * in_stride_row
            + offsets[None, :]
            + SIGMOID_DIM_START,
            mask=pos_mask,
        )
        out = tl.sigmoid(q.to(tl.float32))
        tl.store(
            O_SIGMOID + pos_offset[:, None] * SIGMOID_DIM + offsets[None, :],
            out.to(q.dtype),
            mask=pos_mask,
        )