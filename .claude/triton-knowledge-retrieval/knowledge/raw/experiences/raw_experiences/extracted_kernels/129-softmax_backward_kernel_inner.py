@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        # scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)
        scale = tl.zeros([TILE_M], dtype=tl.float32)

        for off in range(0, N, TILE_N):
            n_offsets = tl.arange(0, TILE_N) + off
            offsets = m_offsets[:, None] * N + n_offsets
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            # scale += out_tile * out_grad_tile
            scale += tl.sum(out_tile * out_grad_tile, axis=1)
            # n_offsets += TILE_N
            # offsets += TILE_N
        # scale = tl.sum(scale, 1)  # (TILE_M,)

        for off in range(0, N, TILE_N):
            n_offsets = tl.arange(0, TILE_N) + off
            offsets = m_offsets[:, None] * N + n_offsets
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)