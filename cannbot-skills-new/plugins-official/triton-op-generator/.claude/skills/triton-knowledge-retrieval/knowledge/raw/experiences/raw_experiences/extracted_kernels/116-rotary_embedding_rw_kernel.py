@triton.jit
def rotary_embedding_rw_kernel(
    state_out,
    state,
    cos,
    sin,
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    num_tokens,
    num_heads,
    token_range,
    head_range,
    dim_range_x,
    dim_range_y,
    rotary_interleaved: tl.constexpr,
):
    state_x_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_x[None, None, :] * stride_state_d
    )
    state_y_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_y[None, None, :] * stride_state_d
    )

    cos_sim_offset = (
        token_range[:, None, None] * stride_cos_n
        + dim_range_x[None, None, :] * stride_cos_d
    )
    if rotary_interleaved:
        sin_sim_offset = (
            token_range[:, None, None] * stride_cos_n
            + dim_range_y[None, None, :] * stride_cos_d
        )
    else:
        sin_sim_offset = cos_sim_offset

    state_x = tl.load(
        state + state_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    state_y = tl.load(
        state + state_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos_loaded = tl.load(
        cos + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)
    sin_loaded = tl.load(
        sin + sin_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)

    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(
        state_out + state_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        state_out + state_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )