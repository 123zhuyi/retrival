@triton.jit()
def _causal_conv1d_update_kernel_no_cache_len_no_mtp(
    x_ptr,
    conv_state_ptr,
    weight_ptr,
    bias_ptr,
    conv_state_indices_ptr,
    out_ptr,
    pad_slot_id,
    batch: tl.constexpr,
    dim: tl.constexpr,
    align_val: tl.constexpr,
    state_len: tl.constexpr,
    seq_len: tl.constexpr,
    width: tl.constexpr,
    out_len: tl.constexpr,
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
):
    pid = tl.program_id(0)
    cat_len: tl.constexpr = state_len + seq_len
    sub_state_len: tl.constexpr = state_len - seq_len
    sub_align_dim: tl.constexpr = DIM_BLOCK // align_val

    conv_begin: tl.constexpr = (cat_len - width + 1) - seq_len

    if IS_CONTINUOUS_BATCHING:
        conv_batch_offs = tl.load(conv_state_indices_ptr + pid)
    else:
        conv_batch_offs = pid

    if USE_PAD_SLOT:
        if conv_batch_offs == pad_slot_id:
            # skip padding
            return

    for doffs in range(0, dim, DIM_BLOCK):

        conv_state = tl.load(
            conv_state_ptr
            + conv_batch_offs * conv_batch_stride
            + doffs * state_len
            + tl.arange(0, DIM_BLOCK * state_len)
        )
        conv_state_T = (
            conv_state.reshape(sub_align_dim, align_val * state_len)
            .trans()
            .reshape(align_val, state_len * sub_align_dim)
            .trans()
            .reshape(
                state_len * DIM_BLOCK,
            )
        )

        x = tl.load(
            x_ptr
            + pid * x_batch_stride
            + doffs * seq_len
            + tl.arange(0, DIM_BLOCK * seq_len)
        )
        x_T = (
            x.reshape(sub_align_dim, align_val * seq_len)
            .trans()
            .reshape(align_val, seq_len * sub_align_dim)
            .trans()
            .reshape(
                seq_len * DIM_BLOCK,
            )
        )

        x_new_T = tl.full([cat_len * DIM_BLOCK], 0, x_ptr.dtype.element_ty)
        x_new_T = tl.insert_slice(
            x_new_T,
            conv_state_T,
            offsets=(0,),
            sizes=(state_len * DIM_BLOCK,),
            strides=(1,),
        )  # [cat_len , DIM_BLOCK].view(-1)
        x_new_T = tl.insert_slice(
            x_new_T,
            x_T,
            offsets=(state_len * DIM_BLOCK,),
            sizes=(seq_len * DIM_BLOCK,),
            strides=(1,),
        )

        new_conv_state_T = tl.extract_slice(
            x_new_T, (seq_len * DIM_BLOCK,), (state_len * DIM_BLOCK,), (1,)
        )  # [state_len, DIM_BLOCK].view(-1)
        new_conv_state = (
            new_conv_state_T.reshape(state_len * align_val, sub_align_dim)
            .trans()
            .reshape(sub_align_dim * state_len, align_val)
            .trans()
            .reshape(
                DIM_BLOCK * state_len,
            )
        )  # [DIM_BLOCK, state_len].view(-1)
        tl.store(
            conv_state_ptr
            + conv_batch_offs * conv_batch_stride
            + doffs * state_len
            + tl.arange(0, DIM_BLOCK * state_len),
            new_conv_state,
        )

        weight = tl.load(weight_ptr + doffs * width + tl.arange(0, DIM_BLOCK * width))
        weight_T = (
            weight.reshape(sub_align_dim, align_val * width)
            .trans()
            .reshape(align_val, width * sub_align_dim)
            .trans()
            .reshape(
                width * DIM_BLOCK,
            )
        )  # [width, DIM_BLOCK].view(-1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + doffs + tl.arange(0, DIM_BLOCK))
        else:
            bias = 0

        if width == cat_len:
            result = (
                tl.sum((x_new_T.to(tl.float32) * weight_T).reshape(width, DIM_BLOCK), 0)
                + bias
            )
            if SILU_ACTIVATION:
                result = result / (1 + tl.exp(-result))
            tl.store(
                out_ptr
                + pid * out_batch_stride
                + (doffs + tl.arange(0, DIM_BLOCK)) * out_len,
                result,
            )
        else:
            for i in range(seq_len):
                x_conv_part = tl.extract_slice(
                    x_new_T, ((conv_begin + i) * DIM_BLOCK), (width * DIM_BLOCK), (1,)
                ).to(tl.float32)
                result = (
                    tl.sum((x_conv_part * weight_T).reshape(width, DIM_BLOCK), 0) + bias
                )
                if SILU_ACTIVATION:
                    result = result / (1 + tl.exp(-result))
                tl.store(
                    out_ptr
                    + pid * out_batch_stride
                    + (doffs + tl.arange(0, DIM_BLOCK)) * out_len,
                    result,
                )