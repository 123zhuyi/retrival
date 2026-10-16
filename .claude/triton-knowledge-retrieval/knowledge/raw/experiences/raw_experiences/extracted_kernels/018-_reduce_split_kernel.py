@triton.jit
def _reduce_split_kernel(
    Acc,
    Out,
    stride_ak,
    stride_abs,
    stride_ah,
    stride_ad,
    stride_obs,
    stride_oh,
    stride_od,
    head_size_v: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """second step kernel of split k attention."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    # initialize offsets
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_k = tl.arange(0, SPLIT_K)
    mask_dv = offs_dv < head_size_v

    offs_acc = (
        cur_batch * stride_abs
        + cur_head * stride_ah
        + offs_k[:, None] * stride_ak
        + offs_dv[None, :] * stride_ad
    )
    offs_mi = (
        cur_batch * stride_abs + cur_head * stride_ah + stride_ak * offs_k + head_size_v
    )

    acc_k = tl.load(Acc + offs_acc, mask=mask_dv[None, :], other=0.0)
    m_k = tl.load(Acc + offs_mi)
    l_k = tl.load(Acc + offs_mi + 1)

    m_max = tl.max(m_k, 0)
    alpha = fast_expf(m_k - m_max)
    acc_k = acc_k * alpha[:, None]
    l_k = l_k * alpha

    acc = tl.sum(acc_k, 0)
    l_sum = tl.sum(l_k, 0)
    acc = acc / l_sum

    out_offs = cur_batch * stride_obs + cur_head * stride_oh + offs_dv * stride_od
    tl.store(Out + out_offs, acc, mask=mask_dv)