@triton.jit
def cube1(
    Q_ptr,
    Wksp_K_ptr,
    # Strides for Q
    stride_qt1,
    stride_qn2,
    stride_qn1,
    stride_qd,
    # Strides for aux K
    stride_wkt1,
    stride_wkn2,
    stride_wksbs,
    stride_wkd,
    # Param
    D_qk: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_SBS: tl.constexpr,
    # Meta
    offs_G,
    offs_sbs,
    q_base,
    wk_base,
    # Block size
    BLOCK_K: tl.constexpr,
):
    loop_times: tl.constexpr = (D_qk + BLOCK_K - 1) // BLOCK_K

    qk_acc = tl.zeros([BLOCK_G, BLOCK_SBS], dtype=tl.float32)

    # Handle nope
    for idx_D in range(0, loop_times):
        offs_D = idx_D * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_D = offs_D < D_qk

        # q shape: [T1, N2, G, D1] -> [G, D1] -> [G, BLOCK_K]
        l1_q = tl.load(
            Q_ptr
            + (q_base + offs_G[:, None] * stride_qn1 + offs_D[None, :] * stride_qd),
            mask=mask_D[None, :],
        )

        # k shape: [T2, N2, 1, D1] -> [SBS, D1] -> [SBS, BLOCK_K]
        l1_k = tl.load(
            Wksp_K_ptr
            + (
                wk_base
                + offs_sbs[:, None] * stride_wksbs
                + offs_D[None, :] * stride_wkd
            ),
            mask=mask_D[None, :],
        )

        # [G, BLOCK_K] @ [BLOCK_K, SBS] -> [G, SBS]
        qk_acc += tl.dot(l1_q, l1_k.trans())

    return qk_acc