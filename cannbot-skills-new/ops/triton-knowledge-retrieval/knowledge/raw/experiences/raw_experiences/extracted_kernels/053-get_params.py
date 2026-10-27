@triton.jit
def get_params(
    Q,
    K,
    V,
    Out,
    WS1,
    WS2,
    WS3,
    WS4,
    block_idx,
    NUM_BLOCKS_M,
    H,
    stride_qz,
    stride_qh,
    N_CTX,
    HEAD_DIM,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_om,
    stride_on,
    w12_stride_nb,
    w12_stride_bm,
    w12_stride_bn,
    w34_stride_nb,
    w34_stride_bm,
    w34_stride_dm,
    BLOCK_M,
    SUB_BLOCK_M,
    BLOCK_N,
    TASK_ID,
):
    task_hz_idx = block_idx // NUM_BLOCKS_M
    task_m_idx = block_idx % NUM_BLOCKS_M
    off_z = task_hz_idx // H
    off_h = task_hz_idx % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    # Create block pointers for Q, K, V, Output
    Q_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr_a = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(task_m_idx * BLOCK_M, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    O_ptr_b = tl.advance(O_ptr_a, (SUB_BLOCK_M, 0))
    O_ptr_c = tl.advance(O_ptr_a, (SUB_BLOCK_M * 2, 0))
    O_ptr_d = tl.advance(O_ptr_a, (SUB_BLOCK_M * 3, 0))

    WS1_block_ptr = tl.make_block_ptr(
        base=WS1 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS1_a = tl.make_block_ptr(
        base=WS1 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS1_b = tl.advance(WS1_a, (SUB_BLOCK_M, 0))
    WS1_c = tl.advance(WS1_a, (SUB_BLOCK_M * 2, 0))
    WS1_d = tl.advance(WS1_a, (SUB_BLOCK_M * 3, 0))

    WS2_block_ptr = tl.make_block_ptr(
        base=WS2 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS2_a = tl.make_block_ptr(
        base=WS2 + TASK_ID * w12_stride_nb,
        shape=(BLOCK_M, BLOCK_N),
        strides=(w12_stride_bm, w12_stride_bn),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    WS2_b = tl.advance(WS2_a, (SUB_BLOCK_M, 0))
    WS2_c = tl.advance(WS2_a, (SUB_BLOCK_M * 2, 0))
    WS2_d = tl.advance(WS2_a, (SUB_BLOCK_M * 3, 0))

    WS3_block_ptr = tl.make_block_ptr(
        base=WS3 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS3_a = tl.make_block_ptr(
        base=WS3 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS3_b = tl.advance(WS3_a, (SUB_BLOCK_M, 0))
    WS3_c = tl.advance(WS3_a, (SUB_BLOCK_M * 2, 0))
    WS3_d = tl.advance(WS3_a, (SUB_BLOCK_M * 3, 0))

    WS4_a = tl.make_block_ptr(
        base=WS4 + TASK_ID * w34_stride_nb,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(w34_stride_bm, w34_stride_dm),
        offsets=(0, 0),
        block_shape=(SUB_BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    WS4_b = tl.advance(WS4_a, (SUB_BLOCK_M, 0))
    WS4_c = tl.advance(WS4_a, (SUB_BLOCK_M * 2, 0))
    WS4_d = tl.advance(WS4_a, (SUB_BLOCK_M * 3, 0))

    return (
        Q_ptr,
        K_ptr,
        V_ptr,
        (O_ptr_a, O_ptr_b, O_ptr_c, O_ptr_d),
        WS1_block_ptr,
        (WS1_a, WS1_b, WS1_c, WS1_d),
        WS2_block_ptr,
        (WS2_a, WS2_b, WS2_c, WS2_d),
        WS3_block_ptr,
        (WS3_a, WS3_b, WS3_c, WS3_d),
        (WS4_a, WS4_b, WS4_c, WS4_d),
    )