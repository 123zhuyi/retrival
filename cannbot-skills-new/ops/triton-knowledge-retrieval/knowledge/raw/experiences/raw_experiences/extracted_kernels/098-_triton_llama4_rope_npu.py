@triton.jit
def _triton_llama4_rope_npu(
    q_ptr,
    k_ptr,
    freqs_complex_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_row_stride,
    sl,
    bs: tl.constexpr,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    imag_sign: tl.constexpr,
):
    """
    Llama4 RoPE on Ascend NPU for interleaved complex layout:
    - q/k shape: (bs, sl, n_heads, hd)
    - freqs_complex_ptr: (sl, hd//2, 2)
    """
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // sl
    seq_idx = pid % sl

    if batch_idx >= bs:
        return

    q_base = q_ptr + pid * q_row_stride
    k_base = k_ptr + pid * k_row_stride

    freq_base = seq_idx * freqs_row_stride
    hd_idx = tl.arange(0, hd)
    hd_mask = hd_idx < (hd)

    freq_idx = tl.arange(0, hd)
    freq_mask = freq_idx < (hd)

    freqs_complex = tl.load(freqs_complex_ptr + freq_base + freq_idx, mask=freq_mask, other=0.0)

    freqs_complex = freqs_complex.reshape(hd // 2, 2, can_reorder=True)
    freqs_real, freqs_imag = tl.split(freqs_complex)
    freqs_imag = freqs_imag * imag_sign

    # Q heads (chunked for UB)
    for qh_block in range(0, n_qh, BLOCK_Q):
        qh_idx = tl.arange(0, BLOCK_Q) + qh_block
        qh_mask = qh_idx < n_qh
        block_mask = qh_mask[:, None] & hd_mask[None, :]

        head_ptr = q_base + qh_idx[:, None] * q_head_stride

        q_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )
        q_pair = q_pair.reshape(BLOCK_Q, hd // 2, 2, can_reorder=True)
        q_real, q_imag = tl.split(q_pair)

        new_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
        new_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)
        new_q_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_q_pair, mask=block_mask)

    # K heads (chunked for UB)
    for kh_block in range(0, n_kh, BLOCK_K):
        kh_idx = tl.arange(0, BLOCK_K) + kh_block
        kh_mask = kh_idx < n_kh
        block_mask = kh_mask[:, None] & hd_mask[None, :]

        head_ptr = k_base + kh_idx[:, None] * k_head_stride

        k_pair = tl.load(
            head_ptr + hd_idx[None, :],
            mask=block_mask,
            other=0.0,
        )

        k_pair = k_pair.reshape(BLOCK_K, hd // 2, 2, can_reorder=True)
        k_real, k_imag = tl.split(k_pair)

        new_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
        new_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)
        new_k_pair = tl.interleave(new_real, new_imag)

        tl.store(head_ptr + hd_idx[None, :], new_k_pair, mask=block_mask)