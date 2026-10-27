@triton.jit
def matmul_bias_persistent_kernel(
    # Input tensor pointers
    x_ptr,
    y_ptr,
    bias_ptr,
    output_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Stride information
    stride_xm,
    stride_xk,  # Strides of x: [M, K]
    stride_yk,
    stride_yn,  # Strides of y: [K, N]
    stride_bias,  # Stride of bias: [N]
    stride_outm,
    stride_outn,  # Strides of output: [M, N]
    # Whether to use bias
    has_bias: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # Row block ID
    pid_n = tl.program_id(1)  # Column block ID

    # Calculate the starting position of the current block in the matrix
    rm_start = pid_m * BLOCK_M
    rn_start = pid_n * BLOCK_N

    # Create index ranges
    rm = rm_start + tl.arange(0, BLOCK_M)  # Row index range [BLOCK_M]
    rn = rn_start + tl.arange(0, BLOCK_N)  # Column index range [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)  # K dimension index range [BLOCK_K]

    # Initialize accumulator to 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over the K dimension, processing BLOCK_K elements per iteration
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        # Calculate pointer offsets for x (row-major)
        x_ptrs = x_ptr + rm[:, None] * stride_xm + (rk[None, :] + k_start) * stride_xk
        # Calculate pointer offsets for y (row-major)
        y_ptrs = y_ptr + (rk[:, None] + k_start) * stride_yk + rn[None, :] * stride_yn

        # Create masks to prevent out-of-bounds access
        x_mask = (rm[:, None] < M) & ((rk[None, :] + k_start) < K)
        y_mask = ((rk[:, None] + k_start) < K) & (rn[None, :] < N)

        # Load data chunks from global memory
        x_chunk = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
        y_chunk = tl.load(y_ptrs, mask=y_mask, other=0.0).to(tl.float32)

        # Compute matrix multiplication accumulation
        acc += tl.dot(x_chunk, y_chunk, allow_tf32=False)

    # Add bias if the has_bias flag is set
    if has_bias:
        # Load bias values (broadcast to all rows)
        bias_ptrs = bias_ptr + rn * stride_bias
        bias_mask = rn < N
        bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        # Add bias to accumulator (automatic broadcasting)
        acc += bias_vals[None, :]

    # Calculate output pointer positions
    out_ptrs = output_ptr + rm[:, None] * stride_outm + rn[None, :] * stride_outn
    out_mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Store result to global memory
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=out_mask)