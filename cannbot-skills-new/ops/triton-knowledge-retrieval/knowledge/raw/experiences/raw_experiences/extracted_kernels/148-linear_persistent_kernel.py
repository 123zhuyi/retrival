@triton.jit
def linear_persistent_kernel(
    a_ptr,  # Pointer to tensor a, shape [M, K]
    b_ptr,  # Pointer to tensor b, shape [N, K]
    c_ptr,  # Pointer to output tensor c, shape [M, N]
    M,  # Number of rows in tensor a
    N,  # Number of rows in tensor b (number of columns in output c)
    K,  # Number of columns in both tensor a and tensor b
    stride_am,  # Stride of tensor a along dimension M (typically K)
    stride_ak,  # Stride of tensor a along dimension K (typically 1)
    stride_bn,  # Stride of tensor b along dimension N (typically K)
    stride_bk,  # Stride of tensor b along dimension K (typically 1)
    stride_cm,  # Stride of tensor c along dimension M (typically N)
    stride_cn,  # Stride of tensor c along dimension N (typically 1)
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
    BLOCK_K: tl.constexpr,  # Block size for K dimension
    NUM_BLOCKS_M: tl.constexpr,  # New: Number of blocks in M dimension
    NUM_BLOCKS_N: tl.constexpr,  # New: Number of blocks in N dimension
    GRID_SIZE: tl.constexpr,  # New: Fixed 1D grid size
):
    # Get current program's 1D index (1D grid)
    pid = tl.program_id(0)
    total_blocks = NUM_BLOCKS_M * NUM_BLOCKS_N  # Total number of output blocks

    # Loop over multiple blocks assigned to the current program
    for block_index in range(pid, total_blocks, GRID_SIZE):
        # Convert 1D block index to 2D coordinates (m_block, n_block)
        m_block = block_index // NUM_BLOCKS_N
        n_block = block_index % NUM_BLOCKS_N

        # Calculate starting indices of the current output block
        start_m = m_block * BLOCK_M
        start_n = n_block * BLOCK_N

        # Create row and column index ranges within the current block
        m_indices = start_m + tl.arange(0, BLOCK_M)
        n_indices = start_n + tl.arange(0, BLOCK_N)

        # Create masks to handle boundaries
        m_mask = m_indices < M
        n_mask = n_indices < N

        # Initialize accumulator to 0
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over K dimension with step size BLOCK_K
        for k_offset in range(0, K, BLOCK_K):
            k_indices = k_offset + tl.arange(0, BLOCK_K)
            k_mask = k_indices < K

            # Load block of tensor a: shape [BLOCK_M, BLOCK_K]
            a_ptrs = a_ptr + m_indices[:, None] * stride_am + k_indices[None, :] * stride_ak
            a_vals = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            # Load block of tensor b: shape [BLOCK_N, BLOCK_K]
            b_ptrs = b_ptr + n_indices[:, None] * stride_bn + k_indices[None, :] * stride_bk
            b_vals = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            # Explicitly transpose b matrix using tl.trans: shape becomes [BLOCK_K, BLOCK_N]
            b_vals_transposed = tl.trans(b_vals)

            # Compute matrix multiplication: a_vals × b_vals_transposed
            product = tl.dot(a_vals, b_vals_transposed)
            acc += product
        # Store result to output tensor c
        c_ptrs = c_ptr + m_indices[:, None] * stride_cm + n_indices[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])