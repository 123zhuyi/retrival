@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    grid_m = tl.cdiv(n_elements, BLOCK_SIZE_M)
    grid_n = tl.cdiv(embedding_dim, BLOCK_SIZE_N)
    total_2d_blocks = grid_m * grid_n

    for block_idx in tl.range(pid, total_2d_blocks, num_progs, num_stages=NUM_STAGES):
        block_m = block_idx // grid_n
        block_n = block_idx % grid_n

        start_m = block_m * BLOCK_SIZE_M
        start_n = block_n * BLOCK_SIZE_N

        offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        mask_m = offsets_m < n_elements

        indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)

        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        mask_n = offsets_n < embedding_dim

        block_mask = mask_m[:, None] & mask_n[None, :]

        grad_output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
        grad_output = tl.load(
            grad_output_ptr + grad_output_offsets,
            mask=block_mask,
            other=0.0,
        )

        grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + grad_weight_offsets,
            grad_output,
            mask=block_mask,
        )