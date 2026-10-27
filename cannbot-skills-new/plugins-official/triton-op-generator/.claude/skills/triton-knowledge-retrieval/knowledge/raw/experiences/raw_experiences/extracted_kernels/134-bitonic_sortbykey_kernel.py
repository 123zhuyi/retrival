@libentry()
@triton.jit
def bitonic_sortbykey_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    index_ptr += cur_batch * N
    y_ptr += cur_batch * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_iinfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < N)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < N)