@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def duplicate_keys_shuffle_kernel(
    value_in, n_elements, philox_seed, philox_offset, BLOCK_SIZE: tl.constexpr
):
    pid0 = tl.program_id(0)
    offset_range = tl.arange(0, BLOCK_SIZE)
    value_offset = pid0.to(tl.int64) * BLOCK_SIZE + offset_range
    value_mask = value_offset < n_elements
    value_data = tl.load(value_in + value_offset, mask=value_mask)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)

    c0 = (philox_offset & 0xFFFFFFFF).to(tl.int32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.int32)

    i4 = pid0 * BLOCK_SIZE + offset_range
    c0 = c0 + i4
    _O = c0 * 0

    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)

    r0_int = r0.to(tl.int32)
    r1 = r0_int - (r0_int // BLOCK_SIZE) * BLOCK_SIZE

    mask_val = tl.full([BLOCK_SIZE], BLOCK_SIZE, dtype=tl.int32)
    r1 = tl.where(value_mask, r1, mask_val)

    _, sorted_chunk_index = argsort(r1, offset_range, 0, descending=False)

    store_offset = pid0.to(tl.int64) * BLOCK_SIZE + sorted_chunk_index.to(tl.int64)
    tl.store(value_in + store_offset, value_data, mask=store_offset < n_elements)