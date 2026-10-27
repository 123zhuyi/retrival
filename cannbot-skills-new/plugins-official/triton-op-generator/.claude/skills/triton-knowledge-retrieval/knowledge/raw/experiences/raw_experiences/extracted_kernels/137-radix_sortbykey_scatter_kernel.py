@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("randperm"),
    key=["n_elements"],
)
@triton.jit
def radix_sortbykey_scatter_kernel(
    key_out,
    value_out,
    key_in,
    value_in,
    digit_hist,
    d_lookback,
    n_elements,
    bit_offset,
    passes,
    p,
    num_portions,
    portion_size,
    portion_id,
    bit_mask,
    bins_segment,
    max_tiles_per_portion,
    bins: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    LOOKBACK_PARTIAL_MASK = 1 << 30
    LOOKBACK_GLOBAL_MASK = 1 << 31
    LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK
    LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK

    pid0 = tl.program_id(0)
    portion_id_i64 = portion_id
    portion_id_i64 = portion_id_i64.to(tl.int64)
    key_offset = (
        portion_id_i64 * portion_size
        + pid0.to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    key_mask = key_offset < n_elements
    value_data = tl.load(value_in + key_offset, mask=key_mask)
    key_data = tl.load(key_in + key_offset, mask=key_mask)

    ikey_data = radix_type_convert(key_data)
    key_digit = (ikey_data >> bit_offset) & bit_mask

    blk_bin_start = tl.program_id(1) * bins_segment
    last_block = tl.program_id(0) == tl.num_programs(0) - 1
    for s in range(bins_segment):
        bin_id = s + blk_bin_start
        key_digit_mask = (key_digit == bin_id) & key_mask
        key_elem_mask = tl.where(key_digit_mask, 1, 0)
        key_block_rank = tl.cumsum(key_elem_mask)
        key_block_rank = tl.where(key_digit_mask, key_block_rank - 1, 0)
        bin_of_bucket = tl.sum(key_elem_mask)
        partial_counter = bin_of_bucket | LOOKBACK_PARTIAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            partial_counter,
            cache_modifier=".cg",
        )
        bin_offset = p * (bins + 1) + bin_id
        prefix_offsets = tl.load(
            digit_hist + bin_offset + portion_id * passes * (bins + 1)
        )
        bk = pid0 - 1

        inc_sum = bin_of_bucket

        while bk >= 0:
            rd_lbk_offset = (
                (portion_id * passes + p) * max_tiles_per_portion + bk
            ) * bins + bin_id
            partial_prefix = tl.load(d_lookback + rd_lbk_offset, volatile=True)
            max_wait = 1000
            wait_count = 0
            while partial_prefix == 0 and wait_count < max_wait:
                partial_prefix = tl.load(d_lookback + rd_lbk_offset, volatile=True)
                wait_count += 1
            inc_sum += (partial_prefix & LOOKBACK_VALUE_MASK).to(tl.int32)
            if partial_prefix & LOOKBACK_GLOBAL_MASK:
                # break
                bk = -1
            else:
                bk -= 1

        global_counter = inc_sum | LOOKBACK_GLOBAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            global_counter,
            cache_modifier=".cg",
        )
        inc_bucket_offset = prefix_offsets.to(tl.int64) + inc_sum.to(tl.int64)
        if last_block and portion_id < num_portions - 1:
            tl.store(
                digit_hist + bin_offset + (portion_id + 1) * passes * (bins + 1),
                inc_bucket_offset,
            )
        global_offsets = (
            inc_bucket_offset - bin_of_bucket.to(tl.int64) + key_block_rank.to(tl.int64)
        )
        tl.store(key_out + global_offsets, key_data, mask=key_digit_mask)
        tl.store(value_out + global_offsets, value_data, mask=key_digit_mask)