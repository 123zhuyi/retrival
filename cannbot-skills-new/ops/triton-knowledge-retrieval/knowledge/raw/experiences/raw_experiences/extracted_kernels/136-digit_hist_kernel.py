@libentry()
@triton.jit
def digit_hist_kernel(
    digit_hist,
    key,
    n_elements,
    bits_per_pass,
    bins,
    passes,
    bit_mask,
    bins_segment,
    BLOCK_SIZE: tl.constexpr,
):
    bin_segid = tl.program_id(1)
    pid0 = tl.program_id(0)
    grid0 = tl.num_programs(0)

    key_offset = pid0.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key_mask = key_offset < n_elements
    key_data = tl.load(key + key_offset, mask=key_mask)
    ikey_data = radix_type_convert(key_data)
    bit_offset = 0
    for p in range(passes):
        key_digit = (ikey_data >> bit_offset) & bit_mask
        blk_bin_start = bin_segid * bins_segment
        for s in range(bins_segment):
            bin_id = s + blk_bin_start
            digit_mask = tl.where(key_digit == bin_id and key_mask, 1, 0)
            digit_sum = tl.sum(digit_mask)
            # +1 for exclusive
            bin_offset = p * (bins + 1) * grid0 + (bin_id + 1) * grid0 + pid0
            # reduce rather than global atomic for perf issue
            tl.store(digit_hist + bin_offset, digit_sum)
        tl.store(digit_hist + p * (bins + 1) * grid0 + pid0, 0, mask=bin_segid == 0)
        bit_offset += bits_per_pass