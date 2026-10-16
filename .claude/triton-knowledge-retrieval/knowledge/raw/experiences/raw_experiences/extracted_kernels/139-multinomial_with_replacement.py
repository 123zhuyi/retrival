@libentry()
@triton.jit(do_not_specialize=["K", "N", "philox_seed", "philox_offset"])
def multinomial_with_replacement(
    cdf_ptr, out_ptr, K, N, philox_seed, philox_offset, NBLOCK: tl.constexpr = 128
):
    # The computation is arranged in a 2d grid of blocks, each producing
    # a batch of samples for a particular distribution.
    #            <------------------- grid.x --------------------->
    #           |   dist0.batch0 | dist0.batch1 | dist0.batch2 ...
    #   grid.y  |   dist1.batch0 | dist1.batch1 | dist1.batch2 ...
    #           |   dist2.batch0 | dist2.batch1 | dist2.batch2 ...
    y_off = tl.program_id(1) * N
    n = tl.program_id(0) * NBLOCK + tl.arange(0, NBLOCK)
    rv, _, _, _ = uniform(philox_seed, philox_offset, y_off + n)

    # Do a binary search for each random number on the cumulative probabilities.
    # Each random number always selects the leftmost index of the data greater
    # than or equal to itself. However, this is likely to give a wrong result
    # in case the first probability is zero which is not expected to selected.
    # This error happens when the tossed random number is also zero. To avoid
    # this mistake, we simply perturb random variable with a small number.
    rv += 0.0001
    rv = tl.where(rv > 0.9999, 0.9999, rv)

    cdf_ptr += tl.program_id(1) * K
    start = tl.zeros((NBLOCK,), dtype=tl.int32)
    end = tl.zeros((NBLOCK,), dtype=tl.int32) + K - 1
    steps = tl.math.log2(K.to(tl.float32)).to(tl.int32) + 1
    for _ in range(steps):
        mid = start + (end - start) // 2
        x = tl.load(cdf_ptr + mid, mask=n < N)
        start = tl.where(x < rv, mid + 1, start)
        end = tl.where(x < rv, end, mid)

    # Returns the last index in case of an overflow
    start = tl.where(start >= K, K - 1, start)

    tl.store(out_ptr + y_off + n, start, mask=n < N)