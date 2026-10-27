@triton.autotune(configs=[triton.Config({"BLOCK_SIZE_M": 128})], key=["num_tokens"])
@triton.jit
def efficient_entropy_triton_epilogue_tp_update(
    num_tokens,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    logprobs_scalar_ptr,
    reduction: int,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    eps = 1e-12
    eps_f32 = tl.full([1], eps, tl.float32)

    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(
        accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens
    )

    accumulate_safe = tl.maximum(accumulate, eps_f32)

    entropy_b = tl.load(
        entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens
    )
    entropy_b = tl.fdiv(entropy_b, accumulate_safe)
    tl.store(
        entropy_b_ptr + offs_m * stride_entropy_b, entropy_b, mask=offs_m < num_tokens
    )

    entropy = tl.log(accumulate_safe) + maximum - entropy_b
    tl.store(entropy_ptr + offs_m * stride_entropy, entropy, mask=offs_m < num_tokens)

    logprobs = tl.load(
        logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens
    )
    logprobs = maximum + tl.log(accumulate_safe) - logprobs

    logprobs = -1 * logprobs
    if reduction == 0:
        tl.store(
            logprobs_ptr + offs_m * stride_logprobs, logprobs, mask=offs_m < num_tokens
        )
    elif reduction == 1:
        logprobs_scalar = tl.sum(logprobs, axis=0)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)
    elif reduction == 2:
        logprobs_scalar = tl.sum(logprobs, axis=0) / num_tokens.to(tl.float32)
        tl.atomic_add(logprobs_scalar_ptr, logprobs_scalar)