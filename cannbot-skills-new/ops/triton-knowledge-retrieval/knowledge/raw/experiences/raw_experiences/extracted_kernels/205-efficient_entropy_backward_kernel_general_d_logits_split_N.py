@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 4,
            },
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    reduction: int,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (
        split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    )
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(
        maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0
    )
    accu = tl.load(
        accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6
    )
    accu_rcp = tl.fdiv(1.0, accu)
    d_entropy = tl.load(
        d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0
    )
    if reduction == 0:
        d_logprobs = tl.load(
            d_logprobs_ptr + offs_am * stride_d_logprobs,
            mask=offs_am < num_tokens,
            other=0.0,
        )
    elif reduction == 1:
        d_logprobs = tl.load(d_logprobs_ptr)
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    else:
        d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
        d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
    d_logprobs = -1 * d_logprobs
    entropy_b = tl.load(
        entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0
    )
    labels = tl.load(
        labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0
    ).to(tl.int32)

    hidden_ptrs = hidden_ptr + (
        offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
    )
    weight_ptrs = weight_ptr + (
        offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
    )

    vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)
    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        _hidden = tl.load(
            hidden_ptrs,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
            & (offs_am[:, None] < num_tokens),
            other=0.0,
        )
        _weight = tl.load(
            weight_ptrs,
            mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
            & (offs_bn[:, None] < vocab_right_bound),
            other=0.0,
        )
        logits = tl.dot(_hidden, _weight.trans(), logits)

        hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
        weight_ptrs += BLOCK_SIZE_K * stride_weight_k

    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    # In NPU，when performing equality checks, if the operands are of type int64, the operation degrades to
    # a scalar operation. Therefore, during equality comparison, if the data can be converted to int32 without overflow,
    # it can be converted to int32 to improve performance.
    mask = ((offs_bn + rank * vocab_size)[None, :]).to(tl.int32) == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    d_logits += (
        d_entropy[:, None]
        * (-exp_logits * accu_rcp[:, None])
        * (logits - entropy_b[:, None])
    )

    d_logits *= rcp_temperature

    # filter d_logits with mask
    result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_am[:, None] < num_tokens) & (result_offs_n[None, :] < vocab_per_split)

    tl.store(
        d_logits_ptr
        + offs_am[:, None] * stride_d_logits_m
        + result_offs_n[None, :] * stride_d_logits_n,
        d_logits,
        mask,
    )