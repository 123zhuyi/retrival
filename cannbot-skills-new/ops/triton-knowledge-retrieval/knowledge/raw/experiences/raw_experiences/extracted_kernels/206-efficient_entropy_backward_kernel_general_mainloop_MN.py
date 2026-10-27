@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 16,
            },
        ),
    ],
    key=["num_tokens", "hidden_size", "vocab_size"],
)
@triton.jit
def efficient_entropy_backward_kernel_general_mainloop_MN(
    num_tokens: int,
    hidden_size: int,
    vocab_size: int,
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
    stride_entropy_b: tl.int64,
    d_hidden_ptr,
    stride_d_hidden_m: tl.int64,
    stride_d_hidden_k: tl.int64,
    d_weight_ptr,
    stride_d_weight_n: tl.int64,
    stride_d_weight_k: tl.int64,
    rcp_temperature: tl.float32,
    num_aicores: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    backward mainloop, where d_logits & d_hidden & d_weight are fused
    """
    pid0 = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_size, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    # num_aicores is the number of cube cores available on the current hardware. When launching the kernel,
    # the grid is set to num_aicores, and inside the kernel a loop is executed with num_aicores as the step size.
    # Compared with setting the grid to num_pid_n * num_pid_m, this approach can reduce the launch and
    # scheduling time caused by a large number of logical cores being dispatched.
    for pid in range(pid0, total_tiles, num_aicores):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_in_group = pid % num_pid_in_group
        valid_in_group = group_size_m * num_pid_n
        if pid_in_group < valid_in_group:
            pid_m = first_pid_m + (pid_in_group % group_size_m)
            pid_n = pid_in_group // group_size_m
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            maximum_ptrs = maximum_ptr + offs_am * stride_maximum
            maximum = tl.load(maximum_ptrs, mask=offs_am < num_tokens, other=0.0)
            accu_ptrs = accu_ptr + offs_am * stride_accu
            accu = tl.load(
                accu_ptrs, mask=offs_am < num_tokens, other=1e-6
            )  # epsilon to avoid division by zero
            accu_rcp = tl.fdiv(1.0, accu)

            d_entropy_ptrs = d_entropy_ptr + offs_am * stride_d_entropy
            d_entropy = tl.load(d_entropy_ptrs, mask=offs_am < num_tokens, other=0.0)
            if reduction == 0:  # none
                d_logprobs_ptrs = d_logprobs_ptr + offs_am * stride_d_logprobs
                d_logprobs = tl.load(
                    d_logprobs_ptrs, mask=offs_am < num_tokens, other=0.0
                )
            elif reduction == 1:  # sum
                d_logprobs = tl.load(d_logprobs_ptr)
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            else:  # mean
                d_logprobs = tl.fdiv(tl.load(d_logprobs_ptr), num_tokens.to(tl.float32))
                d_logprobs = tl.broadcast_to(d_logprobs, (BLOCK_SIZE_M,))
            d_logprobs = -1 * d_logprobs

            entropy_b_ptrs = entropy_b_ptr + offs_am * stride_entropy_b
            entropy_b = tl.load(entropy_b_ptrs, mask=offs_am < num_tokens, other=0.0)

            hidden_ptrs = hidden_ptr + (
                offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k
            )
            weight_ptrs = weight_ptr + (
                offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k
            )
            labels_ptrs = labels_ptr + offs_am * stride_labels
            labels = tl.load(labels_ptrs, mask=offs_am < num_tokens, other=0).to(
                tl.int32
            )

            logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                mask_k = offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=mask_k & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )
                _weight = tl.load(
                    weight_ptrs,
                    mask=mask_k & (offs_bn[:, None] < vocab_size),
                    other=0.0,
                )

                logits = tl.dot(_hidden, _weight.trans(), logits)

                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # scale logits by temperature
            logits *= rcp_temperature

            exp_logits = tl.exp(logits - maximum[:, None])

            mask = (offs_bn + rank * vocab_size)[None, :].to(tl.int32) == labels[
                :, None
            ]
            d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
            d_logits += (
                d_entropy[:, None]
                * (-exp_logits * accu_rcp[:, None])
                * (logits - entropy_b[:, None])
            )

            # scale d_logits by temperature
            d_logits *= rcp_temperature
            d_logits = d_logits.to(tl.bfloat16)
            d_logits_trans = d_logits.trans()

            # loop for d_weight & d_hidden
            for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
                mask_k = offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K
                mask_am = mask_k & (offs_am[:, None] < num_tokens)
                mask_bn = mask_k & (offs_bn[:, None] < vocab_size)

                # When performing loads inside a loop, directly adding the target offset to the base address
                # each time—rather than maintaining a separate offset variable and incrementing it by BLOCK_SIZE_K * stride_hidden_k
                # on each iteration—makes it easier for the compiler to infer information about the data being loaded,
                # thereby improving the performance of the load operations.
                _hidden = tl.load(
                    hidden_ptr
                    + (
                        offs_am[:, None] * stride_hidden_m
                        + offs_k[None, :] * stride_hidden_k
                    )
                    + k * BLOCK_SIZE_K * stride_hidden_k,
                    mask=mask_am,
                    other=0.0,
                )

                _d_weight = tl.dot(d_logits_trans, _hidden)
                _weight = tl.load(
                    weight_ptr
                    + (
                        offs_bn[:, None] * stride_weight_n
                        + offs_k[None, :] * stride_weight_k
                    )
                    + k * BLOCK_SIZE_K * stride_weight_k,
                    mask=mask_bn,
                    other=0.0,
                )
                tl.atomic_add(
                    d_weight_ptr
                    + offs_bn[:, None] * stride_d_weight_n
                    + offs_k[None, :] * stride_d_weight_k
                    + k * BLOCK_SIZE_K * stride_d_weight_k,
                    _d_weight,
                    mask=mask_bn,
                )

                _d_hidden = tl.dot(d_logits, _weight)

                tl.atomic_add(
                    d_hidden_ptr
                    + offs_am[:, None] * stride_d_hidden_m
                    + offs_k[None, :] * stride_d_hidden_k
                    + k * BLOCK_SIZE_K * stride_d_hidden_k,
                    _d_hidden,
                    mask=mask_am,
                )