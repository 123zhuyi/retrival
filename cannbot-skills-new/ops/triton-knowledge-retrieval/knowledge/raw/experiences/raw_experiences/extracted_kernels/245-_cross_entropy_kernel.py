@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
    ],
    key=["n_cols"],
    restore_value=["X_ptr"],
)
@libentry()
@triton.jit
def _cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    weight_ptr,
    loss_ptr,
    z_loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    sum_non_ignore_weight,
    weight_sum,
    ignore_index,
    lse_square_scale: tl.constexpr,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    softcap,
    OVERWRITE_GRAD_LOGITS: tl.constexpr,
    IS_BACKWARD: tl.constexpr,
    RETURN_Z_LOSS: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_SOFTCAPPING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel computes both cross entropy loss and the gradient of the input.
    We only consider hard label + mean reduction for now. Please refer to https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for the math.
    """

    # https://github.com/triton-lang/triton/issues/1058
    # If B*T*V is too large, program_id * stride will overflow out of int32, so we convert to int64
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    if HAS_WEIGHT:
        weight_y = tl.load(weight_ptr + y).cast(tl.float32)

    # Online softmax: 2 loads + 1 store (compared with 3 loads + 1 store for the safe softmax)
    # Refer to Algorithm 3 in the paper: https://arxiv.org/pdf/1805.02867

    # [Online softmax] first pass: find max + sum
    m = float("-inf")  # m is the max value. use the notation from the paper
    d = 0.0  # d is the sum. use the notation from the paper
    ori_X_y = tl.load(X_ptr + y).cast(tl.float32)  # we need to store the original value of X_y for the loss calculation

    if HAS_SOFTCAPPING:
        ori_X_y = softcap * tl.math.tanh(ori_X_y / softcap)

    # Label smoothing is a general case of normal cross entropy
    # See the full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issue-2503665310
    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_mask = X_offsets < n_cols
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_mask,
            other=float("-inf"),
            # Ensure float32 precision for softmax calculation
        ).cast(tl.float32)

        if HAS_SOFTCAPPING:
            X_block = softcap * tl.math.tanh(X_block / softcap)
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            X_block2 = tl.load(
                X_ptr + X_offsets,
                mask=X_mask,
                other=0.0,
            ).cast(tl.float32)
            # scale X beforehand to avoid overflow
            if HAS_WEIGHT:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                scaled_x_sum += tl.sum(-eps * X_block2 * weight_block).to(tl.float32)
            else:
                scaled_x_sum += tl.sum(-eps * X_block2).to(tl.float32)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    # log (sum(e^(X_i))) = log (sum(e ^ (max(X) * e ^ (X_i - max(X)))))
    #                    = log (e^(max(X)) * sum(e ^ (X_i - max(X))))
    #                    = max(X) + log (sum(e ^ (X_i - max(X)))) = m + log d
    lse = m + tl.log(d)

    if OVERWRITE_GRAD_LOGITS:
        # [Online Softmax] Second pass: compute gradients
        # For 'mean' reduction, gradients are normalized by number of non-ignored elements (N)
        # dx_y = (softmax(x_y) - 1) / N
        # dx_i = softmax(x_i) / N, i != y
        # For label smoothing:
        # dx_i = (softmax(x_i) - label_smoothing / V) / N, V = n_cols, i != y
        # dx_y = (softmax(x_y) - label_smoothing / V - (1 - label_smoothing)) / N
        #      = dx_i - (1 - label_smoothing) / N
        # With Z loss:
        # dx_i = ((1 + 2 * lse_square_scale * lse) * softmax(x_i) - label_smoothing / V) / N, i != y
        # dx_y = dx_i - (1 - label_smoothing) / N
        # For 'sum' reduction, no normalization is applied:
        # dx_y = softmax(x_y) - 1
        # dx_i = softmax(x_i), for i ≠ y

        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)

            y_is_in_this_block = (y >= i) and (y < i + BLOCK_SIZE)

            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
                # Ensure float32 precision for softmax calculation
            ).cast(tl.float32)
            if HAS_SOFTCAPPING:
                intermediate = tl.math.tanh(X_block / softcap)
                X_block = softcap * intermediate

            if not HAS_WEIGHT:
                # softmax(x_i)
                X_block = tl.exp(X_block - m) / d
                # derivative of z-loss: 2 * lse_square_scale * lse * softmax(x_i)
                X_block += 2 * lse_square_scale * lse * X_block
                # smoothing term
                X_block += -eps
                # special handle dx_y
                if y_is_in_this_block:
                    is_target_class_mask = X_offsets == y
                    X_block -= is_target_class_mask * (1 - label_smoothing)
                # X_block = tl.where(X_offsets != y, X_block, X_block - (1 - label_smoothing))
                # reduction scale
                if reduction == "mean":
                    X_block = X_block / n_non_ignore
            else:
                weight_block = tl.load(weight_ptr + X_offsets, mask=X_offsets < n_cols)
                softmax_X = tl.exp(X_block - m) / d
                # derivative of original_loss
                dloss_ori = (1 - label_smoothing) * softmax_X
                # specially handle dx_y
                # dloss_ori = tl.where(X_offsets != y, dloss_ori, dloss_ori - (1 - label_smoothing))
                if y_is_in_this_block:
                    is_target_class_mask = X_offsets == y
                    dloss_ori -= is_target_class_mask * (1 - label_smoothing)
                dloss_ori = dloss_ori * weight_y
                # derivative of smooth_loss
                dloss_smooth = eps * (-weight_block + softmax_X * weight_sum)
                # derivative of z-loss
                dz_loss = 2 * lse_square_scale * lse * softmax_X
                # reduction scale
                if reduction == "mean":
                    dloss_ori = dloss_ori / sum_non_ignore_weight
                    dloss_smooth = dloss_smooth / sum_non_ignore_weight
                    # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
                    dz_loss = dz_loss / n_non_ignore
                # derivative of total_loss
                X_block = dloss_ori + dloss_smooth + dz_loss

            # chain rule softcapping
            # d(softcap * tanh(x / softcap)) = (1 - tanh^2(x / softcap))
            if HAS_SOFTCAPPING:
                X_block = X_block * (1 - intermediate * intermediate)

            tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    if not IS_BACKWARD:
        loss_ptr += program_id * loss_stride
        if RETURN_Z_LOSS:
            z_loss_ptr += program_id * loss_stride

        # We need tl.debug_barrier() to ensure the new result of X_ptr is written as mentioned in
        # https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/ops/cross_entropy.py#L34
        # tl.debug_barrier()

        # Calculate the loss

        # loss = log (softmax(X_y)) = log ((e ^ (X_y - max(X)) / sum(e ^ (X - max(X))))
        #      = (X_y - max(X)) - log(sum(e ^ (X - max(X))))
        #      = X_y - m - log d = X_y - lse
        # sum(e ^ (X - max(X))) must >= 1 because the max term is e ^ 0 = 1
        # So we can safely calculate log (softmax(X_y)) without overflow
        loss = lse - ori_X_y
        if HAS_WEIGHT:
            loss = weight_y * loss

        # Original loss = H(q, p),  with label smoothing regularization = H(q', p) and (label_smoothing / V) = eps
        # H(q', p) = (1 - label_smoothing) * H(q, p) + label_smoothing * H(u, p)
        #          = (1 - label_smoothing) * H(q, p) + eps * sum(logsoftmax(x_i))
        # By using m (global max of xi) and d (sum of e^(xi-m)), we can simplify as:
        #          = (1 - label_smoothing) * H(q, p) + (sum(-eps * x_i) + label_smoothing * (m + logd))
        # Refer to H(q', p) in section 7 of the paper: https://arxiv.org/pdf/1512.00567
        # pytorch: https://github.com/pytorch/pytorch/blob/2981534f54d49fa3a9755c9b0855e7929c2527f0/aten/src/ATen/native/LossNLL.cpp#L516
        # See full derivation at https://github.com/linkedin/Liger-Kernel/pull/198#issuecomment-2333753087
        if label_smoothing > 0:
            if HAS_WEIGHT:
                smooth_loss = scaled_x_sum + eps * lse * weight_sum
            else:
                smooth_loss = scaled_x_sum + label_smoothing * lse
            loss = loss * (1 - label_smoothing) + smooth_loss

        # An auxiliary loss, z_loss
        # Refer to Page14 Loss function section in the paper PaLM: https://www.jmlr.org/papers/v24/22-1144.html
        z_loss = lse_square_scale * lse * lse
        # Normalize the loss by the number of non-ignored elements if reduction is "mean"
        if reduction == "mean":
            if HAS_WEIGHT:
                loss = loss / sum_non_ignore_weight
            else:
                loss = loss / n_non_ignore
            # TODO: Implement weighted z_loss. Currently, z_loss is not scaled by weight.
            z_loss = z_loss / n_non_ignore
        loss += z_loss

        tl.store(loss_ptr, loss)
        if RETURN_Z_LOSS:
            tl.store(z_loss_ptr, z_loss)