@triton.jit
def grpo_bwd_kernel(
    # Pointers
    new_logits_ptr,
    ref_logits_ptr,
    old_logits_ptr,
    input_ids_ptr,
    advantages_ptr,
    mask_ptr,
    grad_new_logits_ptr,  # Output
    # Upstream gradient info
    grad_loss_scale: float,
    # Dimensions
    BL,
    V,
    # Hyperparameters
    beta: float,
    epsilon_low: float,
    epsilon_high: float,
    delta: float,
    temperature: float,
    # Meta-parameters
    BLOCK_SIZE_V: tl.constexpr,
    HAS_DELTA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    mask = tl.load(mask_ptr + pid)
    if mask == 0:
        grad_ptr = grad_new_logits_ptr + pid * V
        for v_start in range(0, V, BLOCK_SIZE_V):
            offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
            v_mask = offsets < V
            tl.store(grad_ptr + offsets, 0.0, mask=v_mask)
        return

    # --- 1. Recomputation (Correct, no changes) ---
    # ... (code for lse, lp calculation is omitted for brevity)
    new_ptr = new_logits_ptr + pid * V
    ref_ptr = ref_logits_ptr + pid * V
    old_ptr = old_logits_ptr + pid * V
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = (
            tl.load(new_ptr + offsets, mask=v_mask, other=-float("inf")) / temperature
        )
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_new = m_val + tl.log(s_val)
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = (
            tl.load(ref_ptr + offsets, mask=v_mask, other=-float("inf")) / temperature
        )
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_ref = m_val + tl.log(s_val)
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = (
            tl.load(old_ptr + offsets, mask=v_mask, other=-float("inf")) / temperature
        )
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_old = m_val + tl.log(s_val)
    input_ids = tl.load(input_ids_ptr + pid)
    new_x = (tl.load(new_ptr + input_ids)) / temperature
    ref_x = (tl.load(ref_ptr + input_ids)) / temperature
    old_x = (tl.load(old_ptr + input_ids)) / temperature
    new_lp = new_x - lse_new
    ref_lp = ref_x - lse_ref
    old_lp = old_x - lse_old

    # --- 2. Backward Gradient Calculation ---
    advantages = tl.load(advantages_ptr + pid)
    coef1 = tl.exp(new_lp - old_lp)

    loss1_coef = coef1
    if HAS_DELTA:
        loss1_coef = tl.minimum(coef1, delta)
    loss1 = loss1_coef * advantages

    coef2 = tl.clamp(coef1, 1.0 - epsilon_low, 1.0 + epsilon_high)
    loss2 = coef2 * advantages

    g_loss_i = grad_loss_scale
    g_kl_i = g_loss_i * beta
    g_ppo_loss = g_loss_i

    g_min = -g_ppo_loss

    g_loss1 = tl.where(loss1 <= loss2, g_min, 0.0)
    g_loss2 = tl.where(loss1 > loss2, g_min, 0.0)

    # --- (Rest of the code is correct, no changes) ---
    g_coef2 = g_loss2 * advantages
    g_coef1_from_loss2 = tl.where(
        (coef1 >= 1.0 - epsilon_low) & (coef1 <= 1.0 + epsilon_high), g_coef2, 0.0
    )

    g_loss1_coef = g_loss1 * advantages
    g_coef1_from_loss1 = g_loss1_coef
    if HAS_DELTA:
        g_coef1_from_loss1 = tl.where(coef1 < delta, g_loss1_coef, 0.0)

    g_coef1 = g_coef1_from_loss1 + g_coef1_from_loss2

    g_new_lp_from_ppo = g_coef1 * coef1
    g_new_lp_from_kl = 0.0
    if beta != 0.0:
        g_new_lp_from_kl = g_kl_i * (1.0 - tl.exp(ref_lp - new_lp))

    g_new_lp = g_new_lp_from_ppo + g_new_lp_from_kl

    # --- 3. Final Gradient Writeback (Correct, no changes) ---
    grad_ptr = grad_new_logits_ptr + pid * V
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = (
            tl.load(new_ptr + offsets, mask=v_mask, other=-float("inf")) / temperature
        )
        softmax_chunk = tl.exp(chunk - lse_new)
        grad_dense = -g_new_lp * softmax_chunk
        grad_sparse = tl.where(offsets == input_ids, g_new_lp, 0.0)
        final_grad = (grad_dense + grad_sparse) / temperature
        tl.store(grad_ptr + offsets, final_grad, mask=v_mask)