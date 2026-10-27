@triton.jit
def grpo_kernel(
    # Pointers
    new_logits_ptr,
    ref_logits_ptr,
    old_logits_ptr,
    input_ids_ptr,
    advantages_ptr,
    mask_ptr,
    loss_i_ptr,
    kl_i_ptr,
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
        tl.store(loss_i_ptr + pid, 0.0)
        tl.store(kl_i_ptr + pid, 0.0)
        return

    new_ptr = new_logits_ptr + pid * V
    ref_ptr = ref_logits_ptr + pid * V
    old_ptr = old_logits_ptr + pid * V

    # new_logits
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(new_ptr + offsets, mask=v_mask, other=-float("inf"))
        chunk = chunk / temperature
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_new = m_val + tl.log(s_val)

    # ref_logits
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(ref_ptr + offsets, mask=v_mask, other=-float("inf"))
        chunk = chunk / temperature
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_ref = m_val + tl.log(s_val)

    # old_logits
    m_val, s_val = -float("inf"), 0.0
    for v_start in range(0, V, BLOCK_SIZE_V):
        offsets = v_start + tl.arange(0, BLOCK_SIZE_V)
        v_mask = offsets < V
        chunk = tl.load(old_ptr + offsets, mask=v_mask, other=-float("inf"))
        chunk = chunk / temperature
        m_val, s_val = _update_logsumexp(chunk, m_val, s_val)
    lse_old = m_val + tl.log(s_val)

    input_ids = tl.load(input_ids_ptr + pid)
    new_x = (tl.load(new_ptr + input_ids)) / temperature
    ref_x = (tl.load(ref_ptr + input_ids)) / temperature
    old_x = (tl.load(old_ptr + input_ids)) / temperature

    new_lp = new_x - lse_new
    ref_lp = ref_x - lse_ref
    old_lp = old_x - lse_old

    kl_i = 0.0
    if beta != 0.0:
        # kl(p_ref || p_new) = sum(p_ref * (log(p_ref) - log(p_new)))
        # Approximated as exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0 under single-point sampling
        kl_i = tl.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0

    advantages = tl.load(advantages_ptr + pid)

    # coef1 is the unclipped ratio
    coef1 = tl.exp(new_lp - old_lp)

    # loss1 is the loss term with delta clipping
    loss1_coef = coef1
    if HAS_DELTA:
        loss1_coef = tl.minimum(coef1, delta)
    loss1 = loss1_coef * advantages

    # loss2 is the loss term with epsilon clipping
    coef2 = tl.clamp(coef1, 1.0 - epsilon_low, 1.0 + epsilon_high)
    loss2 = coef2 * advantages

    # Final per-token loss
    loss_i = -tl.minimum(loss1, loss2)
    if beta != 0.0:
        loss_i += beta * kl_i

    tl.store(loss_i_ptr + pid, loss_i)
    tl.store(kl_i_ptr + pid, kl_i)