@triton.jit
def vec_a(Q, WS1, WS2, m_i, l_i, sm):
    qk = tl.load(WS1)
    qk = qk * sm
    m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
    qk = qk - m_ij[:, None]  # Stabilize
    p = tl.math.exp(qk)
    l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
    p_cast = p.to(Q.type.element_ty)
    tl.store(WS2, p_cast)
    # return l_ij, m_ij
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij  # Update softmax denominator
    return m_ij, alpha, l_i