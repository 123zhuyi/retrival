@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    bonus_token_ids_ptr,  # [batch_size]
    recovered_token_ids_ptr,  # [num_tokens]
    uniform_probs_ptr,  # [num_tokens]
    is_greedy_ptr,  # [batch_size]
    max_spec_len,
    vocab_size,
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len
    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0
    start_idxs = tl.where(offsets == 0, 0, tl.load(cu_num_draft_tokens_ptr + offsets - 1, not_greedy_mask))
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask)
    n_num_draft_tokens = end_idxs - start_idxs
    for req_i in range(BLOCK_SIZE):
        not_greedy = tl.get_element(not_greedy_mask, (req_i,))
        if not_greedy:
            rejected = False
            start_idx = tl.get_element(start_idxs, (req_i,))
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = tl.get_element(n_num_draft_tokens, (req_i,))
            for pos in range(num_draft_tokens):
                if not rejected:
                    draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    if NO_DRAFT_PROBS:
                        draft_prob = 1
                    else:
                        draft_prob = tl.load(draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                    target_prob = tl.load(target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id)
                    uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
                    # NOTE(woosuk): While the draft probability should never be 0,
                    # we check it to avoid NaNs. If it happens to be 0, we reject.
                    if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                        # Accept.
                        token_id = draft_token_id
                    else:
                        # Reject. Use recovered token.
                        rejected = True
                        token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)
            if not rejected:
                # If all tokens are accepted, append the bonus token.
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(
                    output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                    bonus_token_id,
                )