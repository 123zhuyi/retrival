@triton.jit
def _join_prob_reject_sampler_kernel(
    output_token_ids_ptr,  # [batch, max_spec_len + 1]
    output_accept_lens_ptr,  # [batch]
    draft_token_ids_ptr,  # [batch, max_spec_len]
    draft_probs_ptr,  # [batch, max_spec_len]
    target_probs_ptr,  # [batch, max_spec_len + 1, vocab_size]
    uniform_random_ptr,  # [batch, max_spec_len]
    cum_probs_ptr,  # [batch, max_spec_len]
    cum_rand_ptr,  # [batch, max_spec_len]
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    # draft
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size
    batch_uniform_random = uniform_random_ptr + batch_idx * max_spec_len

    # cum probs
    batch_cum_probs_ptr = cum_probs_ptr + batch_idx * max_spec_len
    batch_cum_rand_ptr = cum_rand_ptr + batch_idx * max_spec_len

    # output ptr
    batch_output_token_ids_ptr = output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    spec_offset = tl.arange(0, max_spec_len)

    uniform_rand = tl.load(batch_uniform_random + spec_offset)
    draft_token_ids = tl.load(batch_draft_token_ids_ptr + spec_offset)
    draft_probs = tl.load(batch_draft_probs_ptr + spec_offset)
    target_probs = tl.load(batch_target_probs_ptr + spec_offset * vocab_size + draft_token_ids)

    ratio = target_probs / draft_probs

    cum_probs = tl.cumprod(ratio, axis=0)
    cum_probs = tl.clamp(cum_probs, 0, 1)
    cum_rands = tl.cumprod(uniform_rand, axis=0)

    tl.store(batch_cum_probs_ptr + spec_offset, cum_probs)
    tl.store(batch_cum_rand_ptr + spec_offset, cum_rands)

    accept_len = 0
    is_accept = False
    for pos in range(0, max_spec_len):
        if not is_accept:
            index = max_spec_len - pos - 1
            cum_prob = tl.load(batch_cum_probs_ptr + index)
            cum_rand = tl.load(batch_cum_rand_ptr + index)

            if cum_prob >= cum_rand:
                accept_len = index + 1
                is_accept = True

    write_mask = spec_offset < accept_len

    tl.store(batch_output_token_ids_ptr + spec_offset, draft_token_ids, mask=write_mask)
    tl.store(batch_output_accept_lens_ptr, accept_len)