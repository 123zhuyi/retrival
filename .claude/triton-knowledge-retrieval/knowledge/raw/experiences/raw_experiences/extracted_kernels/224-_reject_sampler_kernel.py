@triton.jit()
def _reject_sampler_kernel(
    output_token_ids_ptr,  # [batch, spec_step + 1]
    output_accept_lens_ptr,  # [batch]
    draft_token_ids_ptr,  # [batch, spec_step]
    draft_probs_ptr,  # [batch, spec_step]
    target_probs_ptr,  # [batch, spec_step + 1, vocab_size]
    uniform_random_ptr,  # [batch, 1]
    max_spec_len: tl.constexpr,
    vocab_size: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # draft
    batch_draft_token_ids_ptr = draft_token_ids_ptr + batch_idx * max_spec_len
    batch_draft_probs_ptr = draft_probs_ptr + batch_idx * max_spec_len
    batch_target_probs_ptr = target_probs_ptr + batch_idx * (max_spec_len + 1) * vocab_size

    batch_output_token_ids_ptr = output_token_ids_ptr + batch_idx * (max_spec_len + 1)
    batch_output_accept_lens_ptr = output_accept_lens_ptr + batch_idx

    batch_uniform_random = tl.load(uniform_random_ptr + batch_idx)

    # reject sampler
    accept_len = 0
    rejected = False
    for pos in range(0, max_spec_len):
        if not rejected:
            draft_token_id = tl.load(batch_draft_token_ids_ptr + pos)
            draft_prob = tl.load(batch_draft_probs_ptr + pos)
            target_prob = tl.load(batch_target_probs_ptr + pos * vocab_size + draft_token_id)

            if draft_prob > 0 and target_prob / draft_prob >= batch_uniform_random:
                # accept
                accept_len += 1
                tl.store(batch_output_token_ids_ptr + pos, draft_token_id)
            else:
                rejected = True

    tl.store(batch_output_accept_lens_ptr, accept_len)