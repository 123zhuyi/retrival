@triton.jit
def rms_norm_rope_kernel(
    q1,
    q2,
    k1,
    k2,
    v,
    weight,
    cos,
    sin,
    q1_out,
    q2_out,
    k1_out,
    k2_out,
    v_out,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLK: tl.constexpr,
):
    id = tl.program_id(0)
    num_q_offsets = tl.arange(0, q_size // head_dim)
    num_kv_offsets = tl.arange(0, kv_size // head_dim)
    head_offsets = tl.arange(0, head_dim)
    half_head_offsets = tl.arange(0, head_dim // 2)
    num_token_offsets = tl.arange(0, BLOCK_SIZE) + id * BLOCK_SIZE
    half_q_size = q_size // 2
    half_kv_size = kv_size // 2
    half_dim = head_dim // 2

    q1_data = tl.load(
        q1
        + num_token_offsets[:, None, None] * half_q_size
        + num_q_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    q2_data = tl.load(
        q2
        + num_token_offsets[:, None, None] * half_q_size
        + num_q_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    k1_data = tl.load(
        k1
        + num_token_offsets[:, None, None] * half_kv_size
        + num_kv_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    k2_data = tl.load(
        k2
        + num_token_offsets[:, None, None] * half_kv_size
        + num_kv_offsets[None, :, None] * half_dim
        + half_head_offsets[None, None, :],
    )
    v_data = tl.load(
        v
        + num_token_offsets[:, None, None] * kv_size
        + num_kv_offsets[None, :, None] * head_dim
        + head_offsets[None, None, :]
    )
    cos_data = tl.load(
        cos + num_token_offsets[:, None] * head_dim // 2 + half_head_offsets[None, :]
    )
    sin_data = tl.load(
        sin + num_token_offsets[:, None] * head_dim // 2 + half_head_offsets[None, :]
    )
    weight_data = tl.load(weight + half_head_offsets)

    for s in dl.parallel(0, 2, bind_sub_block=False):
        q1_sub_data = dl.extract_slice(
            q1_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, q_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        q2_sub_data = dl.extract_slice(
            q2_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, q_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        k1_sub_data = dl.extract_slice(
            k1_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, kv_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        k2_sub_data = dl.extract_slice(
            k2_data,
            (s * SUB_BLK, 0, 0),
            (SUB_BLK, kv_size // head_dim, head_dim // 2),
            (1, 1, 1),
        )
        cos_sub_data = dl.extract_slice(
            cos_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )
        sin_sub_data = dl.extract_slice(
            sin_data, (s * SUB_BLK, 0), (SUB_BLK, head_dim // 2), (1, 1)
        )

        # rms norm
        var_q1 = tl.sum(q1_sub_data * q1_sub_data, -1) / head_dim
        var_q2 = tl.sum(q2_sub_data * q2_sub_data, -1) / head_dim
        var_q = var_q1 + var_q2
        var_q = tl.expand_dims(var_q, -1)
        q1_sub_data = q1_sub_data * tl.math.rsqrt(var_q + 1e-5)
        q2_sub_data = q2_sub_data * tl.math.rsqrt(var_q + 1e-5)
        q1_sub_data = weight_data * q1_sub_data
        q2_sub_data = weight_data * q2_sub_data
        var_k1 = tl.sum(k1_sub_data * k1_sub_data, axis=-1) / head_dim
        var_k2 = tl.sum(k2_sub_data * k2_sub_data, axis=-1) / head_dim
        var_k = var_k1 + var_k2
        var_k = tl.expand_dims(var_k, -1)
        k1_sub_data = k1_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k2_sub_data = k2_sub_data * tl.math.rsqrt(var_k + 1e-5)
        k1_sub_data = weight_data * k1_sub_data
        k2_sub_data = weight_data * k2_sub_data

        # rotary embedding
        q1_rope, q2_rope = _compute_rotary_emb(
            q1_sub_data, q2_sub_data, cos_sub_data, sin_sub_data
        )
        k1_rope, k2_rope = _compute_rotary_emb(
            k1_sub_data, k2_sub_data, cos_sub_data, sin_sub_data
        )
        num_token_sub_offsets = tl.arange(0, SUB_BLK) + id * BLOCK_SIZE + s * SUB_BLK

        tl.store(
            q1_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q1_rope,
        )
        tl.store(
            q2_out
            + num_token_sub_offsets[:, None, None] * half_q_size
            + num_q_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            q2_rope,
        )
        tl.store(
            k1_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k1_rope,
        )
        tl.store(
            k2_out
            + num_token_sub_offsets[:, None, None] * half_kv_size
            + num_kv_offsets[None, :, None] * half_dim
            + half_head_offsets[None, None, :],
            k2_rope,
        )

    tl.store(
        v_out
        + num_token_offsets[:, None, None] * kv_size
        + num_kv_offsets[None, :, None] * head_dim
        + head_offsets[None, None, :],
        v_data,
    )