import json
import os
import math
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(
        self,
        k,
        w,
        u,
        g,
        initial_state,
        cu_seqlens,
        chunk_size,
        output_final_state,
    ):
        """
        CPU golden for chunk_gated_delta_rule_fwd_h.
        Layout:
            k:           [B, HK, T, K]
            w:           [B, HV, T, K]
            u:           [B, HV, T, V]
            g:           [B, HV, T]
            initial_state (if not None): [N, HV, K, V]
            cu_seqlens (if not None):    [N+1]
        Returns:
            states:      [B, HV, NT, K, V]
            v_new:       [B, HV, T, V]
            final_state: [N, HV, K, V] or None
        """
        dtype_ = k.dtype
        state_type_ = initial_state.dtype if initial_state is not None else torch.float32

        k = k.to(torch.float32)
        w = w.to(torch.float32)
        u = u.to(torch.float32)
        g = g.to(torch.float32)

        B, HK, T, K = k.shape
        HV, V = u.shape[1], u.shape[3]
        BT = chunk_size

        if cu_seqlens is None:
            N, NT = B, (T + BT - 1) // BT
            chunk_offsets = None
        else:
            lens = cu_seqlens[1:] - cu_seqlens[:-1]
            NT = int((lens + BT - 1) // BT).sum().item()
            N = len(cu_seqlens) - 1
            chunk_offsets = torch.cat(
                [cu_seqlens.new_tensor([0]), (lens + BT - 1) // BT]
            ).cumsum(-1)

        if initial_state is not None:
            initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float32)

        S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=torch.float32)
        v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=torch.float32)
        final_state = torch.zeros((N, HV, K, V), device=k.device, dtype=torch.float32)

        head_ratio = HV // HK
        for n in range(N):
            if cu_seqlens is None:
                bos = 0
                eos = T
                T_inner = T
                NT_inner = (T_inner + BT - 1) // BT
                boh = 0
            else:
                bos = int(cu_seqlens[n].item())
                eos = int(cu_seqlens[n + 1].item())
                T_inner = eos - bos
                NT_inner = (T_inner + BT - 1) // BT
                boh = int(chunk_offsets[n].item())

            for h in range(HV):
                for i in range(NT_inner):
                    actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                    k_sel = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel = w[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel = g[n, h, bos + i * BT : bos + i * BT + actual_len]

                    if initial_state is not None and i == 0:
                        S[n, h, boh + i] = initial_state[n, h]

                    v_new = u_sel - w_sel @ S[n, h, boh + i]
                    update = v_new * (g_sel[actual_len - 1] - g_sel).exp()[..., None]
                    new_state = (
                        S[n, h, boh + i] * g_sel[actual_len - 1].exp()
                        + k_sel.transpose(-1, -2) @ update
                    )

                    if i != NT_inner - 1:
                        S[n, h, boh + i + 1] = new_state
                    else:
                        final_state[n, h] = new_state

                    v_new_output[n, h, bos + i * BT : bos + i * BT + actual_len, :] = v_new

        S = S.to(dtype_)
        v_new_output = v_new_output.to(dtype_)
        final_state = final_state.to(state_type_) if output_final_state else None
        return S, v_new_output, final_state


def get_init_inputs():
    return []


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "chunk_gated_delta_fwd_h_mc.json")
    groups = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spec = json.loads(line)
            groups.append(_make_inputs(spec))
    return groups


def _make_inputs(spec):
    torch.manual_seed(42)
    B = spec["B"]
    kHead = spec["kHead"]
    vHead = spec["vHead"]
    T = spec["T"]
    K = spec["K"]
    V = spec["V"]
    chunk_size = spec["chunk_size"]
    is_varied_len = spec["is_varied_len"]
    dtype = getattr(torch, spec["dtype"])
    g_dtype = getattr(torch, spec["g_dtype"])
    state_dtype = getattr(torch, spec["state_dtype"])
    use_initial_state = spec.get("use_initial_state", True)
    output_final_state = spec.get("output_final_state", True)

    if is_varied_len:
        shape_batch = 1
        token_batch = B
    else:
        shape_batch = B
        token_batch = 1

    # Relaxed input range: mean-1 standard normal N(1,1) instead of N(0,1).
    # g keeps its decay semantics (logsigmoid cumsum, always negative).
    k = torch.randn(shape_batch, kHead, T, K, dtype=dtype, device="cpu") + 1
    w = torch.randn(shape_batch, vHead, T, K, dtype=dtype, device="cpu") + 1
    u = torch.randn(shape_batch, vHead, T, V, dtype=dtype, device="cpu") + 1
    g = _gen_decay(shape_batch, vHead, T, chunk_size, is_varied_len, dtype=g_dtype)

    if use_initial_state:
        if is_varied_len:
            initial_state = torch.randn(token_batch, vHead, K, V, dtype=state_dtype, device="cpu") + 1
        else:
            initial_state = torch.randn(B, vHead, K, V, dtype=state_dtype, device="cpu") + 1
    else:
        initial_state = None

    if is_varied_len:
        cu_seqlens = _gen_seqlen(T, token_batch)
    else:
        cu_seqlens = None

    return [k, w, u, g, initial_state, cu_seqlens, chunk_size, output_final_state]


def _gen_decay(B, HV, T, chunk_size, is_varied_len, dtype):
    # Follow the upstream repo's real usage:
    # g = F.logsigmoid(torch.randn(...))
    # then chunk_local_cumsum inside each chunk.
    g = torch.nn.functional.logsigmoid(torch.randn(B, HV, T, dtype=torch.float32))

    if is_varied_len:
        avg = T // B
        pos = 0
        for b in range(B):
            end = min(T, pos + avg) if b < B - 1 else T
            num_chunks = math.ceil((end - pos) / chunk_size)
            for c in range(num_chunks):
                cs = pos + c * chunk_size
                ce = min(end, cs + chunk_size)
                g[0, :, cs:ce] = g[0, :, cs:ce].cumsum(dim=-1)
            pos = end
    else:
        num_chunks = (T + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            cs = c * chunk_size
            ce = min(T, cs + chunk_size)
            g[:, :, cs:ce] = g[:, :, cs:ce].cumsum(dim=-1)

    return g.to(dtype)


def _gen_seqlen(T, batch):
    lens = []
    avg = T // batch
    remain = T
    for i in range(batch - 1):
        lo = max(1, avg // 2)
        hi = min(remain - (batch - i - 1), avg * 3 // 2)
        length = torch.randint(lo, hi + 1, (1,)).item()
        lens.append(length)
        remain -= length
    lens.append(remain)
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    return torch.tensor(cu, dtype=torch.int64)
