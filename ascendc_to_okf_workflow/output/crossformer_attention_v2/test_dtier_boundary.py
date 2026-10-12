"""v2 专项：D-tier 边界验证（head_dim>128，按 host CalcDBasicBlock 上浮 BLOCK_D）。

192KB UB 限制：head_dim≤128 时用 BLOCK_M=64/BLOCK_N=128/BLOCK_D=128；
head_dim>128 须降 BLOCK_N 腾 UB（v1.5 L2 边界）。这里验证 head_dim=256
（D-tier ALIGNED_256，BLOCK_D=256）在 BLOCK_N=64 下正确性成立。
"""
import os, torch, torch_npu, triton, triton.language as tl

torch.manual_seed(0)
dev = "npu:0"


@triton.jit
def _dot_new(a, b, IS_FP32: tl.constexpr):
    if IS_FP32:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _k(q_ptr, k_ptr, v_ptr, out_ptr, NW, G, C, HEAD_DIM, scale, STRIDE_T,
       IS_FP32: tl.constexpr, OUTPUT_DTYPE: tl.constexpr,
       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    num_m = (G + BLOCK_M - 1) // BLOCK_M
    NTOT = NW * num_m
    NUM_PROG = tl.num_programs(0)
    dcol = tl.arange(0, BLOCK_D)
    d_mask = dcol < HEAD_DIM
    for task in range(pid, NTOT, NUM_PROG):
        blk = task % num_m
        mrow = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        m_ok = mrow < G
        row = mrow
        foff = dcol
        q_off = q_ptr + row[:, None] * STRIDE_T + foff[None, :]
        q = tl.load(q_off, mask=m_ok[:, None] & d_mask[None, :], other=0.0)
        o_i = tl.load(q_off, mask=(mrow[:, None] > G), other=0.0).to(tl.float32)
        l_i = tl.load(q_ptr + row, mask=(mrow > G), other=0.0).to(tl.float32)
        m_i = tl.load(q_ptr + row, mask=(mrow > G), other=float("-inf")).to(tl.float32)
        for n0 in range(0, G, BLOCK_N):
            ncol = n0 + tl.arange(0, BLOCK_N)
            n_ok = ncol < G
            k_off = k_ptr + ncol[None, :] * STRIDE_T + foff[:, None]
            k = tl.load(k_off, mask=n_ok[None, :] & d_mask[:, None], other=0.0)
            v_off = v_ptr + ncol[:, None] * STRIDE_T + foff[None, :]
            v = tl.load(v_off, mask=n_ok[:, None] & d_mask[None, :], other=0.0)
            s = _dot_new(q, k, IS_FP32) * scale
            s = tl.where(n_ok[None, :], s, float("-inf"))
            m_cur = tl.max(s, 1)
            m_new = tl.maximum(m_i, m_cur)
            alpha = tl.exp((m_i - m_new).to(tl.float32))
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            o_i = o_i * alpha[:, None]
            o_i = o_i + _dot_new(p.to(OUTPUT_DTYPE), v, IS_FP32)
            m_i = m_new
        out = (o_i / tl.maximum(l_i, 1e-24)[:, None]).to(OUTPUT_DTYPE)
        tl.store(out_ptr + row[:, None] * STRIDE_T + foff[None, :], out,
                 mask=m_ok[:, None] & d_mask[None, :])


def run(x, scale, BLOCK_M, BLOCK_N, BLOCK_D):
    M, C = x.shape
    G = M
    head_dim = C
    out = torch.empty_like(x)
    grid = min(((G + BLOCK_M - 1) // BLOCK_M), 20)
    IS_FP32 = x.dtype == torch.float32
    tld = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[x.dtype]
    _k[(grid,)](x.reshape(-1, C), x.reshape(-1, C), x.reshape(-1, C), out,
               1, G, C, head_dim, float(scale), C,
               IS_FP32=IS_FP32, OUTPUT_DTYPE=tld, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
    return out


def golden(x, scale):
    import torch.nn.functional as F
    M, C = x.shape
    q = x * scale
    attn = q.float() @ x.float().T
    attn = F.softmax(attn, dim=-1)
    return (attn @ x.float()).to(x.dtype).reshape(M, C)


def rmse(a, b):
    return (a.float().reshape(-1) - b.float().reshape(-1)).abs().mean().item()


ok = True
tests = [
    # (name, shape, dtype, scale, blocks)
    ("hd256-dtier-f16-BN64", (257, 256), torch.float16, 0.063, (64, 64, 256)),   # D-tier 256, 非 2 幂 G
    ("hd256-dtier-f32-BN32", (129, 256), torch.float32, 0.063, (64, 32, 256)),   # fp32 大 D（CANN ③ 降 S1/S2）
    ("hd192-dtier-f16-BN64", (193, 192), torch.float16, 0.077, (64, 64, 256)),   # D-tier 192→BLOCK_D=256 mask
]
for name, shape, dtype, scale, blk in tests:
    try:
        x = torch.randn(shape, dtype=dtype, device=dev)
        out = run(x, scale, *blk)
        ref = golden(x, scale)
        r = rmse(out, ref)
        p = r <= 5e-3
        ok &= p
        print(f"[{'PASS' if p else 'FAIL'}] {name}: rmse={r:.3e} (tol 5e-3)")
    except Exception as e:
        ok = False
        print(f"[ERROR] {name}: {e}")
print("ALL PASS" if ok else "HAS FAILURE/ERROR")