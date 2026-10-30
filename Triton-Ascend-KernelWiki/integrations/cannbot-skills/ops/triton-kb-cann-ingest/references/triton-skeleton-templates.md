# Triton-Ascend skeleton templates

Author a Triton-style skeleton that captures *design intent*, not a compilable port. State in the file header: not compiled, not benchmarked, knowledge artifact only. Set wiki `reproducibility: pseudocode` (cannot compile) or `snippet` (compiles in isolation but unverified on NPU). Never claim it runs.

Common to all: `tl.program_id`, `tl.constexpr` meta, block/tile offsets, masked load/store, accumulation dtype, optional branches, layout/stride params.

## Elementwise

```python
@triton.jit
def eltwise_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # BF16 -> FP32 compute -> BF16 store (FP32 accumulation)
    y = <expr>(x.to(tl.float32)) if needs_fp32 else <expr>(x)
    tl.store(y_ptr + offsets, y, mask=mask)
# grid = ((n + BLOCK - 1) // BLOCK,)
```
Express: broadcast, vectorized load, dtype promotion, fused expression, tail mask.

## Reduction / Norm (RMSNorm example)

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, eps, N: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)              # one program per row
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + pid*N + offs, mask=mask, other=0.0).to(tl.float32)  # FP32 accum
    mean_sq = tl.sum(x*x, axis=0) / N
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = (x * rstd).to(x_ptr.dtype.element_ty())
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + pid*N + offs, x*rstd*w, mask=mask)
```
Express: reduce axis, FP32 accumulation, mean/var/rms, eps, affine, tail mask.

## Matmul

```python
@triton.jit
def matmul_kernel(a_ptr,b_ptr,c_ptr,M,N,K,stride_am,stride_ak,...,
                 BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    acc = tl.zeros((BLOCK_M,BLOCK_N), dtype=tl.float32)      # accumulate dtype
    for k in range(0, K, BLOCK_K):
        a = tl.load(...)   # [BLOCK_M,BLOCK_K], mask tail
        b = tl.load(...)   # [BLOCK_K,BLOCK_N], mask tail
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
    # optional bias / activation fusion here
    tl.store(c_ptr + ..., acc.to(out_dtype), mask=...)
```
Express: M/N/K block, K loop, accumulator dtype, optional bias/activation fusion, transpose via stride.

## Attention (FlashAttention forward, online softmax)

```python
@triton.jit
def attn_kernel(q_ptr,k_ptr,v_ptr,o_ptr,sm_scale,
                stride_q..., stride_kv..., ...,
                BLOCK_M:tl.constexpr, BLOCK_N:tl.constexpr, D:tl.constexpr,
                STAGE:tl.constexpr):       # causal / shift offset via STAGE
    start_m = tl.program_id(0)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running sum
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    q = tl.load(q_ptr + offs_q)            # [BLOCK_M, D]
    lo, hi = causal_lo_hi(start_m, STAGE)  # sparse/causal window
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(k_ptr + offs_kv(start_n), mask=...)   # [BLOCK_N, D]
        v = tl.load(v_ptr + offs_kv(start_n), mask=...)
        s = tl.dot(q, k.T) * sm_scale                      # QK^T, FP32
        s = apply_mask(s, start_n, ...)                    # atten/causal/padding
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s)                                       # online softmax
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)                          # rescale factor
        l_i = l_i*alpha + l_ij
        acc = acc*alpha[:, None] + tl.dot(p.to(v.dtype), v)  # PV, rescale + accumulate
        m_i = tl.maximum(m_i, m_ij)
    o = acc / l_i[:, None]
    tl.store(o_ptr + offs_o, o.to(out_dtype), mask=...)
```
Express: Q/K/V block load, QK, mask/causal/sparse index, scale, online softmax (running max/sum + rescale), PV, output writeback, optional lse/max/sum outputs for backward.

## Mapping notes to record alongside the skeleton

- `program_id` axes <- which CANN core/block id and split axis.
- `BLOCK_*` constexpr <- which CANN base block / tiling key branch (note value is source-reported).
- tail `mask` <- CANN tail-block / boundary handling.
- accumulation dtype (`tl.float32`) <- CANN FP32 intermediate / L0C accumulate.
- optional branches <- tiling-key dispatch (e.g. BF16 path, complex path, sparse path).
- workspace <- dropped or mapped to an intermediate `tl` tensor; never manual MTE.
