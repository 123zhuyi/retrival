"""P2 Triton 骨架 — flash_attention_score（level4 目标 SelfAttention，主目标 8_SelfAttention 大 S）

level4 语义（definition.json golden 为准）：
    q,k,v = x·Wqᵀ, x·Wkᵀ, x·Wvᵀ                      # 4 段投影（seed=42 权重；CANN 参考不含，host GEMM）
    scores = QKᵀ·scale, scale=1/√head_dim             # per-head，fp32 计算域
    w = softmax(scores, dim=-1)
    z = w·V  →  (B,S,D)
    out = z·Woᵀ

设计决策与 backend_facts 取证对照（详见 mapping.md，均为 big-S 主路径）：
- **注意力核 = 1 个 Triton flash-attn 前向 kernel**（`_sa_attn_fwd_kernel`），**S1S2 分块 + 在线 softmax**
  （主目标 100% S>64）：任务轴 = B·H·cdiv(S,BLOCK_M)，key/S2 内环 `for n0 in range(0,S,BLOCK_N)`
  running max + α 补偿 → 与 CANN `_s1s2_bn2gs1.h` 同构；小 S 副分支 BLOCK_N≥S 时内环仅 1 迭代 = 单块
  full-softmax（mapping §0 论证 3）。
- **4 段投影 = host fp32 matmul**（mapping §0 论证 4）：CANN 参考（flash_attention_score）是纯
  注意力核无投影逻辑；投影是平凡 GEMM + 跨 head 参数层。smoke 两路（golden/Triton）共享同一份
  seed=42 权重 → 精确隔离注意力核正确性。
- fp32 计算域：q/k/v promote fp32 → dot ieee（`input_precision="ieee"`，对全 dtype，operator_class_patterns
  §4.2 [EXP]，tf32 达不到容差）；末段一次 RTNE 回写。level4 golden 显式 `.float()`（定义如此）。
- 无 causal/pse/mask/dropout/varlen（level4 恒规整 BSH + 全 softmax）。
- 无 %（C-001）；loop-carried 累加器初值禁 `tl.zeros`（C-016 → 用全 False mask load 产零/产 -inf + 显式
  `.to(fp32)`）；K 以 [HD,N] 镜像、V 以 [N,HD] 2D 仿射 load（**非 gather**，C-018 合规；V 定向 C-030）；
  key 越界列 -inf（C-031）；store/load mask 整型比较（C-002）；无 num_stages/num_warps（C-006）；
  无 autotune（C-007）；int64 仅标量块基址（dtype_support 4e）。
- grid=min(B·H·cdiv(S,BLOCK_M), np)，np 运行时查询禁写死（hardware_platforms §3 / C-011 / C-026）。
- UB 预算（950 248KB 可用 [SRC]）：head_dim>128(BLOCK_HD=256) 需降 BLOCK_M/N=32，否则贴顶（mapping §2）。
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # C-019 环境红线

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401  注册 npu 设备后端（triton-ascend 经此路由）
except ImportError:
    pass


@triton.jit
def _sa_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    NH, S, HD, scale,                 # runtime：NH=B·H；S=S_q=S_k；HD=head_dim；scale=1/√HD
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HD: tl.constexpr,
):
    """S1S2 分块在线 softmax 的 flash 前向（主，big-S）；BLOCK_N≥S 时退化为单块 full-softmax（副，small-S）。

    任务空间 = NH · cdiv(S, BLOCK_M)，persistent grid-stride（C-011）。key/S2 内环：
        QKᵀ·scale → s；running max m；α=exp(m_old−m_new)；p=exp(s−m_new)；l=l·α+sum(p)；o=o·α+p·V
    每个任务独立持其 S1(query) 块输出行，无跨任务部分和/原子。
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(S, BLOCK_M)
    NTOT = NH * num_m
    dcol = tl.arange(0, BLOCK_HD)
    d_ok = dcol < HD
    for task in range(pid, NTOT, tl.num_programs(0)):
        bh = task // num_m                      # 免 %（C-001）
        mblk = task - bh * num_m
        base = bh.to(tl.int64) * S               # head 起始=行偏移；后续每处 load 的 *HD 已还原 HD 因子（勿再乘 HD）
        mrow = mblk * BLOCK_M + tl.arange(0, BLOCK_M)
        mm = mrow < S
        row = base + mrow
        qm = mm[:, None] & d_ok[None, :]
        q = tl.load(q_ptr + row[:, None] * HD + dcol[None, :], mask=qm, other=0.0).to(tl.float32)
        # loop-carried 累加器初值禁 tl.zeros（C-016）：全 False mask load 产零 / 产 -inf + 显式 .to(fp32)
        o_a = tl.load(o_ptr + row[:, None] * HD + dcol[None, :], mask=(mrow[:, None] > S), other=0.0).to(tl.float32)
        l_i = tl.load(q_ptr + row, mask=(mrow > S), other=0.0).to(tl.float32)
        m_i = tl.load(q_ptr + row, mask=(mrow > S), other=float("-inf")).to(tl.float32)
        for n0 in range(0, S, BLOCK_N):         # key/S2 内环（在线 softmax）
            ncol = n0 + tl.arange(0, BLOCK_N)
            n_ok = ncol < S
            nrow = base + ncol
            kt = tl.load(k_ptr + nrow[None, :] * HD + dcol[:, None], mask=n_ok[None, :] & d_ok[:, None], other=0.0).to(tl.float32)  # (HD,N) C-029
            v = tl.load(v_ptr + nrow[:, None] * HD + dcol[None, :], mask=n_ok[:, None] & d_ok[None, :], other=0.0).to(tl.float32)   # (N,HD) C-030
            s = tl.dot(q, kt, input_precision="ieee") * scale                       # (M,N)
            s = tl.where(n_ok[None, :], s, float("-inf"))                           # C-031 key 尾列 −inf
            m_c = tl.max(s, 1)
            m_n = tl.maximum(m_i, m_c)
            alpha = tl.exp((m_i - m_n).to(tl.float32))
            p = tl.exp(s - m_n[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            o_a = o_a * alpha[:, None] + tl.dot(p, v, input_precision="ieee")        # (M,N)@(N,HD)
            m_i = m_n
        out = (o_a / tl.maximum(l_i, 1e-24)[:, None])
        tl.store(o_ptr + row[:, None] * HD + dcol[None, :], out, mask=qm)           # 一次 RTNE 回写


def _num_aicore() -> int:
    """运行时查询核数；禁止写死（hardware_platforms §3）。"""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(0)
        for k in ("num_aicore", "num_core", "num_core_count"):
            if props.get(k):
                return props[k]
        aiv = props.get("num_aivcore")
        if aiv:
            return aiv
    except Exception:
        pass
    return 20


# ---- 权重缓存（与 level4 Model._layers 同构：seed=42、按 (D,H,device,dtype) 缓存）----
_LAYER_CACHE = {}


def _get_weights(x, n_heads):
    D = x.shape[-1]
    key = (D, n_heads, x.device, x.dtype)
    if key not in _LAYER_CACHE:
        torch.manual_seed(42)
        ws = []
        for _ in range(4):                    # Wq, Wk, Wv, Wo（bias=False, nn.Linear 默认 kaiming_uniform init）
            w = torch.empty(D, D, device=x.device, dtype=x.dtype)
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            ws.append(w)
        _LAYER_CACHE.clear()
        _LAYER_CACHE[key] = ws
    return _LAYER_CACHE[key]


def _project_qkv(x, ws, n_heads):
    """4 段投影中取前 3（q/k/v），host fp32（mapping §0 论证 4）。"""
    xf = x.float()
    return [torch.matmul(xf, w.float().t()) for w in ws[:3]], ws[3].float().t()  # q,k,v + Woᵀ


def _sa_attn_fwd(q, k, v, n_heads):
    """Triton 注意力核。输入 q/k/v [B,S,D] fp32 → 返回 z [B,S,D] fp32。

    S1S2 分块在线 softmax（big-S 主路径）；小 S 副分支退化为单 K 块 full-softmax。
    """
    B, S, D = q.shape
    HD = D // n_heads
    assert D % n_heads == 0 and HD > 0
    for t in (q, k, v):
        assert t.is_contiguous()

    def heads(t):
        # P2.5 关键后端发现：`view→transpose→reshape` 在 B=1 时返回**非连续**交错视图，
        # kernel 平坦行主序寻址会静默错值 → 必须 `.contiguous()`（详见 mapping §5 连续性契约）。
        return t.view(B, S, n_heads, HD).transpose(1, 2).reshape(B * n_heads, S, HD).contiguous()  # [B*H,S,HD]

    qh, kh, vh = heads(q), heads(k), heads(v)
    o = torch.empty((B * n_heads, S, HD), device=q.device, dtype=torch.float32)
    block_hd = triton.next_power_of_2(HD)
    # 大 head_dim(BLOCK_HD≥256) 时降 BM/BN=32 控 UB峰值(mapping §2)；否则 64。
    bm = 32 if block_hd >= 256 else 64
    bn = bm
    num_m = (S + bm - 1) // bm
    np = max(_num_aicore(), 1)
    grid = (min(B * n_heads * num_m, np),)
    scale = 1.0 / math.sqrt(HD)
    _sa_attn_fwd_kernel[grid](
        qh, kh, vh, o, B * n_heads, S, HD, scale,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_HD=block_hd)
    z = o.view(B, n_heads, S, HD).transpose(1, 2).reshape(B, S, D)
    return z


def run(x, n_heads, weights=None):
    """level4 接口：run(x, n_heads) → out。投影 host + Triton 注意力核。"""
    ws = weights if weights is not None else _get_weights(x, n_heads)
    qkv, Wo_t = _project_qkv(x, ws, n_heads)
    z = _sa_attn_fwd(qkv[0], qkv[1], qkv[2], n_heads)
    out = torch.matmul(z, Wo_t)
    return out.to(x.dtype)


# ---- torch golden（definition.json run() 逐字移植）----
def golden_run(x, n_heads, weights=None):
    ws = weights if weights is not None else _get_weights(x, n_heads)
    q, k, v = [torch.matmul(x.float(), w.float().t()) for w in ws[:3]]
    B, S, D = q.shape
    HD = D // n_heads
    scale = 1.0 / math.sqrt(HD)
    qh = q.view(B, S, n_heads, HD).transpose(1, 2).float()
    kh = k.view(B, S, n_heads, HD).transpose(1, 2).float()
    vh = v.view(B, S, n_heads, HD).transpose(1, 2).float()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
    w = F.softmax(scores, dim=-1)
    z = torch.matmul(w, vh).transpose(1, 2).contiguous().view(B, S, D)
    out = torch.matmul(z, ws[3].float().t())
    return out.to(x.dtype)


# ---- 在线-softmax 纯 torch 参考（CPU 可跑，验证 running max/α 数学，与 golden 对照）----
def _flash_fwd_tiled(qh, kh, vh, scale, BM=64, BN=64):
    """qh/kh/vh: [BH, S, HD] fp32。S1S2 分块在线 softmax（torch，精确切片无 padding）。"""
    BH, S, HD = qh.shape
    o = torch.empty(BH, S, HD, dtype=torch.float32)
    for i0 in range(0, S, BM):
        nrow = min(BM, S - i0)
        Q = qh[:, i0:i0 + BM, :]
        m = torch.full((BH, nrow), float("-inf"))
        l = torch.zeros(BH, nrow)
        oacc = torch.zeros(BH, nrow, HD)
        for j0 in range(0, S, BN):
            ncol = min(BN, S - j0)
            k = kh[:, j0:j0 + ncol, :]
            v = vh[:, j0:j0 + ncol, :]
            s = torch.matmul(Q, k.transpose(1, 2)) * scale                       # (BH,nrow,ncol)
            mnew = torch.maximum(m, s.max(dim=-1).values)
            alpha = torch.exp((m - mnew).to(torch.float32))
            p = torch.exp(s - mnew.unsqueeze(-1))
            l = l * alpha + p.sum(dim=-1)
            oacc = oacc * alpha.unsqueeze(-1) + torch.matmul(p, v)
            m = mnew
        o[:, i0:i0 + nrow, :] = oacc / l.clamp(min=1e-24).unsqueeze(-1)
    return o


def _selfcheck_cpu():
    """CPU 在线-softmax 数学自检（不依赖 Triton/NPU）：tiled flash vs F.softmax full golden。"""
    print("=== CPU 在线-softmax 数学自检 (S1S2 tiled vs full golden) ===")
    allok = True
    for B, S, Hh, HD in [(1, 511, 3, 64), (2, 129, 4, 96), (1, 260, 5, 128),
                         (1, 65, 3, 256), (2, 70, 4, 32), (1, 1000, 6, 64)]:
        torch.manual_seed(0)
        q = torch.randn(B * Hh, S, HD)
        k = torch.randn(B * Hh, S, HD)
        v = torch.randn(B * Hh, S, HD)
        scale = 1.0 / math.sqrt(HD)
        zfull = torch.matmul(F.softmax(torch.matmul(q, k.transpose(1, 2)) * scale, dim=-1), v)
        ztiled = _flash_fwd_tiled(q, k, v, scale, BM=32 if HD >= 256 else 64, BN=64)
        err = (ztiled - zfull).abs().max().item()
        ok = err < 1e-4
        allok &= ok
        print(f"  B{B} S{S} H{Hh} HD{HD}: max_err={err:.2e} {'OK' if ok else 'FAIL'}")
    print("CPU MATH ALL OK" if allok else "CPU MATH FAIL")
    return allok


# ---- smoke（NPU 上用真机/模拟器；覆盖三 dtype × head_dim 档 ± 大 S 代表面）----
def _mk_x(shape, dtype, device):
    c = torch.Generator(device="cpu")
    c.manual_seed(42)
    return torch.randn(shape, device="cpu", dtype=torch.float32).to(dtype).to(device)


def _check(out, ref, dtype, case):
    a, b = out.float(), ref.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    if dtype == torch.float32:
        tol_el = 1e-3
        bad_ratio = (diff > tol_el).float().mean().item()
        ok = (max_abs <= 1e-3) and (bad_ratio <= 1e-3)
    else:
        thr = torch.maximum(0.005 * torch.maximum(a.abs(), b.abs()), torch.tensor(2.5e-5, device=a.device))
        fail = diff > thr
        bad_ratio = fail.float().mean().item()
        ok = (bad_ratio <= 0.005) and (max_abs <= 0.1)
    print(f"    [{case}] dtype={str(dtype):9} max_abs={max_abs:.3e} bad_ratio={bad_ratio:.4f} -> {'OK' if ok else 'FAIL'}")
    return ok


def smoke(device=None, cases=None):
    abridge = device is None
    if device is None:
        device = "npu" if torch.npu.is_available() else "cpu"
    dm = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    sample = cases or [
        # 主目标 8_SelfAttention 大 S 代表面（S>512 需在线 softmax）
        dict(shape=[1, 512, 4096], dtype="float16", n_heads=32),    # HD 128
        dict(shape=[1, 1024, 3072], dtype="bfloat16", n_heads=32),  # HD 96
        dict(shape=[1, 828, 256], dtype="float32", n_heads=4),      # HD 64
        dict(shape=[2, 701, 288], dtype="float16", n_heads=6),      # HD 48
        dict(shape=[1, 2762, 4096], dtype="float32", n_heads=32),   # make d_model divisible: HD 128
    ]
    print(f"=== self-attention skeleton smoke on {device} ===")
    allok = True
    for c in sample:
        shape, dt, H = c["shape"], dm[c["dtype"]], c["n_heads"]
        assert shape[-1] % H == 0, f"shape {shape} not divisible by n_heads {H}"
        x = _mk_x(shape, dt, device)
        out = run(x, H)
        ref = golden_run(x, H)
        allok &= _check(out, ref, dt, f"S{shape[1]} D{shape[2]} H{H}")
    print("ALL OK" if allok else "SOME FAILED")
    return allok


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else None
    cpu_ok = _selfcheck_cpu()
    if dev == "cpu":
        sys.exit(0 if cpu_ok else 1)
    npu_ok = smoke(device=dev)
    sys.exit(0 if (cpu_ok and npu_ok) else 1)