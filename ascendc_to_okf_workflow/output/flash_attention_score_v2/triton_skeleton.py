"""P2.5 Triton 骨架 — flash_attention_score（服务 level4 SelfAttention，v2 重新提取）

level4 SelfAttention 语义（definition.json reference golden 为准，seed=42 内建投影）：
    x[B,S,D]；q/k/v/out_proj 四段 nn.Linear(D,D,bias=False)（seed42）
    q = q_proj(x).view(B,S,H,hd).transpose(1,2)   # 依此类推 k/v
    scores = matmul(q.float(), k.float().T) / sqrt(hd)
    weights = softmax(scores, -1)
    out = matmul(weights, v.float()).to(dtype)
    return out_proj(out.transpose(1,2).contiguous().view(B,S,D))

本骨架只做**注意力核**（FAS 语义），投影由 wrapper 的 torch nn.Linear 承担——映射见
output/flash_attention_score_v2/mapping.md §0（接口差：投影归 wrapper / 前导 dot）。

kernel 结构 = 标准批多头 flash 注意力（FA设计介绍.md §1 步1~4），与
output/crossformer_attention/triton_skeleton.py 同构：**NW 视为 batch 数、G 视为序列长**，
每个 (batch, head) 独立对自身序列 S 做 Q·Kᵀ·V；head 切片按 (row, head*HEAD_DIM)。

backend_facts 契约引用（沿用 crossformer 骨架已验证修法）：
- persistent-core 分核（C-011）：grid=min(NW*H*cdiv(S,BLOCK_M), _num_aicore())，核内 grid-stride。
- 转置 Kᵀ/V 不用 tl.trans（C-018 规避）：以 (BLOCK_D,BLOCK_N) 镜像跨步 2D 仿射直读转置布局。
- flash 累加器 o_i/l_i/m_i **不用 tl.zeros/tl.full**（C-016）：all-False mask load 产零/−inf。
- 尾列 s 置 −inf（C-031 经验）：不参与 max/sum；稠密仅有自身长度边界，非 masked 稀疏。
- fp32 输入 tl.dot 用 input_precision="ieee"（C-029）；f16/bf16 用缺省 hf32。
- p·V 的 v 按 (BLOCK_N,BLOCK_D) key×feature 加载（C-030，收缩维=key，左手分量随输出 layout）。
- 无 %（C-001，t−(t//b)*b 化解，仅 wrapper）；无 or 链（C-023①）；无 num_stages/num_warps
  （C-006，本后端不生效）；无 autotune（C-007，防 do_bench 噪声）。
- 尺寸 mask 比较：load/store 保持整型 mask（C-002）；d<HEAD_DIM mask。
- HEAD_DIM 非 2 幂 BLOCK_D 取上浮 2 幂 + mask。
- scale/grid=../下 → `min(NW*H*cdiv(S,BLOCK_M), _num_aicore())`。
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # C-019 环境红线

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_new(a, b, IS_FP32: tl.constexpr):
    """fp32 输入必须 ieee（tf32 达不到容差，C-029）。"""
    if IS_FP32:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _selfattn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    NW, G, C, HEAD_DIM, N_HEADS,
    scale,
    STRIDE_T,
    IS_FP32: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """标准批多头 flash 注意力。

    q/k/v 布局 (NW*G, C)，C = N_HEADS*HEAD_DIM，token 行主序、特征列连续。
    NW= batch 数（本目标中独立序列数），G= 序列长 S（每 batch 内独立注意）。
    每个 (w, head) 对自身 G 长序列做 flash；head 切片位列偏移 w*G + head*HEAD_DIM。
    输出与输入同布局（token 行、head 拼接特征）。
    """
    pid = tl.program_id(0)

    num_m = (G + BLOCK_M - 1) // BLOCK_M           # 每 (window,head) query 块数
    NTOT = NW * N_HEADS * num_m
    NUM_PROG = tl.num_programs(0)

    dcol = tl.arange(0, BLOCK_D)
    d_mask = dcol < HEAD_DIM

    for task in range(pid, NTOT, NUM_PROG):        # persistent grid-stride（C-011）
        blk = task % num_m                          # query 块(序列内)
        rem = task // num_m
        head = rem % N_HEADS
        w = rem // N_HEADS

        w0 = w * G
        feat = head * HEAD_DIM
        foff = feat + dcol                          # 本 head 特征列切片

        mrow = blk * BLOCK_M + tl.arange(0, BLOCK_M)  # 序列内行(wind 内 token)
        m_ok = mrow < G
        row = w0 + mrow                             # 全局 token 行

        # ---- Q tile (BLOCK_M, BLOCK_D)，镜像跨步直读 head 切片 ----
        q_off = q_ptr + row[:, None] * STRIDE_T + foff[None, :]
        q = tl.load(q_off, mask=m_ok[:, None] & d_mask[None, :], other=0.0)

        # ---- flash 累加器初值（C-016：all-False mask load，勿 tl.zeros）----
        rowv = w0 + tl.arange(0, BLOCK_M)
        o_i = tl.load(q_off, mask=(mrow[:, None] > G), other=0.0).to(tl.float32)
        l_i = tl.load(q_ptr + rowv, mask=(mrow > G), other=0.0).to(tl.float32)
        m_i = tl.load(q_ptr + rowv, mask=(mrow > G), other=float("-inf")).to(tl.float32)

        # ---- flash 外循环 over key 块（在线 softmax，复刻 FA §1 步1~4）----
        for n0 in range(0, G, BLOCK_N):
            ncol = n0 + tl.arange(0, BLOCK_N)
            n_ok = ncol < G
            nrow = w0 + ncol

            # Kᵀ (BLOCK_D, BLOCK_N) 镜像直读转置
            k_off = k_ptr + nrow[None, :] * STRIDE_T + foff[:, None]
            k = tl.load(k_off, mask=n_ok[None, :] & d_mask[:, None], other=0.0)
            # V (BLOCK_N, BLOCK_D) = (key, feature)：p·V 收缩维=key（C-030）
            v_off = v_ptr + nrow[:, None] * STRIDE_T + foff[None, :]
            v = tl.load(v_off, mask=n_ok[:, None] & d_mask[None, :], other=0.0)

            s = _dot_new(q, k, IS_FP32)   # (BLOCK_M, BLOCK_N) fp32; f16/bf16 缺省 hf32, fp32 ieee
            s = s * scale

            # 尾列（ncol>=G）置于 −inf：不参与 max/sum（flash 关键）
            s = tl.where(n_ok[None, :], s, float("-inf"))

            # 在线 softmax
            m_cur = tl.max(s, 1)                              # (BLOCK_M,)
            m_new = tl.maximum(m_i, m_cur)
            alpha = tl.exp((m_i - m_new).to(tl.float32))
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            o_i = o_i * alpha[:, None]
            p_cast = p.to(OUTPUT_DTYPE)
            o_i = o_i + _dot_new(p_cast, v, IS_FP32)
            m_i = m_new

        # ---- 归一 + 写回 (NW*G, C) ----
        out = (o_i / tl.maximum(l_i, 1e-24)[:, None]).to(OUTPUT_DTYPE)
        out_off = out_ptr + row[:, None] * STRIDE_T + foff[None, :]
        tl.store(out_off, out, mask=m_ok[:, None] & d_mask[None, :])


def _num_aicore() -> int:
    try:
        import torch_npu
        if not torch.npu.is_available():
            return 20
        import subprocess
        subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        # 保守取 20（910B3 常见 Cube 核数，C-011；910 亦可用，铺不满属固有上限）
        return 20
    except Exception:
        return 20


def _proj(x, dev, dtype, dim):
    """seed=42 复刻 golden 的四段投影权重（CPU 生成后搬到设备，保证两路径一致）。"""
    torch.manual_seed(42)
    import torch.nn as nn
    W = [nn.Linear(dim, dim, bias=False) for _ in range(4)]
    return [w.weight.data.to(dev).to(dtype) for w in W]


def self_attention(x, n_heads):
    """level4 SelfAttention wrapper：投影(torch) + flash 注意力 kernel + out_proj(torch)。

    x  (B,S,D)；n_heads。返回 (B,S,D)（out_proj 后最终输出）。
    投影权重由 seed=42 生成，与 golden 一致（映射 §0 接口差）。
    """
    dev = x.device
    dtype = x.dtype
    B, S, D = x.shape
    head_dim = D // n_heads

    # D-tier 自适应（对齐 crossformer_v2 实证，C-032）：Triton BLOCK_D 必须 2 幂，
    # 取 ≥head_dim 的最小 2 幂；BLOCK_D>128 在 BLOCK_M=64 触发 192KB UB 溢出 ⇒
    # 自动降 BLOCK_M=32/BLOCK_N=64 腾 UB（p25 逃逸验证 hd256 rmse≈5e-8）。
    bd = 128
    while bd < head_dim:
        bd *= 2
    BLOCK_D, BM, BN = (bd, 32, 64) if bd > 128 else (bd, 64, 128)

    Wq, Wk, Wv, Wo = _proj(x, dev, dtype, D)
    q = x.reshape(B * S, D) @ Wq    # (B*S, D) token 行主序
    k = x.reshape(B * S, D) @ Wk
    v = x.reshape(B * S, D) @ Wv

    out = torch.empty_like(x).reshape(B * S, D)
    grid = min(B * n_heads * ((S + BM - 1) // BM), _num_aicore())
    _selfattn_kernel[(grid,)](
        q.contiguous(), k.contiguous(), v.contiguous(), out,
        B, S, D, head_dim, n_heads,
        float(head_dim ** -0.5), D,
        IS_FP32=(dtype == torch.float32), OUTPUT_DTYPE=tl.float16 if dtype == torch.float16
                else (tl.bfloat16 if dtype == torch.bfloat16 else tl.float32),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D,
    )
    return (out.reshape(B, S, D) @ Wo).to(dtype)


def golden_run(x, n_heads):
    """逐字复刻 level4 definition.json reference（纯 torch golden）。"""
    import math
    import torch.nn.functional as F
    B, S, D = x.shape
    hd = D // n_heads
    dev, dtype = x.device, x.dtype
    Wq, Wk, Wv, Wo = _proj(x, dev, dtype, D)
    q = (x.reshape(B * S, D) @ Wq).view(B, S, n_heads, hd).transpose(1, 2)
    k = (x.reshape(B * S, D) @ Wk).view(B, S, n_heads, hd).transpose(1, 2)
    v = (x.reshape(B * S, D) @ Wv).view(B, S, n_heads, hd).transpose(1, 2)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(hd)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v.float()).to(dtype=dtype)
    output = output.transpose(1, 2).contiguous().view(B, S, D)
    return (output @ Wo).to(dtype)


# 模块默认仅供 head_dim≤128；实际取值由 wrapper 内 D-tier 计算（C-032：hd>128 降 BM32/BN64）。
BLOCK_M = 64
BLOCK_N = 128
BLOCK_D = 128


def _rmse(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    return (a - b).abs().mean().item()


def _check(name, out, ref, tol=2e-2):
    outfit, refit = out.reshape(-1), ref.reshape(-1)
    ok = outfit.shape == refit.shape and _rmse(outfit, refit) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: rmse={_rmse(outfit, refit):.3e} (tol {tol})")
    return ok


if __name__ == "__main__":
    import torch_npu  # noqa: 注册 NPU 设备（无此回退 cpu 报指针错误）
    dev = "npu:0"
    ok_all = True

    def make(shape, dt):
        return torch.randn(shape, dtype=dt, device=dev)

    cases = [
        # (name, x, n_heads)
        ("bigS-001-f16-S512", make((1, 512, 4096), torch.float16), 32),    # hd 128
        ("bigS-004-f16-S4096", make((1, 4096, 4096), torch.float16), 32),  # hd 128, 大 S flash 多块
        ("bigS-003-bf16-S2048", make((1, 2048, 4096), torch.bfloat16), 32),
        ("midS-f16-B2-S512", make((2, 512, 2048), torch.float16), 16),     # hd 128, B=2
        ("fp32-f32-S256", make((1, 256, 1024), torch.float32), 8),         # hd 128, fp32 ieee
        ("hd256-bf16-S512", make((1, 512, 8192), torch.bfloat16), 32),     # hd 256, BLOCK_D=256
        ("smallS-f16-S64", make((2, 64, 512), torch.float16), 8),          # hd 64, 小 S
    ]
    for name, x, nh in cases:
        try:
            out = self_attention(x, nh)
            ref = golden_run(x, nh)
            ok_all &= _check(name, out, ref)
        except Exception as e:
            ok_all = False
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
    print("ALL_PASS" if ok_all else "SOME_FAIL")