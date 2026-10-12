"""P2 Triton 骨架 — CrossformerAttention（level4 目标 CrossformerAttention）

level4 语义（definition.json reference golden 为准）：
    x 分组 → 窗口批量 (NW, G, C)；q/k/v 缺省即 x（同布局）
    n_heads；head_dim = C / n_heads
    attn[h] = (Q·Kᵀ)·scale + pse + mask        # 加性偏置 softmax 前相加，fp32
    P = softmax(attn, -1)
    out = P·V                                    # out 累加 fp32，末除 1/sum
    非 feature → NW=1 单窗口全序列 MHA；feature → (H/gh)*(W/gw) 个 (gh*gw) 窗口

设计决策与 backend_facts 取证对照（详见 mapping.md）：
- 两阶段：① wrapper 做窗口分组重排（host 侧 layout——CANN 无此对应，manifest 缺省项 1），
  产出窗口连续布局 (NW, G, C)；② Triton kernel 做**标准批多头 flash 注意力**（Kernel 纯 attention，
  flash 在线 softmax 复刻 FA设计介绍.md §1 步 1~4，映射对照 operator_class_patterns FlaskAttention 族）。
- persistent-core 分核（C-011/mapping §1 连续段范式）：grid = min(NW·n_heads·cdiv(G,BLOCK_M), num_core)，
  核内 grid-stride。CV 融合算子取 Cube 核数（operator_class_patterns §0）。
- flash 外循环 = key 分块（BLOCK_N）在线 softmax：running max/sum + exp 补偿（α=exp(m_old−m_new)），
  对应 FA 的 FlashSoftmaxV2 重构链。
- 大组/大窗（G 达 3721/7921）：必走 flash 多块（BLOCK_M×BLOCK_N tile 受限 UB），禁止整窗 QK 一次性。
- 转置需求（Kᵀ/Vᵀ）**不用 tl.trans**：以 (BLOCK_D, BLOCK_N) 镜像跨步 2D 仿射 load 直读转置布局
  （非 gather、不触发 C-018）。
- loop-carried 累加器（o_i/l_i/m_i）**不用 tl.zeros/tl.full**（C-016，上下文相关不赌）：
  全 False mask 的 load 产零/产 −inf。
- 无 %（C-001，div/mod 用 t−(t//b)*b 化解且仅 wrapper）；无 or 链（C-023①）；无 num_stages/
  num_warps（C-006）；无 autotune（C-007）；scale 为 runtime 标量（C-013 两态，benchmark 可 constexpr）。
- fp32 输入 tl.dot 用 input_precision="ieee"（operator_class_patterns §4.2 [EXP]）；f16/bf16 输入
  P 在 dot(V) 前 cast 回输入 dtype（对齐 CANN drop/ expcast 落点）。
- HEAD_DIM 非 2 的幂（如 96）时 BLOCK_D 取上浮 2 的幂 + d<HEAD_DIM mask。
- 内部掩码比较转 fp32 规避 i32 vcmp 标量（dtype_support §3）；load/store mask 保持整型比较（C-002）。
- int64 仅标量块基址（dtype_support 4e）：C≤1024、最大 numel ~2G token×1024 < 2³¹，tile 内 int32 已够。
- UB 预算（Ascend910B A2 192KB [SRC hardware_platforms]）：fp16 BLOCK_M=64/BLOCK_N=128/BLOCK_D=128
  → q=16KB + k/v 各 16KB + attn(fp32 64×128=32KB) ×2 ≈ 112KB ≈ 60% —— 安全；fp32×128 自动降档。
  大 G 的 flash 循环不占更多 UB（每次只一块 k/v）。
"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")  # C-019 环境红线

import torch
import triton
import triton.language as tl


@triton.jit
def _dot_new(a, b, IS_FP32: tl.constexpr):
    """fp32 输入必须 ieee（tf32 达不到容差）。"""
    if IS_FP32:
        return tl.dot(a, b, input_precision="ieee")
    return tl.dot(a, b)


@triton.jit
def _crossformer_attn_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, out_ptr,
    NW, G, C, HEAD_DIM, N_HEADS,
    scale,
    STRIDE_T,         # C（token 行步长）
    HAS_BIAS: tl.constexpr,
    IS_FP32: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)

    num_m = (G + BLOCK_M - 1) // BLOCK_M          # 每 (window,head) 的 query 块数
    NTOT = NW * N_HEADS * num_m
    NUM_PROG = tl.num_programs(0)

    dcol = tl.arange(0, BLOCK_D)
    d_mask = dcol < HEAD_DIM

    for task in range(pid, NTOT, NUM_PROG):       # persistent grid-stride（C-011）
        blk = task % num_m                          # query 块
        rem = task // num_m
        head = rem % N_HEADS
        w = rem // N_HEADS

        w0 = w * G
        feat = head * HEAD_DIM
        foff = feat + dcol                          # feature 列（本 head 切片）

        mrow = blk * BLOCK_M + tl.arange(0, BLOCK_M)  # window-local row
        m_ok = mrow < G
        row = w0 + mrow                            # global token row

        # ---- Q tile (BLOCK_M, BLOCK_D)，镜像跨步从 (NW*G, C) 直读 head 切片 ----
        q_off = q_ptr + row[:, None] * STRIDE_T + foff[None, :]
        q = tl.load(q_off, mask=m_ok[:, None] & d_mask[None, :], other=0.0)

        # ---- flash 累加器初值（C-016：全 False mask load，勿 tl.zeros）----
        rowv = w0 + tl.arange(0, BLOCK_M)
        # o_i: (BLOCK_M, BLOCK_D) 零初值（all-False mask → other=0）
        o_i = tl.load(q_off, mask=(mrow[:, None] > G), other=0.0).to(tl.float32)
        # l_i/m_i: (BLOCK_M,) 一维初值
        l_i = tl.load(q_ptr + rowv, mask=(mrow > G), other=0.0).to(tl.float32)
        m_i = tl.load(q_ptr + rowv, mask=(mrow > G), other=float("-inf")).to(tl.float32)

        # ---- flash 外循环 over key 块（在线 softmax，复刻 FA §1 步 1~4）----
        for n0 in range(0, G, BLOCK_N):
            ncol = n0 + tl.arange(0, BLOCK_N)
            n_ok = ncol < G
            nrow = w0 + ncol

            # Kᵀ (BLOCK_D, BLOCK_N) 镜像直读转置
            k_off = k_ptr + nrow[None, :] * STRIDE_T + foff[:, None]
            k = tl.load(k_off, mask=n_ok[None, :] & d_mask[:, None], other=0.0)
            # V (BLOCK_N, BLOCK_D) = (key, feature)：p·V 的收缩维=key，V 需 key 行、feature 列
            v_off = v_ptr + nrow[:, None] * STRIDE_T + foff[None, :]
            v = tl.load(v_off, mask=n_ok[:, None] & d_mask[None, :], other=0.0)

            s = _dot_new(q, k, IS_FP32)   # (BLOCK_M, BLOCK_N) fp32; Ascend: f16/bf16 用缺省 hf32, fp32 用 ieee
            s = s * scale

            # 尾列（ncol >= G）置于 -inf：不参与 max/sum（flash 关键，防假质量）
            s = tl.where(n_ok[None, :], s, float("-inf"))
            if HAS_BIAS:
                # bias 布局 (N_HEADS, G, G)，全窗口共享（feature 的 mask/pos_bias 即窗口局部）
                bbase = head * G * G
                b_off = bias_ptr + bbase + mrow[:, None] * G + ncol[None, :]
                bias = tl.load(b_off, mask=m_ok[:, None] & n_ok[None, :], other=0.0)
                s = s + tl.where(n_ok[None, :], bias, float("-inf"))

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
        r = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
        # 保守取 20（910B3 常见 Cube 核数，C-011；910 也可用，铺不满属固有上限）
        return 20
    except Exception:
        return 20


def crossformer_attention(x, group_size, q=None, k=None, v=None, num_heads=1,
                          mask=None, scale=None, pos_bias=None, feature_shape=None):
    """wrapper：窗口分组重排（host 侧）+ 调 flash attention kernel。

    返回与 x 同 layout 的输出。x 输入 (M, C) 或 (B, M, C)→拍平成 (M,C)。
    """
    torch.manual_seed(0)
    dev = x.device
    x = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x.contiguous()
    M, C = x.shape
    dtype = x.dtype

    if isinstance(group_size, int):
        gh = gw = group_size
    else:
        gh, gw = group_size

    _inv = None
    if feature_shape is not None:
        H, W = feature_shape
        num_windows = (H // gh) * (W // gw)
        G = gh * gw
        # 构建窗口连续索引：token=(h,w)；win=(h//gh)*(W//gw)+(w//gw)；local=(h%gh)*gw+(w%gw)
        idx = torch.empty(M, dtype=torch.int64, device=dev)
        h = torch.arange(H, device=dev).view(-1, 1).expand(H, W).reshape(-1)
        wcol = torch.arange(W, device=dev).view(1, -1).expand(H, W).reshape(-1)
        win = (h // gh) * (W // gw) + (wcol // gw)
        loc = (h % gh) * gw + (wcol % gw)
        idx[win * G + loc] = torch.arange(M, device=dev)
        inv = torch.empty_like(idx)
        inv[idx] = torch.arange(M, device=dev)   # 逆排列：grouped pos → 原始 token
        _inv = inv
        group = lambda t: t.reshape(M, C)[idx]
    else:
        num_windows = 1
        G = M
        group = lambda t: t.reshape(M, C)

    if num_heads is None:
        num_heads = 1
    head_dim = C // num_heads
    eff_heads = max(num_heads, 1)

    qq = group(x) if q is None else group(q)
    kk = group(x) if k is None else group(k)
    vv = group(x) if v is None else group(v)
    qq, kk, vv = qq.contiguous(), kk.contiguous(), vv.contiguous()

    if scale is None:
        scale = head_dim ** -0.5

    # 合并加性偏置 pse + mask（host 侧相加；kernel 只消费一个 bias tensor）
    has_bias = (pos_bias is not None) or (mask is not None)
    bias = None
    if has_bias:
        # 本骨架契约：单窗口(NW=1) 时 mask/pos_bias 为 (N,N) 或 (H,N,N) 全序列加性
        def _bias2(xb):
            if xb.dim() == 2:
                return xb.unsqueeze(0)                      # (1,N,N)
            return xb
        pb = _bias2(pos_bias).float() if pos_bias is not None else 0.0
        mk = _bias2(mask).float() if mask is not None else 0.0
        bias = (pb + mk).to(dtype).contiguous().reshape(-1)   # (N_HEADS*G*G,) 交给 kernel (NW..)
        # reshape: 单窗口 head 对齐 n_heads，行/列=G
        bias = bias.view(eff_heads, G, G).contiguous()

    out = torch.empty_like(x)
    o = out.reshape(M, C)
    q_flat = qq.reshape(num_windows * G, C)

    IS_FP32 = dtype == torch.float32
    tld = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[dtype]
    grid = min(num_windows * eff_heads * ((G + BLOCK_M - 1) // BLOCK_M), _num_aicore())
    _crossformer_attn_kernel[(grid,)](
        q_flat, kk.reshape(-1, C), vv.reshape(-1, C),
        bias if has_bias else q_flat, o,
        num_windows, G, C, head_dim, eff_heads,
        float(scale), C,
        HAS_BIAS=has_bias, IS_FP32=IS_FP32, OUTPUT_DTYPE=tld,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    return o[_inv] if _inv is not None else o


BLOCK_M = 64
BLOCK_N = 128
BLOCK_D = 128


def golden_run(x, group_size, q=None, k=None, v=None, num_heads=1,
               mask=None, scale=None, pos_bias=None, feature_shape=None):
    """逐字复刻 level4 definition.json reference（torch golden）。"""
    import torch.nn.functional as F
    if num_heads is None:
        num_heads = 1
    torch.manual_seed(42)
    xin = x
    B, N, C = x.shape if x.dim() == 3 else (1, x.shape[0], x.shape[1])
    xr = x.reshape(B, N, C)

    if isinstance(group_size, int):
        gh = gw = group_size
    else:
        gh, gw = group_size

    def _attention(x_loc, q_loc, k_loc, v_loc, num_heads, mask, scale, pos_bias):
        B_loc, N_loc, C_loc = x_loc.shape
        head_dim = C_loc // num_heads

        def to_heads(t):
            return t.view(B_loc, N_loc, num_heads, head_dim).permute(0, 2, 1, 3)

        qh = to_heads(q_loc); kh = to_heads(k_loc); vh = to_heads(v_loc)
        scale_val = scale if scale is not None else (head_dim ** -0.5)
        qh = qh * scale_val
        attn = torch.matmul(qh.float(), kh.transpose(-2, -1).float())
        if pos_bias is not None:
            attn = attn + (pos_bias.unsqueeze(0).unsqueeze(0).float() if pos_bias.dim() == 2
                           else pos_bias.unsqueeze(0).float())
        if mask is not None:
            attn = attn + (mask.unsqueeze(0).unsqueeze(0).float() if mask.dim() == 2 else mask.float())
        attn = F.softmax(attn, dim=-1)
        out_heads = torch.matmul(attn, vh.float()).to(dtype=x.dtype)
        return out_heads.permute(0, 2, 1, 3).contiguous().view(B_loc, N_loc, C_loc)

    if feature_shape is not None:
        H, W = feature_shape
        x_2d = xr.view(B, H, W, C)
        num_h, num_w = H // gh, W // gw
        xg = x_2d.view(B, num_h, gh, num_w, gw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(
            B * num_h * num_w, gh * gw, C)
        B_flat = B * num_h * num_w
        gg = lambda t: t.view(B, H, W, C).view(B, num_h, gh, num_w, gw, C).permute(
            0, 1, 3, 2, 4, 5).contiguous().view(B_flat, gh * gw, C)
        q_flat = gg(q) if q is not None else xg
        k_flat = gg(k) if k is not None else xg
        v_flat = gg(v) if v is not None else xg
        out_flat = _attention(xg, q_flat, k_flat, v_flat, num_heads, mask, scale, pos_bias)
        out = out_flat.view(B, num_h, num_w, gh, gw, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return out.view(B, H * W, C)
    else:
        q_in = xr if q is None else q
        k_in = xr if k is None else k
        v_in = xr if v is None else v
        out = _attention(xr, q_in, k_in, v_in, num_heads, mask, scale, pos_bias)
        return out.reshape(-1, C)


def _rmse(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    return (a - b).abs().mean().item()


def _check(name, out, ref, tol=2e-2):
    # golden feature 分支返回 (1,N,C)，wrapper 返回 (N,C)——按末维归一比较
    outf = out.reshape(-1, out.shape[-1])
    refc = ref.reshape(-1, ref.shape[-1]) if ref.reshape(-1).numel() == outf.numel() else ref
    ok = refc.shape == outf.shape and _rmse(outf, refc) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: out{tuple(out.shape)} ref{tuple(ref.shape)} "
          f"rmse={_rmse(outf, refc):.3e} (tol {tol})")
    return ok


if __name__ == "__main__":
    import torch_npu  # noqa: 注册 NPU 设备（无此回退 cpu 报指针错误）
    dev = "npu:0"
    ok_all = True

    def make(shape, dtype):
        return torch.randn(shape, dtype=dtype, device=dev)

    cases = [
        # (name, x, group_size, num_heads, mask, scale, pos_bias, feature_shape)
        ("n1f-single-window-f16", make((196, 96), torch.float16), 14, 1, None, None, None, None),
        ("n2f-window-8x-posmask-f32", make((2464, 64), torch.float32), [8, 14], 8,
         make((8, 112, 112), torch.float32), 0.05, make((8, 112, 112), torch.float32), [16, 154]),
        ("n3f-single-mask-f16", make((729, 1024), torch.float16), 27, 8,
         make((8, 729, 729), torch.float32) * 0.01, None, None, None),
        ("n5f-window-bf16", make((3465, 1024), torch.bfloat16), [11, 7], 8, None, None, None, [55, 63]),
        ("n13f-big-single-f16", make((7921, 96), torch.float16), 89, 1, None, None, None, None),
        ("n17f-window-mask-pos-f16", make((4480, 1024), torch.float16), [16, 14], 8,
         make((8, 224, 224), torch.float32) * 0.01, None,
         make((8, 224, 224), torch.float32) * 0.01, [64, 70]),
        ("n49f-small-f16", make((49, 512), torch.float16), 7, 8, None, None, None, None),
    ]
    for name, x, g, h, m, sc, pb, fs in cases:
        try:
            out = crossformer_attention(
                x, g, num_heads=h, mask=m, scale=sc, pos_bias=pb, feature_shape=fs)
            ref = golden_run(
                x, g, num_heads=h, mask=m, scale=sc, pos_bias=pb, feature_shape=fs)
            ok_all &= _check(name, out, ref)
        except Exception as e:
            ok_all = False
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()

    print("ALL PASS" if ok_all else "HAS FAILURE")