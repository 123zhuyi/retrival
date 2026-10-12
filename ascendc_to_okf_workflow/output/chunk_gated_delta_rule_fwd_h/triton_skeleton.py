"""P2 Triton 骨架 — chunk_gated_delta_rule_fwd_h（新流程版，2026-09-14）

GDN/KDA chunk 间状态递推前向：
  v_new[t] = u[t] − w[t] @ h_i；h_{i+1} = h_i·exp(g_last)·exp2(gk_last) + kg^T@(v_new·decay)
（README.md:20-27 + golden test_fwd_h.py:111-115，2026-09-14 复验 6/6 锚点）

设计决策与 backend_facts 取证对照（mapping.md §0.5 检索记录）：
- 任务轴 (b, hv, v_block) 状态条带：V 列条带独立（kernel .hpp:556,682），h 条带
  [BK, BV_TILE] fp32 常驻片内——结构性消解 CV 四段流水 / GM workspace 六段 /
  direct-UB 直通（比它更彻底，GM 往返逐 chunk 可数论证，非 C-012 断言）
- **persistent-core grid = min(NUM_TASKS, num_aicore)**（C-011：含 tl.dot = CV 融合
  取 Cube 数 [DOC]；旧版「一块一 program」的自我豁免无实测、违反 C-012，已修正。
  MCTS 910b 实测一块一 program 7.23x 留作 P4 两态对照）。核数运行时查询、禁止写死
  （hardware_platforms §3：950 AIC 4~36 跨度 9 倍）
- **pow2/mask 泛化**（mapping §0.6-3，旧骨架潜伏缺陷）：官方 50-case benchmark 的
  K∈{7..80}、V∈{5..50} 非 2 的幂——tl.arange 须 2 的幂且 tl.dot 最小维 16。
  BK = max(16, next_pow2(K))、BV_TILE = max(16, min(128, next_pow2(V)))，
  kk/vv 全程 mask（旧骨架在官方正确性套件上编译即失败）
- **int64 安全寻址**（dtype_support 4e）：密集 B=16 时 b·h.stride(0) 可达 4e9 > 2³¹；
  大基址用 int64 标量（不占向量 lane），块内偏移 int32（(t0+tt)·K ≤ 1.7e7 安全）
- 精度对齐源码：h 条带/v_new/decay fp32 域；dot 输入 cast 回 input dtype（mmad
  fp16-in/fp32-acc 语义）；store 隐式 RTNE；fp32 final_state 直通天然成立
- exp2 语义 = ×LN2+Exp（regbase.hpp:53-73 已复验，与源码逐位同构）；
  tl.exp2 是否更优 ⚠️ 未实测两态，勿断言（C-012）
- k 转置寻址 load [BK,BT]——禁 tl.trans（标量化 ~15x runbook）；
  列主序访存性能 ⚠️ 实测定（劣化则 wrapper 预转置，未实测勿断言）
- 无 autotune（BLOCK 全静态推导）；无 num_stages/num_warps（C-006 运行时不生效）；
  双缓冲由 multibuffer 承接（bishengir 默认 true [SRC Options.td]；本算子独立 [EXP]
  交叉验证：MCTS opt-round-5 探针 multibuffer=False -5.9% 后回退）
- mask 全 int32 比较（C-002）；无 % 运算符（解码 a-(a//b)*b，C-001）；
  cu_seqlens int64 仅标量 load（i64 不进向量 lane，dtype_support §2.1）
- UB 分平台预算（mapping §4）：950=248KB [SRC] 上 BT=128/BK=128/BV=128（~190KB）可行；
  910b=192KB 上同配置贴顶 → BLOCK_V=64（与 MCTS 910b1 冠军 BV=min(64,V) [EXP] 一致）

preflight（backend_contracts 8 项 + dtype_support 4a~4e，2026-09-14 对本文件执行）：
13 项 0 真命中（头注禁令文字已剥离后扫描；wrapper host-Python 不违禁）。
"""

import torch
import triton
import triton.language as tl

LN2 = 0.6931471805599453


def _aicore_num() -> int:
    """运行时查询 AIC（Cube）核数；CV 融合算子 grid 取 num_aicore（C-011）。
    禁止写死（hardware_platforms §3：950 AIC 4~36 跨度 9 倍）；
    fallback 24 仅为兜底，查询失败时 grid 可能错配——P4 须确认查询成功。"""
    try:
        import triton.runtime.driver as driver
        return driver.active.utils.get_device_properties(
            torch.npu.current_device())["num_aicore"]
    except Exception:
        return 24


@triton.jit
def _chunk_gdn_fwd_h_kernel(
    k_ptr, w_ptr, u_ptr, g_ptr, gk_ptr,
    is_ptr, h_ptr, v_new_ptr, fs_ptr,
    cu_ptr, co_ptr,
    NUM_TASKS, HV, G, T,
    skb, skh, skt,
    swb, swh, swt,
    sub, suh, sut,
    sgb, sgh,
    sgkb, sgkh, sgkt,
    sib, sih,
    shb, shh, shc, shk,
    svb, svh, svt,
    sfb, sfh,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    BK: tl.constexpr,                 # max(16, next_pow2(K))——dot 最小维 16
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    USE_INITIAL: tl.constexpr, STORE_FINAL: tl.constexpr,
    VARLEN: tl.constexpr,
    BV_TILE: tl.constexpr,            # max(16, min(128, next_pow2(V)))
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    NVB: tl.constexpr = (V + BV_TILE - 1) // BV_TILE

    kk = tl.arange(0, BK)
    kmask = kk < K                              # int32 mask（C-002）
    tt = tl.arange(0, BT)

    for task in range(pid, NUM_TASKS, num_progs):
        # 任务解码：禁 % 运算符（C-001）
        vb = task - (task // NVB) * NVB
        tmp = task // NVB
        hv = tmp - (tmp // HV) * HV
        b = tmp // HV
        hk = hv // G                            # GQA 头分组（scheduler:267 已复验）

        if VARLEN:
            bos = tl.load(cu_ptr + b).to(tl.int32)          # int64 张量仅标量 load
            t_total = tl.load(cu_ptr + b + 1).to(tl.int32) - bos
            chunk_off = tl.load(co_ptr + b).to(tl.int32)
            nt = tl.load(co_ptr + b + 1).to(tl.int32) - chunk_off
            sb64 = bos.to(tl.int64) * 0         # 变长时输入/h/v_new 的 shape batch 恒 0
        else:
            bos = 0                           # VARLEN 为 constexpr，死分支 trace 时剪枝，
            t_total = T                       # 故密集路径可直接用 Python int 标量
            chunk_off = 0
            nt = (T + BT - 1) // BT
            sb64 = b.to(tl.int64)             # 4e：大基址 int64 标量
        b64 = b.to(tl.int64)                    # initial_state/final_state 用 b（紧凑 seq 序号）

        vv = vb * BV_TILE + tl.arange(0, BV_TILE)
        vmask = vv < V
        kv_mask = kmask[:, None] & vmask[None, :]

        # h 条带初始化（fp32 常驻，[BK, BV_TILE]）
        if USE_INITIAL:
            h = tl.load(is_ptr + b64 * sib + hv * sih
                        + kk[:, None] * V + vv[None, :],
                        mask=kv_mask, other=0.0).to(tl.float32)
        else:
            h = tl.zeros((BK, BV_TILE), dtype=tl.float32)

        out_dtype = h_ptr.dtype.element_ty

        for i in range(0, nt):
            t0 = bos + i * BT
            actual = tl.minimum(BT, t_total - i * BT)
            tmask = tt < actual

            # 存 chunk i 的起始状态 h[i]（RTNE 回 input dtype；反向要用）
            tl.store(h_ptr + sb64 * shb + hv * shh + (chunk_off + i) * shc
                     + kk[:, None] * shk + vv[None, :],
                     h.to(out_dtype), mask=kv_mask)

            # C1+V1: v_new = u - w @ h（dot 输入 cast 回 input dtype，对齐 mmad 语义）
            w_t = tl.load(w_ptr + sb64 * swb + hv * swh
                          + (t0 + tt)[:, None] * swt + kk[None, :],
                          mask=tmask[:, None] & kmask[None, :], other=0.0)
            u_t = tl.load(u_ptr + sb64 * sub + hv * suh
                          + (t0 + tt)[:, None] * sut + vv[None, :],
                          mask=tmask[:, None] & vmask[None, :], other=0.0)
            v_new = u_t.to(tl.float32) - tl.dot(w_t, h.to(out_dtype))

            tl.store(v_new_ptr + sb64 * svb + hv * svh
                     + (t0 + tt)[:, None] * svt + vv[None, :],
                     v_new.to(out_dtype),
                     mask=tmask[:, None] & vmask[None, :])

            # V1 衰减 + V2 h 更新
            if USE_G:
                g_t = tl.load(g_ptr + sb64 * sgb + hv * sgh + t0 + tt,
                              mask=tmask, other=0.0).to(tl.float32)
                # ⚠️ 正确性高危：g_last 必须取 actual-1 行（尾 chunk 不是 BT-1）
                g_last = tl.load(g_ptr + sb64 * sgb + hv * sgh
                                 + t0 + actual - 1).to(tl.float32)
                v_up = v_new * tl.exp(g_last - g_t)[:, None]
                h = h * tl.exp(g_last)
            else:
                v_up = v_new

            if USE_GK:
                gk_last = tl.load(gk_ptr + sb64 * sgkb + hv * sgkh
                                  + (t0 + actual - 1) * sgkt + kk,
                                  mask=kmask, other=0.0).to(tl.float32)  # [BK]
                # exp2 语义 = ×LN2+Exp（regbase.hpp:53-73 已复验，逐位同构）
                h = h * tl.exp(gk_last * LN2)[:, None]
                # kg = k（KDA 的逐 key 预衰减由调用方完成，vnew epilogue:407）

            # C2: h += k^T @ v_up（k 转置寻址 load [BK,BT]——禁 tl.trans，标量化 15x）
            k_t = tl.load(k_ptr + sb64 * skb + hk * skh
                          + (t0 + tt)[None, :] * skt + kk[:, None],
                          mask=kmask[:, None] & tmask[None, :], other=0.0)
            h += tl.dot(k_t, v_up.to(out_dtype))

        if STORE_FINAL:
            # fp32 state 直通天然成立（条带本就 fp32，store dtype 由 fs 指针定）
            tl.store(fs_ptr + b64 * sfb + hv * sfh
                     + kk[:, None] * V + vv[None, :],
                     h.to(fs_ptr.dtype.element_ty), mask=kv_mask)


def chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None,
                                 output_final_state=False, chunk_size=64,
                                 cu_seqlens=None, chunk_indices=None,
                                 state_v_first=False, BLOCK_V=128):
    """wrapper：gk/g 至少给一个（对齐 tiling.cpp:91-93）。
    BLOCK_V：950 可 128；910b（UB 192KB）BT=128&K=128 时建议 64（mapping §4）。
    """
    assert (g is not None) or (gk is not None), "Either g or gk must be provided."
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    assert HV % HK == 0
    G = HV // HK
    BT = chunk_size
    assert BT & (BT - 1) == 0, "chunk_size 须为 2 的幂"
    BV_TILE = max(16, min(BLOCK_V, triton.next_power_of_2(V)))  # dot 最小维 16
    NVB = (V + BV_TILE - 1) // BV_TILE

    if cu_seqlens is None:
        N, NT_flat = B, (T + BT - 1) // BT
        cu = co = k  # 占位指针，VARLEN=False 不会读
        h = torch.empty((B, HV, NT_flat, K, V), dtype=k.dtype, device=k.device)
    else:
        cu = cu_seqlens.to(torch.int64)
        lens = cu[1:] - cu[:-1]
        nchunks = (lens + BT - 1) // BT
        co = torch.cat([torch.zeros(1, dtype=torch.int64, device=cu.device),
                        nchunks.cumsum(0)])
        N, NT_flat = lens.numel(), int(co[-1])
        h = torch.empty((1, HV, NT_flat, K, V), dtype=k.dtype, device=k.device)

    v_new = torch.empty_like(u)
    if output_final_state:
        fs_dtype = initial_state.dtype if initial_state is not None else torch.float32
        fs = torch.empty((N, HV, K, V), dtype=fs_dtype, device=k.device)
    else:
        # Triton 的 h 条带寄存器常驻，不需要 aclnn 的不可见 finalStateCompute 工作区
        fs = k  # 占位指针，STORE_FINAL=False 不会写

    num_tasks = N * HV * NVB
    grid = (min(num_tasks, _aicore_num()),)   # persistent-core（C-011），CV 融合取 aicore
    _chunk_gdn_fwd_h_kernel[grid](
        k, w, u,
        g if g is not None else k,
        gk if gk is not None else k,
        initial_state if initial_state is not None else k,
        h, v_new, fs,
        cu, co,
        num_tasks, HV, G, T,
        k.stride(0), k.stride(1), k.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        u.stride(0), u.stride(1), u.stride(2),
        g.stride(0) if g is not None else 0, g.stride(1) if g is not None else 0,
        gk.stride(0) if gk is not None else 0, gk.stride(1) if gk is not None else 0,
        gk.stride(2) if gk is not None else 0,
        initial_state.stride(0) if initial_state is not None else 0,
        initial_state.stride(1) if initial_state is not None else 0,
        h.stride(0), h.stride(1), h.stride(2), h.stride(3),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        fs.stride(0) if output_final_state else 0,
        fs.stride(1) if output_final_state else 0,
        K=K, V=V, BT=BT, BK=max(16, triton.next_power_of_2(K)),
        USE_G=g is not None, USE_GK=gk is not None,
        USE_INITIAL=initial_state is not None,
        STORE_FINAL=output_final_state,
        VARLEN=cu_seqlens is not None,
        BV_TILE=BV_TILE,
    )
    if state_v_first:
        h = h.transpose(-1, -2)  # 零拷贝 stride 交换（aclnn TransposeLastTwo 不需要复刻）
        if output_final_state:
            fs = fs.transpose(-1, -2)
    return h, v_new, (fs if output_final_state else None)


def ref_chunk_gated_delta_rule_fwd_h(k, w, u, g=None, gk=None, initial_state=None,
                                     chunk_size=64, cu_seqlens=None):
    """golden：复刻 tests/pta/test_fwd_h.py forward_h_trans_cpu，全 fp32 域。
    填充纪律：g 必须 chunk 内 cumsum（单调负衰减），见 mapping.md §7。"""
    dtype_ = k.dtype
    k, w, u = k.float(), w.float(), u.float()
    B, HK, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    BT = chunk_size
    G = HV // HK

    if cu_seqlens is None:
        lens = [T] * B
    else:
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    N = len(lens)
    NTs = [(l + BT - 1) // BT for l in lens]
    co = [0]
    for c in NTs:
        co.append(co[-1] + c)
    NT_flat = co[-1]

    S = torch.zeros((B if cu_seqlens is None else 1, HV, NT_flat, K, V),
                    dtype=torch.float32)
    v_new_out = torch.zeros_like(u)
    fs = torch.zeros((N, HV, K, V), dtype=torch.float32)

    for n in range(N):
        bos = 0 if cu_seqlens is None else int(cu_seqlens[n])
        sb = n if cu_seqlens is None else 0
        for hv in range(HV):
            h_state = (initial_state[n, hv].float()
                       if initial_state is not None else torch.zeros(K, V))
            for i in range(NTs[n]):
                t0 = bos + i * BT
                actual = min(BT, lens[n] - i * BT)
                S[sb, hv, co[n] + i] = h_state
                kk_ = k[sb, hv // G, t0:t0 + actual]          # [a, K]
                ww = w[sb, hv, t0:t0 + actual]                # [a, K]
                uu = u[sb, hv, t0:t0 + actual]                # [a, V]
                v_new = uu - ww @ h_state
                v_new_out[sb, hv, t0:t0 + actual] = v_new
                v_up = v_new
                if g is not None:
                    g_seg = g[sb, hv, t0:t0 + actual].float()
                    g_last = g_seg[-1]
                    v_up = v_new * torch.exp(g_last - g_seg)[:, None]
                    h_state = h_state * torch.exp(g_last)
                if gk is not None:
                    gk_last = gk[sb, hv, t0 + actual - 1].float()   # [K]
                    h_state = h_state * torch.exp2(gk_last)[:, None]
                    # kg = k（调用方已预衰减）
                h_state = h_state + kk_.t() @ v_up
            fs[n, hv] = h_state
    return S.to(dtype_), v_new_out.to(dtype_), fs
