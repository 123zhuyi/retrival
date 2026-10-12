"""P2 Triton 骨架 v2 — chunk_gated_delta_rule_fwd_h（2026-09-15，三组生成实验回流版）

相对 v1（2026-09-14，冻结于 ../triton_skeleton.py）的修订——全部有三组同机生成实验
（A1 KB 2.3535x / A2 仅 A 卡 1.6873x / B3 v1卡+KB 1.3251x）与 MCTS 0901 的 [EXP]
依据，详见 ../generation_retrospective_20260915.md 与
backend_facts/operator_class_patterns.md §4（状态常驻递推核）：

1. **grid 改回一块一 program**（grid=NUM_TASKS，删 persistent 外循环与 _aicore_num）：
   v1 按 C-011 改的 persistent-core 被 B3 证伪——iter_0 即 13/50 UB 溢出，5 轮 gen
   迭代耗于修复；A1/A2/MCTS 三独立 [EXP] 均一块一 program。C-011 已按算子类收窄。
2. **fp32 输入走原生 dot**：IN_FP32 时 tl.dot(..., input_precision="ieee")
   （A1/A2 双证；B3 的「原生 fp32 dot 不可用 → bf16x3 仿真」是失败上下文过度泛化，
   17 个 fp32 case 净亏 0.80x vs A1 2.05x——勿重试）。
3. **BV_TILE 实测标定规则**：BK·BV_TILE ≤ 2048 元素（超出减半至 16）——A1 50-case
   标定；cap 4096/8192 反亏。替代 v1 的解析 UB 分态（hardware_platforms §3.2 校准）。
4. **语义面向评测 golden**：85_/kernelbench Model 的 g/gk **同为 log2 域**
   （torch.exp2）——与 AscendC 源码 GDN 分支的 g 自然对数域不同，以评测 golden
   为准（三组独立确认）；tl.exp2 直译（≡ ×LN2+Exp，实现侧差 ≈0，A1 两态③闭环）。
   ref golden 同步改 85 语义（含三级舍入复刻：states 存 k.dtype / C1 前 state 降
   w.dtype / C2 前 update_value 降 k.dtype）。
5. **零初始化避 C-016**：tl.zeros 作 loop-carried dot 累加初值在 A1/A2 上下文触发
   bishengir 段错误（无诊断）——改全 False mask load 产零（B3 上下文幸存，不赌）。
6. svf 仍由 wrapper permute-view 消化（+177%/+173% 双证，勿回退）；对照法（kernel
   内小 tile tl.trans 连续 store，A2）留作 Phase 4 两态。
7. 保留 v1 正确增量：pow2/mask 泛化（BK/BV_TILE ≥16）、int64 标量大基址（4e）、
   actual−1 行、STORE_FINAL 两态、varlen 前缀和、USE_G/USE_GK constexpr。

**v2.1 修复（2026-09-15，B4 复跑暴露，v2 初版两处缺陷）**：
8. w 的 head 维索引 hk → **hv**（w 是逐 value-head [B,T,HV,K]；v1 原为 hv，v2 重写时
   引入回归——B4 iter_1 的 7 个 GQA case 全错即此，耗 1 轮 gen 迭代修复）
9. wrapper 的 BV 守卫改**纯三元表达式**（B4 iter_0：管线 AST checker 禁 forward 内
   max/min/while，耗 1 轮 gen 迭代修复；形式取自 B4 终版代码）

Phase 4 待试杠杆（已证 [EXP]，实现要点见 v2 卡 §10——勿在 Phase 3 内建，起点保持最小）：
k 条件预转置（+21%）、g 整块 load+片上 select（+17%）、h-无关 load 前置（+4.5%）

preflight（2026-09-15 对本文件执行）：13 项 grep 0 真命中（wrapper 不违禁）：
无 `%` / num_stages / num_warps / autotune；int64 仅标量块基址；mask 全 int32 比较；
grid=NUM_TASKS 一块一 program（本族实测纪律）；无大 tile tl.trans；tl.zeros 不在
loop-carried dot 上下文（C-016）；BK/BV_TILE 恒 2 的幂且 ≥16；BK·BV_TILE ≤ 2048 守卫。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _chunk_gdn_fwd_h_kernel(
    k_ptr, w_ptr, u_ptr, g_ptr, gk_ptr,
    is_ptr, h_ptr, v_new_ptr, fs_ptr,
    cu_ptr, co_ptr,
    HV, G, T, NVB,
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
    BK: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr,
    USE_INITIAL: tl.constexpr, STORE_FINAL: tl.constexpr,
    VARLEN: tl.constexpr,
    BV_TILE: tl.constexpr,
    IN_FP32: tl.constexpr,
):
    # 一块一 program [EXP×3]：状态常驻递推核不适用 persistent-core（C-011 已收窄）
    task = tl.program_id(0)
    vb = task - (task // NVB) * NVB
    tmp = task // NVB
    hv = tmp - (tmp // HV) * HV
    b = tmp // HV
    hk = hv // G

    # 变长（4e：前缀和为标量；4c：int64 仅限标量 load）
    if VARLEN:
        n = b.to(tl.int64)
        bos = tl.load(cu_ptr + n).to(tl.int64)
        eos = tl.load(cu_ptr + n + 1).to(tl.int64)
        seq_len = eos - bos
        sb64 = tl.load(co_ptr + n).to(tl.int64)
    else:
        bos = (b * T).to(tl.int64)
        seq_len = T
        sb64 = (b * (T + BT - 1) // BT).to(tl.int64)
    NT = (seq_len + BT - 1) // BT

    b64 = b.to(tl.int64)  # 4e：batch/大偏移只用标量 int64

    tt = tl.arange(0, BT)
    kk = tl.arange(0, BK)  # BK = max(16, next_pow2(K))，恒 2 的幂且 ≥16
    vv = vb * BV_TILE + tl.arange(0, BV_TILE)  # BV_TILE 恒 2 的幂且 ≥16，BK·BV_TILE ≤ 2048

    kmask = kk < K
    vmask = vv < V
    kv_mask = kmask[:, None] & vmask[None, :]

    # h 初始化 [BK, BV_TILE] fp32。
    # C-016：tl.zeros 作 loop-carried dot 累加初值在部分上下文触发 bishengir 段错误
    # （A1/A2 双命中，无诊断）——零初值用全 False mask load 产零，等价且免疫。
    if USE_INITIAL:
        h = tl.load(
            is_ptr + b64 * sib + hv * sih + kk[:, None] * V + vv[None, :],
            mask=kv_mask, other=0.0).to(tl.float32)
    else:
        h = tl.load(
            is_ptr + b64 * sib + hv * sih + kk[:, None] * V + vv[None, :],
            mask=kv_mask & (kk[:, None] < 0), other=0.0).to(tl.float32)

    for i in range(0, NT):
        t0 = bos + i * BT
        actual = seq_len - i * BT
        if actual > BT:
            actual = BT
        tmask = tt < actual

        # 1) 存 chunk i 的起始状态（三级舍入①：states 存 k.dtype——85 Model:65）
        tl.store(
            h_ptr + sb64 * shb + hv * shh + i * shc + kk[:, None] * shk + vv[None, :],
            h.to(h_ptr.dtype.element_ty), mask=kv_mask)

        # 2) C1：v_new = u − w @ h（w/u 行块 [BT, BK]/[BT, BV_TILE]）
        wm = tmask[:, None] & kmask[None, :]
        vm = tmask[:, None] & vmask[None, :]
        # w 是逐 value-head [B,T,HV,K] → 用 hv 索引（v2 初版误写 hk，B4 复跑 iter_1
        # 暴露：G=2 的 7 个 GQA case 全错；v1 骨架原为 hv，本行已修正）
        w_t = tl.load(w_ptr + b64 * swb + hv * swh + (t0 + tt)[:, None] * swt
                      + kk[None, :], mask=wm, other=0.0)
        u_t = tl.load(u_ptr + b64 * sub + hv * suh + (t0 + tt)[:, None] * sut
                      + vv[None, :], mask=vm, other=0.0)
        # 三级舍入②：C1 前 state 降 w.dtype，fp32 累加（85 Model:70）。
        # fp32 输入走原生 ieee dot（A1/A2 双证）；bf16x3 仿真勿用（B3 反例）。
        if IN_FP32:
            acc = tl.dot(w_t, h.to(w_ptr.dtype.element_ty), input_precision="ieee")
        else:
            acc = tl.dot(w_t, h.to(w_ptr.dtype.element_ty))
        v_new = u_t.to(tl.float32) - acc  # v_new 输出 = 未衰减残差（golden）
        tl.store(
            v_new_ptr + b64 * svb + hv * svh + (t0 + tt)[:, None] * svt + vv[None, :],
            v_new.to(v_new_ptr.dtype.element_ty), mask=vm)

        # 3) 衰减（评测 golden 语义：g/gk 同为 log2 域 → exp2；与 AscendC 源码
        #    GDN 分支的 g 自然对数域不同，以 85 Model:74-83 为准。actual−1 行！）
        if USE_G:
            g_t = tl.load(g_ptr + b64 * sgb + (t0 + tt) * sgh + hv,
                          mask=tmask, other=0.0).to(tl.float32)
            g_last = tl.load(g_ptr + b64 * sgb + (t0 + actual - 1) * sgh + hv
                             ).to(tl.float32)
            v_up = v_new * tl.exp2(g_last - g_t)[:, None]  # fp32 残差先衰减
            h = h * tl.exp2(g_last)
        else:
            v_up = v_new
        if USE_GK:
            gk_last = tl.load(gk_ptr + b64 * sgkb + hv * sgkh
                              + (t0 + actual - 1) * sgkt + kk,
                              mask=kmask, other=0.0).to(tl.float32)
            h = h * tl.exp2(gk_last)[:, None]

        # 4) C2：h += kgᵀ @ (v_new⊙decay)（k 转置寻址 load [BK, BT]；
        #    Phase 4 杠杆② 可改条件预转置路径）
        km2 = kmask[:, None] & tmask[None, :]
        k_t = tl.load(k_ptr + b64 * skb + hk * skh + (t0 + tt)[None, :] * skt
                      + kk[:, None], mask=km2, other=0.0)
        # 三级舍入③：C2 前 update_value 降 k.dtype，fp32 累加（85 Model:79-80）
        if IN_FP32:
            h += tl.dot(k_t, v_up.to(k_ptr.dtype.element_ty), input_precision="ieee")
        else:
            h += tl.dot(k_t, v_up.to(k_ptr.dtype.element_ty))

    if STORE_FINAL:
        tl.store(fs_ptr + b64 * sfb + hv * sfh + kk[:, None] * V + vv[None, :],
                 h, mask=kv_mask)  # final_state 恒 fp32 输出


def _next_pow2(n):
    p = 16  # tl.dot 最小维 16
    while p < n:
        p *= 2
    return p


def chunk_gated_delta_rule_fwd_h(
    k, w, u, g=None, gk=None, initial_state=None,
    output_final_state=False, chunk_size=64,
    cu_seqlens=None, chunk_indices=None, state_v_first=False,
    gate_mode=None, BLOCK_V=64,
):
    """wrapper：计算/转换分离 + grid 一块一 program + BV 2048 守卫 + svf permute-view。

    gate_mode：85 套件显式传入（1=USE_G、2=USE_GK、0=都不用）；None 时按 g/gk 是否为
    None 推断（fla 语义）。
    """
    B, T, HK, K = k.shape
    HV = u.shape[2]
    V = u.shape[3]
    G = HV // HK
    BT = chunk_size

    varlen = cu_seqlens is not None
    if gate_mode is None:
        use_g = g is not None
        use_gk = gk is not None
    else:
        use_g = (gate_mode == 1)
        use_gk = (gate_mode == 2)
    if varlen:
        N = cu_seqlens.numel() - 1
        cu = cu_seqlens.to(torch.int32).contiguous()
        if chunk_indices is None:
            seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
            chunk_indices = torch.cat([
                torch.cat([torch.arange(n), torch.full((n,), i, dtype=torch.int64)])
                for i, n in enumerate(((seq_lens + BT - 1) // BT).tolist())
            ])
        co = chunk_indices.to(torch.int32).contiguous()
        total_chunks = int(co[-1].item()) + int((cu_seqlens[-1].item() + BT - 1) // BT)
        batch_size = N
        total_seq = T  # 此处 T 为 cu_seqlens[-1]（变长时 k 按 [1, sum(T), ...] 传入）
    else:
        cu = torch.empty(0, dtype=torch.int32, device=k.device)
        co = torch.empty(0, dtype=torch.int32, device=k.device)
        total_chunks = B * ((T + BT - 1) // BT)
        batch_size = B
        total_seq = T

    if initial_state is None:
        initial_state = torch.empty(0, dtype=torch.float32, device=k.device)
        sib, sih, use_initial = 0, 0, False
    else:
        # 接受 [B,HV,K,V]（state_v_first=False）或 [B,HV,V,K]（True）：统一转 [K,V]
        if state_v_first:
            initial_state = initial_state.transpose(-1, -2).contiguous()
        sib, sih, use_initial = HV * K * V, K * V, True

    BK = _next_pow2(K)  # 从 16 起步，恒 ≥16 且 2 的幂
    # BV_TILE 实测标定（A1 [EXP]）：上限 BLOCK_V(64)，守卫 BK·BV_TILE ≤ 2048 元素。
    # 纯三元表达式——B4 复跑 iter_0 教训：生成管线的 AST checker 禁 forward 内
    # max/min/while（A-PyTorchFallback-Type3）；cap=2048//BK 在 BK≤128 时恒 ≥16
    # 且为 2 的幂。K>128 时 2048 守卫不可满足 → 须 split-K（v2 卡 §5 大 shape 起点假设）
    BV_TILE = _next_pow2(V)
    BV_TILE = BV_TILE if BV_TILE <= BLOCK_V else BLOCK_V
    _cap = 2048 // BK
    BV_TILE = BV_TILE if BV_TILE <= _cap else _cap
    NVB = (V + BV_TILE - 1) // BV_TILE

    states = torch.empty(batch_size, HV, total_chunks, K, V, dtype=k.dtype, device=k.device)
    v_new = torch.empty_like(u)
    fs = (torch.empty(batch_size, HV, K, V, dtype=torch.float32, device=k.device)
          if output_final_state else torch.empty(0, dtype=torch.float32, device=k.device))

    grid = (batch_size * HV * NVB,)  # 一块一 program [EXP×3]，勿 persistent-core
    _chunk_gdn_fwd_h_kernel[grid](
        k, w, u,
        g if g is not None else k, gk if gk is not None else k,
        initial_state, states, v_new, fs,
        cu, co,
        HV, G, total_seq, NVB,
        T * HK * K, K * K, K,          # k strides（末维连续：skt=1）
        T * HK * K, K * K, K,          # w strides
        T * HV * V, V * V, V,          # u strides
        T * HV, HV,                    # g strides [B,T,HV]
        T * HV * K, K * K, K,          # gk strides [B,T,HV,K]
        sib, sih,
        states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        v_new.stride(0), v_new.stride(1), v_new.stride(2),
        (fs.stride(0) if output_final_state else 0),
        (fs.stride(1) if output_final_state else 0),
        K=K, V=V, BT=BT, BK=BK,
        USE_G=use_g, USE_GK=use_gk,
        USE_INITIAL=use_initial, STORE_FINAL=output_final_state,
        VARLEN=varlen,
        BV_TILE=BV_TILE,
        IN_FP32=(k.dtype == torch.float32),
    )
    if state_v_first:
        # 内部恒 [K,V] 连续写，svf 由返回 permute 视图消化（+177%/+173% 双证，勿回退）
        return states.transpose(-1, -2), v_new, (fs.transpose(-1, -2) if output_final_state else None)
    return states, v_new, (fs if output_final_state else None)


def ref_chunk_gated_delta_rule_fwd_h(
    k, w, u, g=None, gk=None, initial_state=None, chunk_size=64,
    gate_mode=None, state_v_first=False, cu_seqlens=None,
):
    """golden：复刻 85_ChunkGatedDeltaRuleFwdH.py Model 语义（评测 ground truth）：
    g/gk 同为 log2 域（torch.exp2）；三级舍入——states 存 k.dtype、C1 前 state 降
    w.dtype、C2 前 update_value 降 k.dtype；fp32 域累加。
    ⚠️ 与 fla 仓 test_fwd_h.py 的源码语义 golden 不同（其 g 为自然对数域 exp）——
    面向 85/kernelbench 评测用本函数；面向 fla 语义对照用 v1 骨架的 ref。
    """
    B, T, HK, K = k.shape
    HV = u.shape[2]
    V = u.shape[3]
    G = HV // HK
    BT = chunk_size
    if gate_mode is None:
        gate_mode = 1 if g is not None else (2 if gk is not None else 0)

    varlen = cu_seqlens is not None
    if varlen:
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        bos_list = cu_seqlens.tolist()
        NTs = [(n + BT - 1) // BT for n in seq_lens]
        co = torch.cumsum(torch.tensor([0] + NTs), dim=0)
        batch_size = len(seq_lens)
    else:
        seq_lens = [T] * B
        bos_list = [i * T for i in range(B)]
        NT = (T + BT - 1) // BT
        NTs = [NT] * B
        co = torch.arange(B + 1) * NT
        batch_size = B

    dtype_ = k.dtype
    S = torch.zeros(batch_size, HV, int(co[-1]), K, V, dtype=dtype_, device=k.device)
    v_new_out = torch.zeros(batch_size, HV, T, V, dtype=dtype_, device=k.device)
    fs = torch.zeros(batch_size, HV, K, V, dtype=torch.float32, device=k.device)

    for sb in range(batch_size):
        for hv in range(HV):
            if initial_state is not None:
                h_state = initial_state[sb, hv].float()
                if state_v_first:
                    h_state = h_state.t().contiguous()
            else:
                h_state = torch.zeros(K, V, dtype=torch.float32, device=k.device)
            for i in range(NTs[sb]):
                t0 = bos_list[sb] + i * BT
                actual = min(BT, seq_lens[sb] - i * BT)
                S[sb, hv, co[sb] + i] = h_state.to(dtype_)  # 舍入①：states 存 k.dtype
                kk_ = k[sb, hv // G, t0:t0 + actual].float()
                ww = w[sb, hv, t0:t0 + actual].float()
                uu = u[sb, hv, t0:t0 + actual].float()
                v_new = uu - ww @ h_state.to(w.dtype).float()  # 舍入②：C1 前 state 降 w.dtype
                v_new_out[sb, hv, t0:t0 + actual] = v_new.to(dtype_)
                v_up = v_new  # 衰减作用于 fp32 残差（v_new 输出保持未衰减）
                if gate_mode == 1 and g is not None:
                    g_seg = g[sb, hv, t0:t0 + actual].float()
                    g_last = g_seg[-1]  # actual−1 行；log2 域 → exp2（85 Model:74-80）
                    v_up = v_new * torch.exp2(g_last - g_seg)[:, None]
                    h_state = h_state * torch.exp2(g_last)
                if gate_mode == 2 and gk is not None:
                    gk_last = gk[sb, hv, t0 + actual - 1].float()  # actual−1 行
                    h_state = h_state * torch.exp2(gk_last)[:, None]
                h_state = h_state + kk_.t() @ v_up.to(dtype_).float()  # 舍入③：C2 前降 k.dtype
            fs[sb, hv] = h_state
    if state_v_first:
        return S.transpose(-1, -2), v_new_out, fs.transpose(-1, -2)
    return S, v_new_out, fs


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    for dtype in (torch.float16, torch.float32):
        for gm in (1, 2):
            k = torch.randn(1, 192, 2, 17, device=dev, dtype=dtype) * 0.1  # K=17 非 2 的幂
            w = torch.randn(1, 192, 4, 17, device=dev, dtype=dtype) * 0.1
            u = torch.randn(1, 192, 4, 9, device=dev, dtype=dtype) * 0.1  # V=9 非 2 的幂
            g = torch.rand(1, 192, 4, device=dev).cumsum(1) * -0.1
            gk = torch.rand(1, 192, 4, 17, device=dev).cumsum(1) * -0.1
            o = chunk_gated_delta_rule_fwd_h(
                k, w, u, g=g, gk=gk, output_final_state=True, chunk_size=64,
                gate_mode=gm)
            r = ref_chunk_gated_delta_rule_fwd_h(
                k, w, u, g=g, gk=gk, chunk_size=64, gate_mode=gm)
            for name, a, b in zip(("states", "v_new", "fs"), o, r):
                assert torch.allclose(a.float(), b.float(), atol=1e-2, rtol=1e-2), \
                    (dtype, gm, name)
            print(f"dtype={dtype} gate_mode={gm}: OK")
