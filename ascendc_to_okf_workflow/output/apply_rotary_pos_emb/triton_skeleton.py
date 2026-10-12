"""P2 Triton 骨架 — apply_rotary_pos_emb（新流程版，2026-09-14）
out = x*cos + rotate(x)*sin，q/k 两路同式分两次 launch（[EXP] 融合 3.8252 < 拆分 4.2060）。

设计决策与 backend_facts 取证对照（mapping.md §0.5 检索记录）：
- (b,s) 组为 persistent-core 外循环（C-011 范式），grid=min(BS, 物理向量核数)；
  核数运行时查询、禁止写死（hardware_platforms §3：950 向量核 8~72）
- cos/sin 组级外提：每 (b,s) 只载一次（[EXP] +100%；C-012 禁止「cache 驻留等价」断言，
  源码依据 BA/BAB「cos/sin 载一次复用」aba_and_ba.h:185-196）
- bs 标量循环行解码：b = bs // S; s = bs - b * S（禁 % 运算符，C-001；
  标量 div 非向量 vmod，不受 dtype_support vmod-i32 标量化影响）
- interleave：全程连续 load/store + tl.split/tl.interleave 寄存器重排——
  HIVM VInterleave/VDeinterleave fp32 向量原生 [SRC dtype_support §2.1]；
  strided-2 半宽访存 ❌ 禁用（[EXP] 静默错值 + masked store 相邻行污染）
- 无 @triton.autotune（C-007 µs-kernel do_bench 噪声误选）；
  无 num_stages/num_warps（C-006 运行时不生效）；双缓冲由 multibuffer 选项承接
  （bishengir 默认 true [SRC Options.td EnableAutoMultiBuffer]）
- mask 保持 int32 比较（C-002 fp32 导出 mask 非确定 store 污染；
  i32 LT 的 vcmp 标量化代价在此为正确性让路，dtype_support §3 规则 2）
- 形状/stride 全 constexpr 化（C-013 benchmark 稳定值，[EXP] +2.2%）
- 全程 fp32 计算域（common.h 全部 RegTensor<float>），末段单次 RTNE 回写
- partial rope（RD<D）：wrapper out=x.clone() 保底直通段，kernel 只写 [0,RD)
- UB 预算：每迭代活跃 < 32KB vs 950 UB 248KB 可用 [SRC hardware_platforms]，富余 ~8 倍
"""
import torch
import triton
import triton.language as tl

HALF, INTERLEAVE, QUARTER = 1, 2, 3  # 对齐 tiling.h:146-150 枚举值


@triton.jit
def apply_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    BS, S, N, RD,                     # BS = B*S（TND 时 B=1）；D 不进 kernel（直通段 wrapper 保底）
    sxb, sxs, sxn,                    # x 的 (b, s, n) 三轴元素 stride（layout 归一化）
    scb, scs,                         # cos/sin 的 (b, s) 两轴 stride；CosB==1 时 scb=0
    SEG,                              # half/interleave: RD//2；quarter: RD//4
    ROTARY_MODE: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_DH: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    per_prog = tl.cdiv(BS, num_progs)  # 连续块划分（非交织）：相邻 bs 共享 cos 行，L2 友好
    bs_start = pid * per_prog
    bs_end = tl.minimum(bs_start + per_prog, BS)

    nn_idx = tl.arange(0, BLOCK_N)    # BLOCK_N = next_pow2(N)，单 tile 覆盖组内全部 n 行
    nm = nn_idx < N                   # int32 mask（C-002：store mask 保持整型比较）
    i = tl.arange(0, BLOCK_DH)        # 配对索引
    mi = i < SEG

    for bs in range(bs_start, bs_end):
        b = bs // S
        s = bs - b * S                # 标量解码；禁 % 运算符（C-001）
        x_base = b * sxb + s * sxs + nn_idx * sxn      # [BLOCK_N]
        c_base = b * scb + s * scs                     # 标量

        if ROTARY_MODE == HALF or ROTARY_MODE == QUARTER:
            m = nm[:, None] & mi[None, :]
            # cos/sin 循环不变量外提：每 (b,s) 组只载一次（[EXP] +100%，C-012）
            c1 = tl.load(cos_ptr + c_base + i, mask=mi, other=0).to(tl.float32)
            c2 = tl.load(cos_ptr + c_base + SEG + i, mask=mi, other=0).to(tl.float32)
            s1 = tl.load(sin_ptr + c_base + i, mask=mi, other=0).to(tl.float32)
            s2 = tl.load(sin_ptr + c_base + SEG + i, mask=mi, other=0).to(tl.float32)
            p1 = x_base[:, None] + i[None, :]
            p2 = p1 + SEG
            x1 = tl.load(x_ptr + p1, mask=m, other=0).to(tl.float32)
            x2 = tl.load(x_ptr + p2, mask=m, other=0).to(tl.float32)
            # out1 = x1*c1 - x2*s1 ; out2 = x2*c2 + x1*s2（common.h:111-119 已复验）
            o1 = x1 * c1[None, :] - x2 * s1[None, :]
            o2 = x2 * c2[None, :] + x1 * s2[None, :]
            tl.store(out_ptr + p1, o1.to(out_ptr.dtype.element_ty), mask=m)  # RTNE 回写
            tl.store(out_ptr + p2, o2.to(out_ptr.dtype.element_ty), mask=m)
            if ROTARY_MODE == QUARTER:
                # 第二对段 (i+2SEG, i+3SEG)，同构（common.h:201-209）
                c3 = tl.load(cos_ptr + c_base + 2 * SEG + i, mask=mi, other=0).to(tl.float32)
                c4 = tl.load(cos_ptr + c_base + 3 * SEG + i, mask=mi, other=0).to(tl.float32)
                s3 = tl.load(sin_ptr + c_base + 2 * SEG + i, mask=mi, other=0).to(tl.float32)
                s4 = tl.load(sin_ptr + c_base + 3 * SEG + i, mask=mi, other=0).to(tl.float32)
                p3 = p1 + 2 * SEG
                p4 = p1 + 3 * SEG
                x3 = tl.load(x_ptr + p3, mask=m, other=0).to(tl.float32)
                x4 = tl.load(x_ptr + p4, mask=m, other=0).to(tl.float32)
                o3 = x3 * c3[None, :] - x4 * s3[None, :]
                o4 = x4 * c4[None, :] + x3 * s4[None, :]  # p4 区段用 s4（baseline 初版笔误 s3，已修）
                tl.store(out_ptr + p3, o3.to(out_ptr.dtype.element_ty), mask=m)
                tl.store(out_ptr + p4, o4.to(out_ptr.dtype.element_ty), mask=m)
        else:
            # INTERLEAVE：全程连续 load/store + 寄存器重排（strided-2 ❌ 禁用）
            j = tl.arange(0, 2 * BLOCK_DH)        # 连续覆盖 [0, RD)
            jm = j < RD
            m = nm[:, None] & jm[None, :]
            p = x_base[:, None] + j[None, :]
            xrow = tl.load(x_ptr + p, mask=m, other=0).to(tl.float32)
            crow = tl.load(cos_ptr + c_base + j, mask=jm, other=0).to(tl.float32)
            srow = tl.load(sin_ptr + c_base + j, mask=jm, other=0).to(tl.float32)
            # 寄存器级 deinterleave：(BLOCK_N, BLOCK_DH, 2) → even/odd（common.h:270-280）
            xe, xo = tl.split(tl.reshape(xrow, (BLOCK_N, BLOCK_DH, 2)))
            ce_v, co_v = tl.split(tl.reshape(crow, (BLOCK_DH, 2)))
            se_v, so_v = tl.split(tl.reshape(srow, (BLOCK_DH, 2)))
            # out[2i] = xe*ce - xo*se ; out[2i+1] = xo*co + xe*so
            oe = xe * ce_v[None, :] - xo * se_v[None, :]
            oo = xo * co_v[None, :] + xe * so_v[None, :]
            o = tl.interleave(oe, oo)             # 寄存器重组回连续布局
            tl.store(out_ptr + p, o.to(out_ptr.dtype.element_ty), mask=m)


def _vector_core_num() -> int:
    """运行时查询向量核数；禁止写死（hardware_platforms §3：950 向量核 8~72 跨度 9 倍）。
    fallback 40 仅为 910B3/910_93 常见值，950 上必须查询成功，否则 grid 严重错配。"""
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except Exception:
        return 40


# layout → (b轴, s轴, n轴) 在 shape 中的下标（tiling_base_arch35.cpp:284-320 SetBsnd 同构）
_LAYOUT_DIMS = {1: (0, 1, 2), 2: (1, 0, 2), 3: (0, 2, 1), 4: (None, 0, 1)}  # TND 无 b 轴


def _launch_one(x, cos, sin, out, layout: int, rotary_mode: int):
    bi, si, ni = _LAYOUT_DIMS[layout]
    if layout == 4:                       # TND: (T, N, D)，T 视作 S，B=1
        B, S, N = 1, x.shape[si], x.shape[ni]
        sxb = 0
    else:
        B, S, N = x.shape[bi], x.shape[si], x.shape[ni]
        sxb = x.stride(bi)
    sxs, sxn = x.stride(si), x.stride(ni)
    RD = cos.shape[-1]
    D = x.shape[-1]
    assert RD <= D and (RD % 2 == 0 if rotary_mode != QUARTER else RD % 4 == 0)
    assert cos.shape[-2] == 1 and cos.shape == sin.shape      # cosN==1（tiling_base L182-184）
    cos_b_dim = 0 if layout in (1, 3) else 1                # cos 的 B 轴下标（TND 无）
    scb = 0 if (layout == 4 or cos.shape[cos_b_dim] == 1) else cos.stride(cos_b_dim)
    scs = cos.stride(si if layout != 4 else 0)
    BS = B * S
    if BS == 0 or N == 0:
        return
    # half/interleave: 配对宽=RD//2；quarter: 段宽=RD//4（源码 dSplitCoef 2/1/4 的对偶）
    seg = RD // 4 if rotary_mode == QUARTER else RD // 2
    block_dh = triton.next_power_of_2(seg)
    block_n = triton.next_power_of_2(N)   # N>128 时需改内层循环（本骨架面向 N≤128 主流）
    num_cores = _vector_core_num()
    grid = (min(BS, num_cores),)          # persistent-core（C-011）；勿用裸 BS grid
    apply_rope_kernel[grid](x, cos, sin, out, BS, S, N, RD, sxb, sxs, sxn, scb, scs,
                            seg, rotary_mode, BLOCK_N=block_n, BLOCK_DH=block_dh)


def apply_rotary_pos_emb(query, key, cos, sin, layout: int = 1, rotary_mode: str = "half"):
    """对应 aclnnApplyRotaryPosEmbV2；rotary_mode ∈ {"half","interleave","quarter"}"""
    assert query.dtype == key.dtype == cos.dtype == sin.dtype
    assert query.dtype in (torch.float16, torch.float32, torch.bfloat16)
    q, k, cos, sin = (t.contiguous() for t in (query, key, cos, sin))  # AutoContiguous 对应
    mode = {"half": HALF, "interleave": INTERLEAVE, "quarter": QUARTER}[rotary_mode]
    RD = cos.shape[-1]
    # partial rope：RD<D 时直通段由 clone 保底（common.h CopyIn/Out*Mode 双段搬运的等价）；
    # 源码是原地更新语义（op_graph tensormove pass 图级证据），性能敏感场景可改 out=x 真原地
    q_out = q.clone() if RD < q.shape[-1] else torch.empty_like(q)
    k_out = k.clone() if RD < k.shape[-1] else torch.empty_like(k)
    _launch_one(q, cos, sin, q_out, layout, mode)
    _launch_one(k, cos, sin, k_out, layout, mode)
    return q_out, k_out


# ---- 自建标杆（benchmark 已有 45-case 精确 case，本节用于补边界）----
def _rotate(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "half":
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    if mode == "quarter":
        x1, x2, x3, x4 = x.chunk(4, dim=-1)
        return torch.cat((-x2, x1, -x4, x3), dim=-1)
    x1, x2 = x[..., 0::2], x[..., 1::2]               # interleave
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def ref_apply_rotary_pos_emb(query, key, cos, sin, layout=1, rotary_mode="half"):
    """fp32 计算域 + 末段单次 RTNE，与 kernel 舍入序列对齐；支持 partial rope。
    填充纪律：输入幅度 [-1,1]（官方 ST 实践）；fp16 大量级填充会撞 evaluator max_abs cap。"""
    outs = []
    for x in (query, key):
        RD = cos.shape[-1]
        xf = x.float()
        rot = x[..., :RD]
        c, s = cos.float(), sin.float()               # N 维=1，广播由 shape 自然完成
        emb = rot * c + _rotate(rot, rotary_mode) * s
        out = torch.cat((emb, xf[..., RD:]), dim=-1) if RD < x.shape[-1] else emb
        outs.append(out.to(x.dtype))
    return tuple(outs)
