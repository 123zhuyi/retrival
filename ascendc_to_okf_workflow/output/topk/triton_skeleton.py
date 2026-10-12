# P2 骨架 — topk / top_k_v2（CANN ops-math 转化）v3
#
# 语义：沿 dim 选 k 个极值，largest/sorted/indices_dtype 属性；值 dtype=x，索引 dtype=indices_dtype。
# 源锚点：src/ops-math/math/top_k_v2/（apt tiling key 全表 + arch35 radix/bitonic/merge 三族 + tiling 双文件）
# 算法路线演进：
#   v1 packed-key tl.topk → 废弃（vsort 仅 f16/f32，C-027）
#   v2 kernel 内动态 k 环（迭代 max+argmax 选择）→ 编译 OK 但 (BLOCK=512,k=32) 设备侧挂死
#       （vector core timeout 507034，scf.for 动态界+归约后端 loop 代码生成缺陷类）
#   v3 **发射级迭代**：wrapper 循环 j∈[0,k)，每次发射无环单选 kernel（直列码，probe6/7 原语
#       实证：int64 max-reduce + int32 flag-argmax 正确），选中位经 torch scatter 屏蔽。
#       k 次发射 + k 次 host 同步 = 性能两态（~ms 级 launch+sync 开销），正确性优先。
# 全序键：值位模式零扩展 int64 无符号保序键（浮点 sign-flip：负 ~b / 正 b^SIGN，NaN=全 1；
#   u64 ^SIGN64）；largest=False ~key 序反转（NaN 排最后）。golden _radix_keys 同构。
#   ⚠️ v2 首战抓获：负值分支 XOR 掩码必须全 1（~b），0x7FFFFFFF 笔误与正值全域碰撞。
# tie：严格比较稳定 → 左most（golden SmallBitonicSort 同构）；argmax 键=BLOCK-offs 左most。
# 大 N：两行段（stage1 tile 局选候选 + stage2 归并，官方 MULT_CORE_OPTIM 同构），同为发射级迭代。
# 已知角落（登记卡片）：屏蔽填充 type-min/-inf 与实键 tie 时（行含 NaN/∓inf 且 k 触及该值域）
#   左most 可能误选屏蔽槽——仅影响 k 触及极值域的语义区 case。
# 纪律：无 %（pid 解码 // 与 a-(a//b)*b）；stride 运行传参（C-017）；LARGEST/DT constexpr（C-023）；
#   无动态循环（v2 挂死类规避）；无 atomic（C-024 不命中）；grid=M 一行一 program（C-011 偏离登记：
#   行任务重串行，一行为一 program 忠实映射；发射级迭代下 persistent 无收益）。

import torch
import triton
import triton.language as tl

# DT constexpr 码：0=f32 1=f16 2=bf16 3=i8 4=u8 5=i16 6=u16 7=i32 8=u32 9=i64 10=u64
_DT_CODE = {
    torch.float32: 0, torch.float16: 1, torch.bfloat16: 2,
    torch.int8: 3, torch.uint8: 4, torch.int16: 5, torch.uint16: 6,
    torch.int32: 7, torch.uint32: 8, torch.int64: 9, torch.uint64: 10,
}


@triton.jit
def _sortable_key(x, DT: tl.constexpr):
    """值 → int64 无符号保序全序键（大=优；NaN 最大；~key 即序反转）。
    golden _radix_keys 同构：负值分支 ~bits（全 1 XOR），正值分支 bits^SIGN。"""
    if DT == 0:      # f32
        b = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        u = tl.where(b >= 0x80000000, b ^ 0xFFFFFFFF, b ^ 0x80000000)
        return tl.where(x != x, 0xFFFFFFFF, u)
    elif DT == 1 or DT == 2:  # f16/bf16：16bit sign-flip
        b = x.to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
        u = tl.where(b >= 0x8000, b ^ 0xFFFF, b ^ 0x8000)
        return tl.where(x != x, 0xFFFF, u)
    elif DT == 10:   # u64：^SIGN64 保序映射（golden 同构）
        return x.to(tl.int64, bitcast=True) ^ (-0x8000000000000000)
    else:            # i8~i64：符号/零扩展（.to 按 signedness；i64 恒等）
        return x.to(tl.int64)


@triton.jit
def _pick_kernel(
    x_ptr, v_ptr, i_ptr, N, j, sv, si,
    LARGEST: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 无环单选：一行一 program（grid=M），输出该行第 j 优（列偏移 j 运行传参）
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64) * N + offs, mask=msk, other=0)
    cur = _sortable_key(x, DT)
    if not LARGEST:
        cur = ~cur                      # 序反转（~ = 序取反双射；NaN→最小排最后）
    sel = tl.where(msk, cur, tl.full((BLOCK_N,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)            # int64 reduce（probe7 证正确；标量化仅性能）
    # 左most：键=BLOCK_N-offs（大=左）；掩码槽 sel=INT64_MIN 不构成竞争（实键 ≥ 0 域或 alive 语义由
    # 发射级 x_work 屏蔽保证——已选位已被 wrapper 改写为 type-min 沉底）
    m = tl.argmax(tl.where(sel == mx, BLOCK_N - offs, 0), axis=0)
    v = tl.sum(tl.where(offs == m, x, 0))  # 取 m 处值（x[m] 标量索引不支持）
    tl.store(v_ptr + pid.to(tl.int64) * sv + j, v.to(v_ptr.dtype.element_ty))
    tl.store(i_ptr + pid.to(tl.int64) * si + j, m.to(i_ptr.dtype.element_ty))


@triton.jit
def _pick_tile_kernel(
    x_ptr, wv_ptr, wi_ptr, N, num_tiles, j, kp,
    LARGEST: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 两行段①单选：一 (row,tile) 一 program（grid=M×T）；第 j 优候选写 ws（携带全局索引）
    pid = tl.program_id(0)
    r = pid // num_tiles
    tb = pid - r * num_tiles  # C-001：禁 %
    offs = tb * BLOCK_N + tl.arange(0, BLOCK_N)
    msk = offs < N
    x = tl.load(x_ptr + r.to(tl.int64) * N + offs, mask=msk, other=0)
    cur = _sortable_key(x, DT)
    if not LARGEST:
        cur = ~cur
    sel = tl.where(msk, cur, tl.full((BLOCK_N,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)
    m = tl.argmax(tl.where(sel == mx, BLOCK_N - offs, 0), axis=0)  # m=局部 lane（argmax 返块内下标）
    # 取值必须用局部 lane 比较（offs 是全局下标，offs==m 对 tb>0 恒假 → v=0 事故，smoke5 两行段抓获）
    v = tl.sum(tl.where(tl.arange(0, BLOCK_N) == m, x, 0))
    gidx = (tb * BLOCK_N + m).to(tl.int64)
    base = pid.to(tl.int64) * kp
    tl.store(wv_ptr + base + j, v.to(wv_ptr.dtype.element_ty))
    tl.store(wi_ptr + base + j, gidx)


@triton.jit
def _pick_merge_kernel(
    wv_ptr, wi_ptr, wpos_ptr, v_ptr, i_ptr, num_cand, j, sv, si,
    LARGEST: tl.constexpr, DT: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # 两行段②单选：一行一 program（grid=M）；候选第 j 优 → 值+**全局索引**（wi gather）
    # wpos 侧输出候选位（供 wrapper scatter 屏蔽 ws_work）
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    msk = offs < num_cand
    x = tl.load(wv_ptr + pid.to(tl.int64) * num_cand + offs, mask=msk, other=0)
    cur = _sortable_key(x, DT)
    if not LARGEST:
        cur = ~cur
    sel = tl.where(msk, cur, tl.full((BLOCK_C,), -(2**63), tl.int64))
    mx = tl.max(sel, axis=0)
    m = tl.argmax(tl.where(sel == mx, BLOCK_C - offs, 0), axis=0)
    v = tl.sum(tl.where(offs == m, x, 0))
    gi = tl.load(wi_ptr + pid.to(tl.int64) * num_cand + m)
    tl.store(v_ptr + pid.to(tl.int64) * sv + j, v.to(v_ptr.dtype.element_ty))
    tl.store(i_ptr + pid.to(tl.int64) * si + j, gi.to(i_ptr.dtype.element_ty))
    tl.store(wpos_ptr + pid.to(tl.int64) * sv + j, m)


def _mask_val(dtype, largest):
    """发射级迭代的屏蔽填充值：largest=True 填 type-min（沉底）；False 填 type-max。"""
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return float("-inf") if largest else float("inf")
    info = torch.iinfo(dtype)
    return info.min if largest else info.max


def _mask_out(xs, sel, mv, dt):
    """发射级屏蔽：选中位改写为沉底值。u64 NPU scatter_ 不支持（smoke5/6 161002 抓获）→
    走 int64 位视图屏蔽（u64^sign 后 int64 保序，屏蔽值即 int64 min/max）。
    NPU scatter_ 当 index 多列(>1)时 src 须与 index 同维（0 维标量被拒，smoke7 捕获）→
    广播为 index 形状张量。"""
    sel64 = sel.to(torch.int64)
    if sel64.dim() == 1:            # 单行踩一列：索引需与 self 同维（2D）
        sel64 = sel64.unsqueeze(1)
    if dt == torch.uint64:
        # u64 沉底值在位视图 int64 下的同构：u64 值域位模式原样重释（2^63 起视为负）。
        # ⚠️ NPU 不能 torch.full(uint64, 大 int)（int64 提升冲突 161002）→ 全程 int64 视图走 scatter。
        xw = xs.view(torch.int64)
        mv64 = (mv - 2**64) if mv >= 2**63 else mv
        xw.scatter_(1, sel64, torch.full(sel64.shape, mv64, dtype=torch.int64, device=xs.device))
    else:
        src = torch.full(sel64.shape, mv, dtype=dt, device=xs.device)  # index 同维 src 广播
        xs.scatter_(1, sel64, src)


def topk(x, k, dim: int = -1, largest: bool = True, sorted: bool = True,
         indices_dtype: torch.dtype = torch.int32):
    """沿 dim 选 top-k。对齐 GEIR TopKV2（values=x dtype；indices=indices_dtype；sorted 恒有序=超集）。

    k: int 或 0D tensor；dim: [-x.dim(), x.dim())；indices_dtype: i32/i64。
    发射级迭代：k 次 kernel 发射 + k 次 host 同步（选中位回读做 x_work 屏蔽）——性能两态
    （launch+sync 开销 ~ms 级），正确性优先（v3 规避 kernel 内动态循环挂死类）。
    已知角落：行含 NaN/∓inf 且 k 触及该值域时屏蔽槽 tie 可能误选（卡片登记）。
    """
    k_val = int(k.reshape(-1)[0].item()) if isinstance(k, torch.Tensor) else int(k)
    ndim = x.dim()
    dim = dim + ndim if dim < 0 else dim
    assert 0 <= dim < ndim, "dim 越界（golden _normalize_axis 同构）"
    N = x.shape[dim]
    assert 0 <= k_val <= N, f"k={k_val} 越界 [0, {N}]（golden ValueError 同构）"
    idx_dt = indices_dtype if indices_dtype in (torch.int32, torch.int64) else torch.int32

    if not x.is_contiguous():
        x = x.contiguous()
    xt = x.movedim(dim, -1)
    M = xt.numel() // N if N > 0 else 0

    device = x.device
    out_shape = list(xt.shape)
    out_shape[-1] = k_val
    values = torch.empty(out_shape, dtype=x.dtype, device=device)  # xt 轴序（moved 布局），返回前归位
    indices = torch.empty(out_shape, dtype=idx_dt, device=device)
    if M == 0 or k_val == 0:
        # 输出归位：xt 轴序 → 原 x 轴序（dim≠-1 时 movedim 回去；smoke5 中间轴/首轴 FAIL 抓获）
        return values.movedim(-1, dim), indices.movedim(-1, dim)

    dt = _DT_CODE[x.dtype]
    mv = _mask_val(x.dtype, largest)
    BLOCK_N = min(2048, triton.next_power_of_2(N))
    vals2 = values.reshape(M, k_val)
    idxs2 = indices.reshape(M, k_val)

    if N <= BLOCK_N:
        x_work = xt.reshape(M, N).contiguous()  # ⚠️ 须 contiguous 化：reshape 保留 transpose 步幅(1,32)，
        # clone() 复制数据不改正布局 → kernel 按行主序读错行（dim≠-1 smoke5/6 FAIL，探针14 实证）
        for j in range(k_val):
            _pick_kernel[(M,)](x_work, vals2, idxs2, N, j, k_val, k_val,
                               LARGEST=largest, DT=dt, BLOCK_N=BLOCK_N)
            _mask_out(x_work, idxs2[:, j], mv, x.dtype)
    else:
        # 两行段：stage1 每 tile 局选 kp 候选（kp 发射），stage2 归并（k 发射）
        num_tiles = triton.cdiv(N, BLOCK_N)
        kp = min(k_val, BLOCK_N)
        num_cand = num_tiles * kp
        BLOCK_C = min(2048, triton.next_power_of_2(num_cand))
        assert num_cand <= BLOCK_C, f"候选数 {num_cand} 超上限（sort_and_top_k 域，登记）"
        x_work = xt.reshape(M, N).contiguous()  # 同上：非尾轴 movedim 需 contiguous 化
        wv = torch.empty(M, num_tiles, kp, dtype=x.dtype, device=device)
        wi = torch.empty(M, num_tiles, kp, dtype=torch.int64, device=device)
        for j in range(kp):
            _pick_tile_kernel[(M * num_tiles,)](x_work, wv, wi, N, num_tiles, j, kp,
                                                LARGEST=largest, DT=dt, BLOCK_N=BLOCK_N)
            # 屏蔽已选：wi[:, :, j] = 各 tile 本块选中的全局索引（M, num_tiles）按行 scatter
            gsel = wi[:, :, j].reshape(M, num_tiles)
            _mask_out(x_work, gsel, mv, x.dtype)
        ws_work = wv.reshape(M, num_cand).clone()
        wpos = torch.empty(M, k_val, dtype=torch.int32, device=device)
        for j in range(k_val):
            _pick_merge_kernel[(M,)](ws_work, wi.reshape(M, num_cand), wpos,
                                     vals2, idxs2, num_cand, j, k_val, k_val,
                                     LARGEST=largest, DT=dt, BLOCK_C=BLOCK_C)
            pos = wpos[:, j]
            _mask_out(ws_work, pos, mv, x.dtype)
    return values.movedim(-1, dim).contiguous(), indices.movedim(-1, dim).contiguous()


if __name__ == "__main__":
    # P2.5 ②正确性门：dtype 族 × largest × k 档 × N 形态 × 注入模式，对 CPU torch.topk
    import os
    import sys

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401

    device = "npu:0"
    gen = torch.Generator().manual_seed(20260918)
    n_pass = n_fail = 0

    def semantic_ok(x_cpu, vals, idxs, k, dim, largest):
        """golden 语义比较器：索引合法 + 无重复 + gather 回值等 + 按全序键有序"""
        axis = dim if dim >= 0 else dim + x_cpu.dim()
        if k == 0:
            return vals.numel() == 0
        xs = x_cpu.movedim(axis, -1).reshape(-1, x_cpu.shape[axis])
        vs = vals.movedim(axis, -1).reshape(-1, k)
        ii = idxs.movedim(axis, -1).reshape(-1, k).to(torch.int64)
        N = xs.shape[1]
        if not bool(((ii >= 0) & (ii < N)).all()):
            return False
        if k > 1:
            srt = torch.sort(ii, dim=-1).values
            if bool((srt[:, 1:] == srt[:, :-1]).any()):
                return False
        g = torch.gather(xs, 1, ii.clamp(0, N - 1))
        if x_cpu.is_floating_point():
            if not bool(((g == vs) | (torch.isnan(g) & torch.isnan(vs))).all()):
                return False
        else:
            if not bool((g == vs).all()):
                return False
        if k > 1:  # 有序性：输出值按全序键降（largest）/升（False）
            keys = []
            for r in range(xs.shape[0]):
                row = g[r]
                if x_cpu.is_floating_point():
                    w = row.element_size() * 8
                    vd = torch.int16 if w == 16 else torch.int32
                    ab = (1 << w) - 1
                    sb = 1 << (w - 1)
                    bits = row.contiguous().view(vd).to(torch.int64) & ab
                    u = torch.where((bits & sb) != 0, bits ^ ab, bits ^ sb)
                    u = torch.where(torch.isnan(row), ab, u)
                    keys.append(u)
                else:
                    keys.append(row.to(torch.int64))
            kk = torch.stack(keys)
            d = kk[:, 1:] - kk[:, :-1]
            return bool((d <= 0).all()) if largest else bool((d >= 0).all())
        return True

    def check(tag, x, k, dim=-1, largest=True, idt=torch.int32, exact=True):
        global n_pass, n_fail
        xn = x.to(device)
        vals, idxs = topk(xn, k, dim=dim, largest=largest, indices_dtype=idt)
        ok = vals.shape == idxs.shape
        if ok:
            ok = semantic_ok(x, vals.cpu(), idxs.cpu(), int(k), dim, largest)
        axis = dim if dim >= 0 else dim + x.dim()
        if ok and exact and 1 <= int(k) <= 32:
            xs = x.movedim(axis, -1).reshape(-1, x.shape[axis])
            row = xs[0]
            if x.is_floating_point():
                can = (not torch.isnan(row).any()) and len(torch.unique(row)) == row.numel()
            else:
                can = len(torch.unique(row)) == row.numel()
            if can:
                ref = torch.topk(x, int(k), dim=axis, largest=largest)
                ok = torch.equal(idxs.cpu().to(torch.int64), ref.indices.to(torch.int64))
        n_pass += ok
        n_fail += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:42s} shape={tuple(x.shape)} k={k} dim={dim} lg={largest}", flush=True)

    f32 = lambda *s: torch.randn(*s, generator=gen)
    # —— 浮点族：largest 双向 / k 档 / N 形态 ——
    check("f32 尾轴 k=8 N=1536", f32(32, 1536), 8)
    check("f32 尾轴 k=8 N=1536 lg=False", f32(32, 1536), 8, largest=False)
    check("f32 k=1 N=700 非2幂", f32(16, 700), 1)
    check("f32 k=32 N=64 小", f32(64, 64), 32)
    check("f32 k=33 N=300 大k", f32(4, 300), 33, exact=False)
    check("f32 k=200 N=1536 大k", f32(4, 1536), 200, exact=False)
    check("f32 k=N 全选 N=64", f32(8, 64), 64, exact=False)
    check("f32 N=8192 两行段 k=5", f32(4, 8192), 5)
    check("f32 N=8192 两行段 k=5 lg=False", f32(4, 8192), 5, largest=False)
    check("f16 尾轴 k=8 N=1024", f32(16, 1024).half(), 8)
    check("bf16 尾轴 k=8 N=1024 idt=i64", f32(16, 1024).bfloat16(), 8, idt=torch.int64)
    check("f32 中间轴 3D k=4", f32(8, 16, 32), 4, dim=1)
    check("f32 首轴 k=3", f32(16, 32), 3, dim=0)
    check("f32 4D 中间轴 k=5", f32(4, 8, 16, 32), 5, dim=2)
    # —— 整型族 ——
    for dtt in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        lo = 0 if dtt in (torch.uint8,) else -100  # 无符号 dtype 禁负区间（randint 越界即抛，smoke5 抓获）
        check(f"{str(dtt)[6:]} 尾轴 k=7 N=1000", torch.randint(lo, 100, (16, 1000), generator=gen, dtype=dtt), 7)
    check("i64 大值域 k=5 N=777", torch.randint(-2**40, 2**40, (8, 777), generator=gen, dtype=torch.int64), 5)
    check("u64 k=5 N=777", torch.randint(0, 2**40, (8, 777), generator=gen, dtype=torch.int64).to(torch.uint64), 5)
    check("u8 全值域 k=4 N=300", torch.randint(0, 255, (8, 300), generator=gen, dtype=torch.uint8), 4)
    # —— golden 注入模式 ——
    dup = torch.tensor([7., 1., 7., 3., 7., 2., 8., 8.])
    check("duplicate 模式 k=4", dup.expand(8, 8).contiguous(), 4, exact=False)
    check("all_equal 模式 k=4", torch.full((8, 8), 3.0), 4, exact=False)
    nanp = torch.tensor([float('nan'), 3., float('nan'), 1., 2., 2., -1., 0.])
    check("nan 模式 k=4", nanp.expand(8, 8).contiguous(), 4, exact=False)
    check("nan 模式 lg=False k=4", nanp.expand(8, 8).contiguous(), 4, largest=False, exact=False)
    infp = torch.tensor([float('inf'), 3., float('-inf'), 0., 2., -2.])
    check("infinity 模式 k=3", infp.expand(8, 6).contiguous(), 3)
    sz = torch.tensor([-0., 0., 0., -0.])
    check("signed_zero 模式 k=2", sz.expand(8, 4).contiguous(), 2, exact=False)
    # —— k=0 / k tensor 输入 ——
    v0, i0 = topk(f32(4, 8).to(device), 0)
    ok0 = v0.shape[-1] == 0 and i0.shape[-1] == 0
    n_pass += ok0; n_fail += (not ok0)
    print(f"[{'PASS' if ok0 else 'FAIL'}] k=0 空输出", flush=True)
    vkt, ikt = topk(f32(4, 64).to(device), torch.tensor(3), dim=-1)
    okt = vkt.shape[-1] == 3
    n_pass += okt; n_fail += (not okt)
    print(f"[{'PASS' if okt else 'FAIL'}] k=tensor 输入", flush=True)

    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)
