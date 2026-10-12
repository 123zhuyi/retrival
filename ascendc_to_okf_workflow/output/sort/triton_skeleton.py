# P2 骨架 — sort（CANN ops-math 转化），sort 卡主路线（topk v3 姊妹）
#
# 语义：values,indices = sort(x, axis, descending, stable, y2_dtype)——沿 axis 全排列排序。
#   stable=True 铁律（golden `_require_stable` ST 只覆盖 stable=True）；descending 双向；
#   任意轴（非尾轴 movedim 归尾）；y2 indices{i32,i64}；0-dim->indices 全0；axis 长==1->恒等。
# 源锚点：src/ops-math/math/sort/（12 SCHID 分派 + arch35 radix/merge/small-axis 四族 + tiling）
# 算法路线（mapping.md §1）：packed-key tl.sort ❌（C-027 vsort 仅 f16/f32，int64 编译失败）
#   → 迭代 max+argmax 选择（topk v3 同构）✅；f32 tl.sort 直排留为 ①门附带验证的候选 B。
# 结构：一行一 program（grid=M），**N 次发射**（每次并行取所有 M 行的第 j 列），选中位 torch
#   scatter 掩码。发射级（无 kernel 内动态循环——topk v2 scf.for 挂死类规避 507034）。
#########################################################################################
# ⚠️ sort 与 topk 的序方向语义切换（mapping §2 关键）：
#   迭代选择天然「每发射取当前最大」。要产出有序输出须按方向反选：
#   - 升序（descending=False）：应放...最小~次小~...→ 取**最小**先 → **反键** `~key`，
#     已选沉 **type-max**（反键后为最小值永不被选）；
#   - 降序（descending=True）：取最大先 → **正键**，已选沉 **type-min**。
#   **NaN 键=最大（CANN CANONICAL_NAN_KEY / golden 同构）**：升序反键后 NaN→最小→**先出排最前**；
#   这与 torch.sort 升序 NaN 排最后不同——②门将实测 torch.sort 对 NaN/升降序的真实尾排行为，
#   若与 key 约定不符则登记该 corner（见卡片「边界」）。
#########################################################################################

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
    """值 → int64 无符号保序全序键（大=优；NaN 最大；~key 即序反转）。golden 同构（sign-flip）。"""
    if DT == 0:      # f32
        b = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        u = tl.where(b >= 0x80000000, b ^ 0xFFFFFFFF, b ^ 0x80000000)
        u = tl.where(x == 0.0, 0x80000000, u)   # ±0 折叠为单一 0 键（torch 视 ±0 相等，稳定 tie 保序）
        return tl.where(x != x, 0xFFFFFFFF, u)
    elif DT == 1 or DT == 2:  # f16/bf16：16bit sign-flip
        b = x.to(tl.int16, bitcast=True).to(tl.int64) & 0xFFFF
        u = tl.where(b >= 0x8000, b ^ 0xFFFF, b ^ 0x8000)
        u = tl.where(x == 0.0, 0x8000, u)
        return tl.where(x != x, 0xFFFF, u)
    elif DT == 10:   # u64：^SIGN64 保序映射
        return x.to(tl.int64, bitcast=True) ^ (-0x8000000000000000)
    else:            # i8~i64：符号/零扩展
        return x.to(tl.int64)


@triton.jit
def _sort_col_kernel(x_ptr, vis_ptr, v_ptr, i_ptr, N, j, sv, si,
                     DESCENDING: tl.constexpr, DT: tl.constexpr, BLOCK_N: tl.constexpr):
    """一行一 program（grid=M）取第 j 列：从 x 取序向极值（升序=最小/降序=最大），左most tie（稳定）。
    已选位经 **vis 缓冲**（int32，0=未选/1=已选）在键域排除——沉 INT64_MIN 恒低于一切实键
    （含 ±inf/NaN），绝不与实数据碰撞（值域沉底 ±inf 会撞图中 +inf/+inf 与 NaN 极值键，CPU 探针
    定位的根因；本设计不修改 x 值，value 提取恒取原值）。无 kernel 内循环（发射级）。"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    msk = offs < N
    x = tl.load(x_ptr + pid.to(tl.int64) * N + offs, mask=msk, other=0)
    vis = tl.load(vis_ptr + pid.to(tl.int64) * N + offs, mask=msk, other=1)  # mask 外视为已选
    cur = _sortable_key(x, DT)            # 内含 ±0 折叠与 NaN=最大
    if not DESCENDING:
        cur = ~cur                       # 升序：反键 → 选最小先出；NaN 反键后最小 → 排末（torch 同构）
    zero64 = -(2**63)
    cur = tl.where((vis == 0) & msk, cur, zero64)            # 已选/越界槽 → 键域绝对最小（永不选）
    mx = tl.max(cur, axis=0)             # int64 reduce（标量化，topk 已证正确）
    m = tl.argmax(tl.where(cur == mx, BLOCK_N - offs, 0), axis=0)  # 左most 稳定tie；块内 lane=轴内下标
    v = tl.sum(tl.where(offs == m, x, 0))  # 取 m 处原值（x 未改，恒为原始值含 NaN）
    tl.store(v_ptr + pid.to(tl.int64) * sv + j, v.to(v_ptr.dtype.element_ty))
    tl.store(i_ptr + pid.to(tl.int64) * si + j, m.to(i_ptr.dtype.element_ty))


def sort(x, axis: int = -1, descending: bool = False, stable: bool = True,
         y2_dtype: torch.dtype = torch.int32):
    """沿 axis 全排列排序。对齐 GEIR Sort / aclnnSort。

    stable：忽略（骨架恒稳定输出=超集合法；ST 只覆盖 stable=True）；
    axis：[-x.dim(), x.dim())；descending：双向；y2_dtype：i32/i64（indices dtype）。
    发射级迭代：**N 次 kernel 发射**（每次 grid=M 并行取所有行第 j 列）+ N 次 host 同步 +
    scatter 掩码——性能两态（launch+sync 开销随 N 线性，正确性优先；大 N 需二段归并=边界受限，登记）。

    单 segment 覆盖 N≤BLOCK_N(=min 2048|next_pow2(N))，即官方 small-axis 主导区（≤512/2048）
    与 radix-one/more 中轴；N>2048 大尾轴二段选+归并**未在骨架实现**（mapping §2 登记边界受限）。
    """
    ndim = x.dim()
    axis = axis + ndim if axis < 0 else axis
    assert 0 <= axis < ndim, "axis 越界"
    N = x.shape[axis]
    idx_dt = y2_dtype if y2_dtype in (torch.int32, torch.int64) else torch.int32

    if not x.is_contiguous():
        x = x.contiguous()
    # 非尾轴 movedim 归尾；须 contiguous（topk 探针14 实证：reshape 保留 transpose 步幅→错读行）
    xt = x.movedim(axis, -1)
    M = xt.numel() // N if N > 0 else 0

    device = x.device
    values = torch.empty(xt.shape, dtype=x.dtype, device=device)
    indices = torch.empty(xt.shape, dtype=idx_dt, device=device)

    # 0-dim / 空 / N==0：values=self、indices=全 0（golden 同构）
    if xt.numel() == 0 or N == 0:
        if ndim == 0:  # 0-dim 特例：indices 全 0
            return x, torch.zeros((), dtype=idx_dt, device=device)
        return values, indices
    # axis 长==1：恒等（values=self、indices=全 0；aclnn axis1 恒等 shortcut 同构）
    if N == 1:
        idx = torch.zeros_like(values, dtype=idx_dt)
        return x.clone(), idx

    dt = _DT_CODE[x.dtype]
    BLOCK_N = min(2048, triton.next_power_of_2(N))
    vals2 = values.reshape(M, N)
    idxs2 = indices.reshape(M, N)

    assert N <= BLOCK_N, f"N={N} 超单 segment 上限（大尾轴二段归并=边界受限登记，未实现）"
    x_work = xt.reshape(M, N).contiguous()  # ⚠️ contiguous：非尾轴 movedim 须改正布局
    vis = torch.zeros((M, N), dtype=torch.int32, device=device)  # 0=未选
    for j in range(N):
        _sort_col_kernel[(M,)](x_work, vis, vals2, idxs2, N, j, N, N,
                               DESCENDING=descending, DT=dt, BLOCK_N=BLOCK_N)
        vis.scatter_(1, idxs2[:, j].to(torch.int64).unsqueeze(1), 1)  # 标记已选（键域排除，x 不改）
    return values.movedim(-1, axis).contiguous(), indices.movedim(-1, axis).contiguous()


if __name__ == "__main__":
    # P2.5 ②正确性门：对 CPU torch.sort(stable=True)（golden `_sort`）比对 values+indices
    import os
    import sys

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch_npu  # noqa: F401

    device = "npu:0"
    gen = torch.Generator().manual_seed(20260920)
    n_pass = n_fail = 0

    def check(tag, x, axis=-1, descending=False, idt=torch.int32):
        """ref=torch.sort(stable=True) 逐位比对：values（isnan-aware）+ indices 全等。
        唯一值条件达 for exact index（重复值仍须 values 全等=顺序正确）。"""
        global n_pass, n_fail
        xn = x.to(device)
        vals, idxs = sort(xn, axis=axis, descending=descending, y2_dtype=idt)
        vals, idxs = vals.cpu(), idxs.cpu().to(torch.int64)
        ref_v, ref_i = torch.sort(x, dim=axis, descending=descending, stable=True)
        ok = (vals.shape == ref_v.shape)
        if ok:
            if x.is_floating_point():
                veq = (vals == ref_v) | (torch.isnan(vals) & torch.isnan(ref_v))
                ok = bool(veq.all())
            else:
                ok = bool(torch.equal(vals, ref_v))
        if ok:
            ok = bool(torch.equal(idxs, ref_i.to(torch.int64)))
        n_pass += ok; n_fail += (not ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:46s} shape={tuple(x.shape)} axis={axis} desc={descending} idt={str(idt)[6:]}", flush=True)

    f32 = lambda *s: torch.randn(*s, generator=gen)
    # —— 浮点族：升/降序 × N 形态 ——
    check("f32 尾轴 N=64 升序", f32(8, 64))
    check("f32 尾轴 N=64 降序", f32(8, 64), descending=True)
    check("f32 尾轴 N=32 idt=i64", f32(16, 32), idt=torch.int64)
    check("f32 N=700 非2幂 升序", f32(4, 700))
    check("f32 中间轴 3D N=16", f32(8, 16, 32), axis=1)
    check("f32 首轴 N=16", f32(16, 8), axis=0)
    check("f16 尾轴 N=32 升序", f32(8, 32).half())
    check("bf16 尾轴 N=32 降序", f32(8, 32).bfloat16(), descending=True)
    # —— 整型族（语义面：AiCore 原生仅浮点，Triton 自然全 dtype）——
    for dtt in (torch.int8, torch.int16, torch.int32, torch.int64):
        lo = 0 if dtt is torch.uint8 else -100
        check(f"{str(dtt)[6:]} 尾轴 N=32 升序", torch.randint(lo, 100, (8, 32), generator=gen, dtype=dtt))
    check("u64 尾轴 N=16 升序 idt=i64", torch.randint(0, 2**40, (8, 16), generator=gen, dtype=torch.int64).to(torch.uint64), idt=torch.int64)
    # —— golden 四注入模式（重复值稳定序：values 比对=顺序正确）——
    dpat = torch.tensor([2., 1., 1., 3., 2., 1., 3., 1.])
    check("duplicate 模式 N=8 升序", dpat.expand(8, 8).contiguous())
    check("duplicate 模式 N=8 降序", dpat.expand(8, 8).contiguous(), descending=True)
    szp = torch.tensor([-0., 0., 0., -0., 1., -1., 0., -0.])
    check("signed_zero 模式 N=8 升序", szp.expand(8, 8).contiguous())
    npp = torch.tensor([3., float('nan'), 1., float('nan'), 2., 2., -1., 0.])
    check("nan 模式 N=8 升序", npp.expand(8, 8).contiguous())
    check("nan 模式 N=8 降序", npp.expand(8, 8).contiguous(), descending=True)
    ifp = torch.tensor([float('inf'), 3., float('-inf'), 0., 3., -0., 1., -1.])
    check("infinity 模式 N=8 升序", ifp.expand(8, 8).contiguous())
    check("infinity 模式 N=8 降序", ifp.expand(8, 8).contiguous(), descending=True)
    # —— 特例 ——
    v0, i0 = sort(f32(4, 8).to(device), axis=0)
    okc = (i0.shape == (4, 8)) and v0.shape == (4, 8)
    n_pass += 0;  # 首轴 shape 检查交由上方 check

    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    sys.exit(1 if n_fail else 0)