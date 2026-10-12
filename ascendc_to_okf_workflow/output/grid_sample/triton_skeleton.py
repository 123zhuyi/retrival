"""P2 Triton 骨架 v1 — grid_sample（12_grid_sample，level3，2026-09-17）

语义锚点（extraction.md，全部 [SRC]/[GOLDEN] 对齐）：
- 反归一化两式、zeros/border/reflection 三 padding、bilinear 四权重、
  nearest=CAST_RINT（银行家舍入，非 floor(x+0.5)）、bicubic A=-0.75。
- 结构决策（mapping.md）：gather 类 Vector 算子；program=输出 tile [BC,BHW]；
  坐标每块算一次被 BC 通道复用；角点片上累加单次 store（消解 CANN 的
  标量 gather 逐 c 循环 + GM atomicAdd）。
- golden = F.grid_sample 直调（融合版）——smoke 直接对 torch 原生。

preflight（2026-09-17）：kernel 体无 %（flips 奇偶用 f-(f//2)*2）；mask：
load/store 用整型比较（C-002），权重清零等算术选择用 fp32 比较（C-009 分离
写法）；int64 仅标量基址（n·C·H·W 最大 561M < 2³¹，族纪律保留）；stride
运行时传参（C-017）；无常驻状态无 loop-carried tile（C-016 不适用）；
BC/BHW 恒 2 的幂 ≥16。
"""

import torch
try:
    import torch_npu  # NPU 设备注册（P2.5 正确性门教训，94 v1.1.1）
except ImportError:
    pass
import triton
import triton.language as tl


@triton.jit
def _rint(x):
    # 银行家舍入（round-half-to-even），等价 CANN CAST_RINT（nearest.h:1039）。
    # P2.5 编译门实证：libdevice.rint 触发 TritonToLinalg「unsupported control
    # flow」（bisect k3）——手写展开规避；flips 奇偶用 f−(f//2)*2（C-001 禁 %）。
    t = x + 0.5
    f = tl.floor(t)
    is_tie = t == f            # x 恰好是 k+0.5
    odd = (f - tl.floor(f * 0.5) * 2.0) != 0.0
    return tl.where(is_tie & odd, f - 1.0, f)


# ---------------- 坐标与 padding（fp32 域，全部被 BC 通道复用） ----------------

@triton.jit
def _unnormalize(g, size, AC: tl.constexpr):
    if AC:
        return (g + 1.0) * (0.5 * (size - 1.0))
    else:
        return (g + 1.0) * (0.5 * size) - 0.5


@triton.jit
def _reflect(x, twice_low, twice_high):
    # torch/CANN 反射折叠（grid_sampler_2d.h:280-354 同构）；fp32 域。
    # P2.5 编译门实证：运行时标量条件的早退触发 TritonToLinalg「unsupported
    # control flow」——改无分支形式：span 退化（≤0，即 size==1&ac 等）由
    # where 吸收，禁止 if 早退。
    minv = twice_low * 0.5
    span = (twice_high - twice_low) * 0.5
    span_ok = span > 0.0
    span_s = tl.where(span_ok, span, 1.0)
    t = tl.abs(x - minv)
    flips = tl.floor(t / span_s)
    extra = t - flips * span_s
    even = (flips - tl.floor(flips * 0.5) * 2.0) == 0.0  # C-001：禁 %
    res = tl.where(even, extra + minv, span_s - extra + minv)
    return tl.where(span_ok, res, x * 0.0 + minv)


@triton.jit
def _pad_coords(x, size, PAD: tl.constexpr, AC: tl.constexpr):
    # 浮点坐标域变换：仅 reflection 折叠；border 不在此夹取——
    # torch 语义是 tap 级夹取（smoke 实证：bilinear 浮点夹取恰好等价，
    # bicubic 浮点夹取错误 0.2 偏差——4 tap 位置各异，权重非仿射可约）。
    if PAD == 2:  # reflection：端点依赖 align_corners（CANN :242-249）
        if AC:
            return _reflect(x, 0.0, 2.0 * (size - 1.0))
        else:
            return _reflect(x, -1.0, 2.0 * size - 1.0)
    return x


# ---------------- 2D kernel（bilinear/nearest/bicubic × 三 padding） ----------------

@triton.jit
def _grid_sample_2d_kernel(
    x_ptr, grid_ptr, y_ptr,
    C, H_in, W_in, HW_out, NC, NHW,
    BC: tl.constexpr, BHW: tl.constexpr,
    MODE: tl.constexpr,   # 0=bilinear 1=nearest 2=bicubic
    PAD: tl.constexpr,    # 0=zeros 1=border 2=reflection
    AC: tl.constexpr,
):
    task = tl.program_id(0)
    hw_blk = task - (task // NHW) * NHW
    tmp = task // NHW
    c_blk = tmp - (tmp // NC) * NC
    n = tmp // NC
    n64 = n.to(tl.int64)

    hw = hw_blk * BHW + tl.arange(0, BHW)
    cc = c_blk * BC + tl.arange(0, BC)
    hw_m = hw < HW_out
    c_m = cc < C

    # grid[n, hw, 2] 交错 xy（x 沿 W、y 沿 H）
    gbase = n64 * (HW_out * 2) + hw * 2
    gx = tl.load(grid_ptr + gbase + 0, mask=hw_m, other=0.0).to(tl.float32)
    gy = tl.load(grid_ptr + gbase + 1, mask=hw_m, other=0.0).to(tl.float32)

    xf = _unnormalize(gx, W_in, AC)
    yf = _unnormalize(gy, H_in, AC)
    if not (PAD == 2 and MODE == 2):
        # bicubic+reflection 例外：不预折叠采样坐标——CANN bicubic 路径
        # （grid_sampler_2d_bicubic.h PerLoopCompute）权重用原始小数偏移、
        # 逐 tap 浮点折叠；预折叠会同时改权重与 tap，smoke 实测 0.2 偏差。
        xf = _pad_coords(xf, W_in, PAD, AC)
        yf = _pad_coords(yf, H_in, PAD, AC)

    HW_in = H_in * W_in
    out_base = n64 * (C * HW_in)  # 输入/输出同 N/C 划分；输出 HW=HW_out

    acc = tl.zeros((BC, BHW), dtype=tl.float32)  # 无 loop-carried dot，C-016 不适用

    if MODE == 1:
        # nearest：rint 银行家舍入（CANN CAST_RINT，nearest.h:1039-1040）
        xi = _rint(xf).to(tl.int32)
        yi = _rint(yf).to(tl.int32)
        if PAD == 0:  # zeros：rint 后的整数坐标判越界（torch 语义；
            # smoke 实证：用浮点 xf 判会把 x∈(−0.5,0) 误杀，max_diff≈1）
            inb = (xi >= 0) & (xi <= W_in - 1) & (yi >= 0) & (yi <= H_in - 1)
        else:
            xi = tl.minimum(tl.maximum(xi, 0), W_in - 1)
            yi = tl.minimum(tl.maximum(yi, 0), H_in - 1)
            inb = hw_m
        corner = yi * W_in + xi
        addr32 = cc[:, None] * HW_in + corner[None, :]  # 4e：索引张量保持 int32
        v = tl.load(x_ptr + out_base + addr32, mask=c_m[:, None] & inb[None, :],
                    other=0.0)  # int64 仅标量基址（gather 不接受 i64 索引）
        acc = v.to(tl.float32)
    else:
        x0f = tl.floor(xf)
        y0f = tl.floor(yf)
        x0 = x0f.to(tl.int32)
        y0 = y0f.to(tl.int32)
        NCORNER: tl.constexpr = 4 if MODE == 0 else 16
        for i in tl.static_range(NCORNER):
            dx = i - (i // 2) * 2          # 列偏移 0/1（bilinear）
            dy = i // 2
            if MODE == 2:                  # bicubic：4×4，dx/dy ∈ -1..2
                dx = i - (i // 4) * 4 - 1
                dy = i // 4 - 1
            cx_f = x0f + dx
            cy_f = y0f + dy
            if PAD == 2 and MODE == 2:
                # 逐 tap 浮点折叠（CANN bicubic ReflectCoordinates 同构）；
                # 权重用折叠前偏移（下行 tx/ty 用 cx_f 原值，故先算权重）
                pass
            cx = x0 + dx
            cy = y0 + dy
            if MODE == 0:
                wx = tl.where(dx == 0, 1.0 - (xf - x0f), xf - x0f)
                wy = tl.where(dy == 0, 1.0 - (yf - y0f), yf - y0f)
            else:
                tx = tl.abs(xf - cx_f)
                ty = tl.abs(yf - cy_f)
                wx = tl.where(tx <= 1.0, ((1.25 * tx) - 2.25) * tx * tx + 1.0,
                              ((-0.75 * tx + 3.75) * tx - 6.0) * tx + 3.0)
                wy = tl.where(ty <= 1.0, ((1.25 * ty) - 2.25) * ty * ty + 1.0,
                              ((-0.75 * ty + 3.75) * ty - 6.0) * ty + 3.0)
            w = wx * wy
            if PAD == 2 and MODE == 2:
                # 权重用折叠前 tap 位置算完后，再折叠 tap 浮点坐标取址
                cx_f2 = _reflect(cx_f, 0.0 if AC else -1.0,
                                 2.0 * (W_in - 1.0) if AC else 2.0 * W_in - 1.0)
                cy_f2 = _reflect(cy_f, 0.0 if AC else -1.0,
                                 2.0 * (H_in - 1.0) if AC else 2.0 * H_in - 1.0)
                cx = tl.floor(cx_f2).to(tl.int32)
                cy = tl.floor(cy_f2).to(tl.int32)
            if PAD == 0:  # zeros：角点 int 坐标越界 → 贡献 0（CANN :154-157 同语义）
                inb = (cx >= 0) & (cx <= W_in - 1) & (cy >= 0) & (cy <= H_in - 1)
            else:
                cx = tl.minimum(tl.maximum(cx, 0), W_in - 1)
                cy = tl.minimum(tl.maximum(cy, 0), H_in - 1)
                inb = hw_m
            corner = cy * W_in + cx
            addr32 = cc[:, None] * HW_in + corner[None, :]  # 同上：int32 索引
            v = tl.load(x_ptr + out_base + addr32, mask=c_m[:, None] & inb[None, :],
                        other=0.0)
            acc += v.to(tl.float32) * w[None, :]

    ybase = n64 * (C * HW_out) + hw_blk * BHW
    tl.store(y_ptr + ybase + cc[:, None] * HW_out + tl.arange(0, BHW)[None, :],
             acc.to(y_ptr.dtype.element_ty), mask=c_m[:, None] & hw_m[None, :])


# ---------------- 3D kernel（trilinear/nearest，套件仅 3 个小 case） ----------------

@triton.jit
def _grid_sample_3d_kernel(
    x_ptr, grid_ptr, y_ptr,
    C, D_in, H_in, W_in, DHW_out, NC, NDHW,
    BC: tl.constexpr, BDHW: tl.constexpr,
    MODE: tl.constexpr, PAD: tl.constexpr, AC: tl.constexpr,
):
    task = tl.program_id(0)
    dhw_blk = task - (task // NDHW) * NDHW
    tmp = task // NDHW
    c_blk = tmp - (tmp // NC) * NC
    n = tmp // NC
    n64 = n.to(tl.int64)

    dhw = dhw_blk * BDHW + tl.arange(0, BDHW)
    cc = c_blk * BC + tl.arange(0, BC)
    hw_m = dhw < DHW_out
    c_m = cc < C

    gbase = n64 * (DHW_out * 3) + dhw * 3
    gx = tl.load(grid_ptr + gbase + 0, mask=hw_m, other=0.0).to(tl.float32)
    gy = tl.load(grid_ptr + gbase + 1, mask=hw_m, other=0.0).to(tl.float32)
    gz = tl.load(grid_ptr + gbase + 2, mask=hw_m, other=0.0).to(tl.float32)

    xf = _pad_coords(_unnormalize(gx, W_in, AC), W_in, PAD, AC)
    yf = _pad_coords(_unnormalize(gy, H_in, AC), H_in, PAD, AC)
    zf = _pad_coords(_unnormalize(gz, D_in, AC), D_in, PAD, AC)

    DHW_in = D_in * H_in * W_in
    HW_in = H_in * W_in
    acc = tl.zeros((BC, BDHW), dtype=tl.float32)

    if MODE == 1:
        xi = _rint(xf).to(tl.int32)
        yi = _rint(yf).to(tl.int32)
        zi = _rint(zf).to(tl.int32)
        if PAD == 0:  # 同 2D：rint 后整数判越界
            inb = ((xi >= 0) & (xi <= W_in - 1) & (yi >= 0) & (yi <= H_in - 1)
                   & (zi >= 0) & (zi <= D_in - 1))
        else:
            xi = tl.minimum(tl.maximum(xi, 0), W_in - 1)
            yi = tl.minimum(tl.maximum(yi, 0), H_in - 1)
            zi = tl.minimum(tl.maximum(zi, 0), D_in - 1)
            inb = hw_m
        corner = (zi * H_in + yi) * W_in + xi
        addr32 = cc[:, None] * DHW_in + corner[None, :]  # int32 索引
        v = tl.load(x_ptr + n64 * (C * DHW_in) + addr32,
                    mask=c_m[:, None] & inb[None, :], other=0.0)
        acc = v.to(tl.float32)
    else:
        # trilinear：8 角点
        x0f = tl.floor(xf); y0f = tl.floor(yf); z0f = tl.floor(zf)
        x0 = x0f.to(tl.int32); y0 = y0f.to(tl.int32); z0 = z0f.to(tl.int32)
        fx = xf - x0f; fy = yf - y0f; fz = zf - z0f
        for i in tl.static_range(8):
            dx = i - (i // 2) * 2
            dy = (i // 2) - (i // 4) * 2
            dz = i // 4
            cx = x0 + dx; cy = y0 + dy; cz = z0 + dz
            w = (tl.where(dx == 0, 1.0 - fx, fx) * tl.where(dy == 0, 1.0 - fy, fy)
                 * tl.where(dz == 0, 1.0 - fz, fz))
            if PAD == 0:
                inb = ((cx >= 0) & (cx <= W_in - 1) & (cy >= 0) & (cy <= H_in - 1)
                       & (cz >= 0) & (cz <= D_in - 1))
            else:
                cx = tl.minimum(tl.maximum(cx, 0), W_in - 1)
                cy = tl.minimum(tl.maximum(cy, 0), H_in - 1)
                cz = tl.minimum(tl.maximum(cz, 0), D_in - 1)
                inb = hw_m
            corner = (cz * H_in + cy) * W_in + cx
            addr32 = cc[:, None] * DHW_in + corner[None, :]  # int32 索引
            v = tl.load(x_ptr + n64 * (C * DHW_in) + addr32,
                        mask=c_m[:, None] & inb[None, :], other=0.0)
            acc += v.to(tl.float32) * w[None, :]

    ybase = n64 * (C * DHW_out) + dhw_blk * BDHW
    tl.store(y_ptr + ybase + cc[:, None] * DHW_out + tl.arange(0, BDHW)[None, :],
             acc.to(y_ptr.dtype.element_ty), mask=c_m[:, None] & hw_m[None, :])


# ---------------- wrapper ----------------

def _next_pow2(n):
    p = 16
    while p < n:
        p *= 2
    return p


_MODE = {"bilinear": 0, "nearest": 1, "bicubic": 2}
_PAD = {"zeros": 0, "border": 1, "reflection": 2}


def grid_sample_triton(x, grid, mode="bilinear", padding_mode="zeros",
                       align_corners=False):
    y = torch.empty((x.shape[0], x.shape[1]) + tuple(grid.shape[1:-1]),
                    dtype=x.dtype, device=x.device)
    N, C = x.shape[0], x.shape[1]
    BC = _next_pow2(C)
    BC = BC if BC <= 32 else 32
    if x.dim() == 4:
        H_in, W_in = x.shape[2], x.shape[3]
        H_out, W_out = grid.shape[1], grid.shape[2]
        HW_out = H_out * W_out
        BHW = 256
        NC = (C + BC - 1) // BC
        NHW = (HW_out + BHW - 1) // BHW
        grid_launch = (N * NC * NHW,)
        _grid_sample_2d_kernel[grid_launch](
            x, grid, y, C, H_in, W_in, HW_out, NC, NHW,
            BC=BC, BHW=BHW, MODE=_MODE[mode], PAD=_PAD[padding_mode],
            AC=align_corners)
    else:
        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        DHW_out = grid.shape[1] * grid.shape[2] * grid.shape[3]
        BDHW = 256
        NC = (C + BC - 1) // BC
        NDHW = (DHW_out + BDHW - 1) // BDHW
        grid_launch = (N * NC * NDHW,)
        _grid_sample_3d_kernel[grid_launch](
            x, grid, y, C, D_in, H_in, W_in, DHW_out, NC, NDHW,
            BC=BC, BDHW=BDHW, MODE=_MODE[mode], PAD=_PAD[padding_mode],
            AC=align_corners)
    return y


if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)
    dev = torch.device("npu") if torch.npu.is_available() else torch.device("cpu")
    tol = {torch.float16: 2e-2, torch.float32: 1e-4}
    fails = 0
    for dtype in (torch.float16, torch.float32):
        for mode in ("bilinear", "nearest", "bicubic"):
            for pad in ("zeros", "border", "reflection"):
                for ac in (False, True):
                    x = (torch.rand(1, 8, 24, 24, device=dev) * 2 - 1).to(dtype)
                    g = torch.rand(1, 24, 24, 2, device=dev) * 2.6 - 1.3  # 超界覆盖
                    ref = F.grid_sample(x.float(), g.float(), mode=mode,
                                        padding_mode=pad, align_corners=ac)
                    out = grid_sample_triton(x, g, mode, pad, ac)
                    ok = torch.allclose(out.float(), ref.float(),
                                        atol=tol[dtype], rtol=tol[dtype])
                    if not ok:
                        fails += 1
                        d = (out.float() - ref.float()).abs().max().item()
                        print(f"FAIL {dtype} {mode}/{pad}/ac={ac} max_diff={d:.2e}")
    for dtype in (torch.float16, torch.float32):
        for mode in ("bilinear", "nearest"):
            for pad in ("zeros", "border", "reflection"):
                for ac in (False, True):
                    x = (torch.rand(1, 4, 12, 12, 12, device=dev) * 2 - 1).to(dtype)
                    g = torch.rand(1, 12, 12, 12, 3, device=dev) * 2.6 - 1.3
                    ref = F.grid_sample(x.float(), g.float(), mode=mode,
                                        padding_mode=pad, align_corners=ac)
                    out = grid_sample_triton(x, g, mode, pad, ac)
                    ok = torch.allclose(out.float(), ref.float(),
                                        atol=tol[dtype], rtol=tol[dtype])
                    if not ok:
                        fails += 1
                        d = (out.float() - ref.float()).abs().max().item()
                        print(f"FAIL 3D {dtype} {mode}/{pad}/ac={ac} max_diff={d:.2e}")
    print("SMOKE", "PASS (all 2D×3modes + 3D×2modes × 3pads × 2ac)" if fails == 0
          else f"{fails} FAILURES")
