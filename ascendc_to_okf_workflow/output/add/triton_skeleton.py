"""
add Triton 骨架（P2 产出，可直接翻译为 @triton.jit）

对应 Ascend C：Triton-Ascend-KernelWiki/.../ops-math/math/add/
  - op_kernel/arch35/add_dag.h            四类计算 DAG（本骨架覆盖 WithCast 分支）
  - op_api/aclnn_add.cpp                  alpha 四策略（本骨架一律单 kernel 融合）
  - op_host/arch35/add_tiling_arch35.cpp  dtype 分派（本骨架由指针 dtype 编译期推导）

语义（benchmark 3_Add.py 为标杆）：
  out = x + alpha * y          # alpha 先乘 y 再加 x（与 torch.add 顺序一致）
  - 中间 fp32 计算，RTNE 回写 out dtype（对齐 add_dag.h:48-51 AddWithCastCompute）
  - 任意 numpy 广播（benchmark 域：1D~4D，等形/[M,1]/[1,N]/[N]/[1,C,1,1]/4D×2D）
  - alpha 为 runtime fp32 标量（benchmark json 给出，大量 ≠1）

布局决策：
1. 输出展平 1-D，grid = AIV 物理核数 + program 内 NUM_BLOCKS grid-stride 循环
   （ascendc_to_okf 固定范式，同 mhc_sinkhorn/mhc_pre）。
2. SAME_SHAPE=1 等形直通（零整数除法，覆盖大 2D 性能 case）；
   SAME_SHAPE=0 通用广播：输出线性 offs 逐维 div/mod 分解坐标，
   按各输入 expand 后 stride 折算输入偏移（广播维 stride=0 自动复用）。
   最多 4 维（benchmark 域）；扩维只需加 stride 参数对 + 一段静态分支。
3. dtype 不分支：load → .to(f32) → alpha*y + x → .to(out_dtype) → store，
   fp32 时 cast 恒等，天然覆盖 benchmark 的 fp32/fp16/bf16 三种同型组合。
"""
import torch
import triton
import triton.language as tl

try:
    import triton.runtime.driver as driver
    NUM_CORE = driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"]
except Exception:
    NUM_CORE = 48  # 按目标平台物理核数填写


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}),
        triton.Config({"BLOCK": 2048}),
        triton.Config({"BLOCK": 4096}),
        triton.Config({"BLOCK": 8192}),
    ],
    key=["NUMEL", "SAME_SHAPE"],
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    alpha,                     # fp32 runtime 标量（benchmark json 的 attr）
    NUMEL,
    # 各输入 expand 到 out_shape 后的 stride（元素单位，广播维=0），恒 4 组
    xs0, xs1, xs2, xs3,
    ys0, ys1, ys2, ys3,
    # out（连续）的外维 stride；最内维 stride=1 隐含。NDIM<=4 时使用前 NDIM-1 个
    os0, os1, os2,
    NDIM: tl.constexpr,
    SAME_SHAPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(NUMEL, BLOCK)

    for blk in range(pid, NUM_BLOCKS, NUM_CORE):
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < NUMEL

        if SAME_SHAPE:
            x_off = offs                      # 等形直通
            y_off = offs
        else:
            # 输出线性下标 → 逐维坐标 → 按 expand stride 折算输入偏移
            rem = offs
            x_off = tl.zeros([BLOCK], dtype=tl.int64)
            y_off = tl.zeros([BLOCK], dtype=tl.int64)
            if NDIM == 4:
                c = rem // os0
                rem = rem - c * os0
                x_off += c * xs0
                y_off += c * ys0
            if NDIM >= 3:
                c = rem // os1
                rem = rem - c * os1
                x_off += c * xs1
                y_off += c * ys1
            if NDIM >= 2:
                c = rem // os2
                rem = rem - c * os2
                x_off += c * xs2
                y_off += c * ys2
            x_off += rem * xs3                  # 最内维（out stride=1）
            y_off += rem * ys3

        x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        y = tl.load(y_ptr + y_off, mask=mask, other=0.0)
        # AddWithCastCompute：升 fp32 → alpha·y + x → RTNE 回写（add_dag.h:48-53）
        res = x.to(tl.float32) + alpha * y.to(tl.float32)
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def _pad4(strides, ndim):
    """右对齐补齐到 4 维（外维补 0，kernel 内被静态分支裁掉）"""
    st = [0, 0, 0, 0]
    for i, s in enumerate(strides):
        st[4 - ndim + i] = s
    return st


def add(x, y, alpha=1.0):
    """
    x, y: 同 dtype 张量（fp32/fp16/bf16），torch.add(x, y, alpha=alpha) 语义。
    返回 out（broadcast 后形状，dtype = x.dtype）。
    """
    assert x.dtype == y.dtype and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
    x = x.contiguous()
    y = y.contiguous()
    out_shape = torch.broadcast_shapes(x.shape, y.shape)
    NUMEL = out_shape.numel()
    xe = x.expand(out_shape)      # 广播维 stride=0
    ye = y.expand(out_shape)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    ndim = len(out_shape)
    same_shape = (x.numel() == NUMEL) and (y.numel() == NUMEL)

    if same_shape:
        # 展平后等形直通（含 1D）；stride 参数占位即可
        add_kernel[(NUM_CORE,)](
            x.reshape(-1), y.reshape(-1), out.reshape(-1),
            float(alpha), NUMEL,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
            NDIM=1, SAME_SHAPE=True,
        )
    else:
        xs = _pad4(xe.stride(), ndim)
        ys = _pad4(ye.stride(), ndim)
        # out 连续 ⇒ 外维 stride：os0=外维总跨（D1·D2·D3），os1=D2·D3，os2=D3
        os0 = out_shape[1] * out_shape[2] * out_shape[3] if ndim == 4 else 0
        os1 = out_shape[2] * out_shape[3] if ndim >= 3 else 0
        os2 = out_shape[-1] if ndim >= 2 else 0
        add_kernel[(NUM_CORE,)](
            xe, ye, out,
            float(alpha), NUMEL,
            xs[0], xs[1], xs[2], xs[3],
            ys[0], ys[1], ys[2], ys[3],
            os0, os1, os2,
            NDIM=ndim, SAME_SHAPE=False,
        )
    return out
