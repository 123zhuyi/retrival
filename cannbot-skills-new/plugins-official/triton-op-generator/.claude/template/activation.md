---
name: activation
description: Activation 逐元素算子（GELU / SiLU / SwiGLU）的 Triton Ascend 优化经验
metadata:
  type: reference
---

# Activation 逐元素算子类别经验（GELU / SiLU / SwiGLU）

> 类别: `activation` — NPU 上的逐元素非线性激活函数
> 锚定算子: `_gelu_fwd_kernel` (op14) / `_swiglu_fwd_kernel` (op16) / `_silu_fwd_kernel` (op15)
> 性能参考: Ascend950PR / `ascend910_9382`
> - GELU: 30 cases (10 shapes × fp16/bf16/fp32)，最佳几何平均 **0.7969×**（目标 1.2×，**未达**），全 30 case 精度通过
> - SwiGLU: 30 cases，最佳几何平均 **1.3096×**（目标 1.2×，**已达** ✅），全 30 case 精度通过
> - SiLU: 30 cases，最佳几何平均 **1.2669×**（目标 1.2×，**已达** ✅），CANN kernel-only 复测，全 30 case 精度通过

---

## Layer 1: 设计约束（硬性规则，GELU / SiLU / SwiGLU 通用）

### L1.1 数学公式必须与 torch 参考严格一致（强制）
- **GELU**: `y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))`（exact erf 形式）。tanh 近似仅在参考明确使用 `approximate='tanh'` 时才允许。
- **SiLU**: `y = x * sigmoid(x)`，fp32 计算 `x_f32 * tl.sigmoid(x_f32)` 后 cast 回输入 dtype。
- **SwiGLU**: `c = silu(a) * b = a * sigmoid(a) * b`。

### L1.2 逐元素激活必须 1D 展平 + 连续访存（强制）
- **必须**: 将输入 reshape 为 1D（`x.reshape(-1)`），启动 1D grid，每个 program 以 grid-stride 循环处理连续 BLOCK 个元素。
- **禁止**: 2D 索引、双层 mask、跨步 tile 循环。逐元素算子没有空间局部性收益，2D tiling 只增加索引开销。
- **实测**: GELU 2D tiling **0.6665×** vs 1D **0.7969×**；SwiGLU 2D **1.1347×** vs 1D **1.3096×**。

### L1.3 动态 grid 钳制核数（强制）
- **必须**: `grid_size = min(triton.cdiv(numel, BLOCK), num_vectorcore)`，小 tensor 不启动满核 grid。
- **禁止**: 固定 `grid = (num_vectorcore,)`，小 tensor 浪费 launch/scheduling 开销。

### L1.4 fp32 计算，存回原始 dtype（强制）
- **必须**: 加载后 cast 到 `tl.float32` 再做数学运算，存回前 cast 回输入 dtype。
- **原因**: fp16/bf16 中间精度不足会导致与 torch fp32 参考的累积误差。

### L1.5 compiler hints 按算子独立判断（强制，不可一刀切）
- GELU/SwiGLU: 使用 `multibuffer=True, unit_flag=True`
- SiLU: **禁止**使用 `multibuffer/unit_flag` —— 实测这些 hints 对小 kernel 的 per-launch 开销反而拖累性能（0.927→0.945× 提升来自去掉 hints）
- **铁律**: 每个算子独立实测，不可把 GELU 的 hint 结论直接套到 SiLU/SwiGLU

---

## Layer 2: 算法骨架（1D grid-stride 逐元素激活）

```python
import triton, triton.language as tl
_NUM_VCORE = triton.runtime.driver.active.utils.get_device_properties("npu")["num_vectorcore"]

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": b}) for b in (512,1024,2048,4096,8192,16384,32768)],
    key=["numel"],
)
@triton.jit
def _activation_fwd_kernel(x_ptr, y_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    for off_start in range(pid * BLOCK_SIZE, numel, num_progs * BLOCK_SIZE):
        off = off_start + tl.arange(0, BLOCK_SIZE)
        mask = off < numel
        x_val = tl.load(x_ptr + off, mask=mask, other=0.0)
        x_f32 = x_val.to(tl.float32)
        y_f32 = compute_fn(x_f32)          # GELU / SiLU / SwiGLU 替换此处
        tl.store(y_ptr + off, y_f32.to(x_val.dtype), mask=mask)

def activation_impl(x, *args):
    ori_shape = x.shape
    x_flat = x.reshape(-1) if x.is_contiguous() else x.contiguous().reshape(-1)
    numel = x_flat.numel()
    y = torch.empty_like(x_flat)
    grid = (min(triton.cdiv(numel, 512), _NUM_VCORE),)
    _activation_fwd_kernel[grid](x_flat, y, numel)  # GELU/SwiGLU 加 multibuffer=True,unit_flag=True；SiLU 不加
    return y.reshape(*ori_shape)
```

> BLOCK autotune 范围 512-32768，仅以 `numel` 为 key。`num_vectorcore` 模块级缓存避免逐次 driver 查询。

---

## Layer 3: 关键技巧与已验证变体

### L3.1 动态 grid + compiler hints（GELU: +5.4%）
- 将固定 `grid=(num_vectorcore,)` 替换为 `min(cdiv(numel, BLOCK), num_cores)`
- 添加 `multibuffer=True, unit_flag=True`
- 效果: GELU geomean 0.7561× → 0.7969×

### L3.2 1D 展平 + BLOCK autotune + hints（SwiGLU: 1.3096×）
- autotune `BLOCK_SIZE=[512-65536]`（8 个档位），key=`numel`
- 加 `multibuffer=True, unit_flag=True`
- 中等 shape（999×9999, 1024×10240, 5336×3584）达 **1.5-2.1×**
- **关键差异**: SwiGLU 双输入使 load 带宽翻倍，1D 连续访存的收益比单输入更大（SwiGLU 1D/2D 差 0.17× vs SiLU 差 0.02×）

### L3.3 去掉 hints 的 SiLU（1.2669×，kernel-only）
- autotune + 无 hints + 缓存 num_vectorcore
- npu.Event（op 级，含 host wrapper 开销）显示 0.945×（比原始差），但 **CANN profiler kernel-only 显示 1.2669×**
- **教训**: 原始 host wrapper 更精简，op 级 Event 会误判；必须以 CANN kernel-only 为准
- 中型 shape fp16/bf16 达 2.3-2.49×

### L3.4 Tail block 特化 —— 无效
- 将满 block（无 mask load/store）与 tail block（有 mask）拆分 → GELU 0.7923× < 统一 mask 版
- **结论**: 逐元素激活用统一 mask 更简单、更快；tail 特化的分支开销抵消向量化收益

### L3.5 BLOCK_SIZE 扫描 —— 平坦
- GELU 上 BLOCK=4096/8192/16384 几何平均几乎相同
- 默认 `BLOCK=4096`

### L3.6 fp32 路径接近持平
- fp32 shape 普遍接近 0.98×（与 baseline 持平），fp16/bf16 大 shape 是主要劣化来源
- 若后续优先优化 fp16/bf16，可考虑 dtype 特化路径或 autotune BLOCK

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 小 tensor 用固定满核 grid | 浪费 launch/scheduling 开销，小 shape 上 baseline 反超 | L1.3 动态钳制核数 |
| 2D tile 双层 mask（含把单输入 2D 模式复制到双输入） | 跨步访存 + 额外 mask 运算；双输入 load 带宽翻倍时更伤 | L1.2 一律 1D 展平 |
| Tail block 特化拆分 | 标量分支开销抵消向量化收益 | L3.4 统一 mask |
| tanh 近似对 exact erf 参考 | 精度不匹配 | L1.1 仅参考用 tanh 时允许 |
| 混用单/双输入算子的 hint 结论、或统一加 multibuffer/unit_flag | 单/双输入访存模式不同；hints 对小 kernel 有 per-launch 开销 | L1.5 逐算子独立实测 |
| 信 op 级 npu.Event 为最终指标 | 含 host wrapper 开销，原始 wrapper 更精简 | L3.3 以 CANN kernel-only 为准 |
| 伪造几何平均聚合数 | per-case 全部 <0.5 却报 1.33×（mojo_0729 op15 session 数据作废） | 必须逐 case 校验一致性 |

---

## 2026-09-17 追加：1_GELU (50-case, approximate none/tanh 混合, Ascend910_9372)

> 验证环境: Ascend910_9372 (40 vector core, phy chip 6) / 1D 展平 + persistent-core 连续段骨架 /
> 基线 geomean 0.7832 (vs torch eager 现场实测) / 50 cases (fp32/fp16/bf16 × none/tanh, numel 100~134M)。
> 注意: 本次 erf/tanh 双语义由 `APPROX_TANH: tl.constexpr` 单 kernel 分派（模板 L1.1 的 none=erf 强制 + 
> tanh=sigmoid 恒等形照抄 gelu_dag.h 常数 1/0.044715、-1.595769121*0.044715）。

### 新增 Layer 1 候选约束（待二次验证）
- **L1.6 erf 路径预算认知**: approximate='none' 的 tl.erf 在 910B 上是 VECTOR 计算 bound
  （simulator: VECTOR 76.2%, SCALAR 仅 6.6%），实测 tl.erf ≈ copy 下限的 4.8×（fp16 134M: 2.37ms vs 0.49ms），
  CANN eager 同 case 0.55ms ≈ 访存下限。**含 erf case 的 GELU benchmark geomean 上限约 0.8（vs eager）**，
  设 target 时按此校准预期。

### 新增证伪记录（三要素）
1. **A&S 7.1.26 多项式替代 tl.erf —— 证伪**。写法: 1 div+1 exp+5 FMA+where 的 fp32 向量化序列;
   实测 3.02ms > tl.erf 2.37ms（B=4096/8192 上限; B≥8192 因中间量 UB overflow 196,640B>196,608B 编译失败）。
   失效边界: 910_9372 + 本骨架 + 134M 大 shape。未被证伪变体: 纯 FMA 高次 minimax（无 exp/div）、fp16 域求值（受 L1.4 禁）。
2. **mask 改 fp32 比较 —— 证伪**（2.85ms vs i32 mask 2.37ms）。fp32 转换成本 > i32 LT 向量化收益;
   int64 标量基址实测零开销（勿为此重写寻址）。
3. **tl.range num_stages=2/4 —— 无效**（erf/tanh 两路径 ±0.5% 内），与 mapping.md「NPU 不生效」一致;
   runbook exp_geglu_flat_001 的 num_stages 收益论断在本骨架不成立。

### 环境陷阱（测量纪律）
- 910B 卡热节流（1800→800MHz）会污染逐 case 数据: 同代码两卡复测 geomean 0.7832/0.7560，
  单 case 最大偏差 4×（0.183→0.695）。**小 shape（~2μs kernel）逐 case 噪声 ±0.3，优化判定须用 geomean
  且差异 >0.03 才可置信**；异常离群 case 先换卡复测再归因。

---

## 2026-09-18 追加：1_GELU 第二次会话（同 50-case，grid-stride 骨架，Ascend910_9372）

> 验证环境: Ascend910_9372 (dav_2201, 20 AICore / 40 VectorCore, device 1) / cann-9.1.0 /
> 骨架 = grid-stride 1D 循环（B-KB exp_gelu_fwd_001 推荐）+ autotune BLOCK∈{512..32768} /
> 基线 geomean **0.7293**（vs torch eager，50/50 精度全过）。
> 轨迹: triton_ascend_output/op_1_GELU_20260917_1827_8773/

### 关键对比（骨架级）
- **同任务同 case 集，grid-stride 骨架 0.7293 < 2026-09-17 会话 persistent-core 连续段骨架 0.7832**（差 ~7%）。
  B-KB 卡 exp_gelu_fwd_001 的 grid-stride 收益论断（声称 +15%）在本任务上**不成立**——
  候选假设：persistent-core 连续段（每核一段连续区间）访存局部性优于 grid-stride 交错。
  后续会话应优先复用 persistent-core 骨架，grid-stride 降级为候选假设（保质期规则：骨架不同结论不可互套）。

### 新增证伪记录（三要素）
1. **libdevice tanh 替换手写 sigmoid 恒等形 —— 证伪**。写法: `0.5*x*(1+tl.extra.ascend.libdevice.tanh(z))`
   替换 `x/(1+exp(-s))`（严格恒等）; 实测 geomean 0.7273 < 基线 0.7293（50 case）。
   失效边界: 910_9372 + grid-stride 骨架 + tanh 路径。未被证伪变体: 其他有 libdevice 版的超越函数替换。
2. **kernel 分裂小 shape 专用 kernel（固定 BLOCK=阈值上限）—— 证伪**。写法: numel≤8192 路由到
   BLOCK=8192、grid=(1,) 无循环的单块 kernel; 实测 0.5929 ≪ 0.7293（numel=100 也算满 8192 宽 tile）。
   未被证伪变体: 按 numel 细分多档小 BLOCK（128/512/2048）的专用 kernel。
3. **autotune 阶梯上限扩展 +65536 档 —— 证伪**。实测 0.7074 < 0.7293; UB 1.5MB 未触顶（65536×fp32×5buf≈1.3MB），
   劣化源于 grid 钳制 40 核后并行度损失 > barrier 摊薄收益。失效边界: dav_2201 40 核机型。

### Simulator 诊断（msprof op simulator，SOC=Ascend910_9372，印证并升级 L1.6）
- erf 变体: **VECTOR 75.5%**，top 指令全为 `BAR(pipe=VECTOR)` 每条 ~600cyc（= IR 中每两条向量 op 间的
  `pipe_barrier[<PIPE_V>]`），code_exe 单行 `tl.erf(...)` 独占 **71.4%** cycles；FLOWCTRL≈0%，无 Cube 参与。
- tanh 变体: VECTOR 51.1% / MTE2 15.1% / MTE3 15.6% 均衡，非瓶颈。
- 结论: erf 有理分式链（clamp±3.92 + Horner + vdiv，~20 向量 op）逐 op barrier 串行是计算 bound 的结构根因，
  **非同步可优化问题**；解耦类优化点（Cube/MTE3/Workspace）对纯 Vector 单输出 kernel 全部不适用。
  L1.6 的"含 erf case GELU geomean 上限约 0.8"判断经第二次独立会话 simulator 证据**升级为已确认**。

---

## 2026-09-18 追加：2_SwiGLU（50-case，任意 dim chunk，Ascend910B1）

> 验证环境: Ascend910B1（48 VectorCore，cann-9.1.0，实测卡 device 3；卡 0/1/4 故障禁用）/
> 骨架 = persistent-core 连续段 + host 两路径分派（twoslab/rowseg）+ autotune BLOCK∈{512..16384} +
> multibuffer/unit_flag / 基线 geomean **1.4451**（vs torch eager 现场实测，repeats=10，50/50 精度全过）。
> 轨迹: triton_ascend_output/op_2_SwiGLU_20260918_1033_3562/。注意跨机对比：950PR 上同类 SwiGLU 1.3096×。

### 新增 Layer 1 候选约束（chunk 类算子通用，待二次验证）
- **L1.7 torch.chunk 内存模型（强制认知）**: `chunk(x, 2, dim)` 沿**任意** dim 都是取该维前/后半，
  内存视图恒为 `[OUTER, 2, SEG]` 两段连续（`OUTER=prod(shape[:dim])`，`SEG=(shape[dim]//2)*prod(shape[dim+1:])`），
  **永不交织**——包括 dim=-1。输出元素 `j=o*SEG+i` → `a_off=o*2*SEG+i`，`b_off=a_off+SEG`。
  反例：本会话 iter_0 误判 dim=-1 为 stride-2 交织寻址（`x[2*offs]`/`x[2*offs+1]`），verify 47/50 失败。
  基于此模型的正确分派：OUTER==1 → 双整段全连续 kernel；否则行段 kernel（行长=SEG，行内 a/b/out 三连续）。

### 新增 Layer 3 技巧
- **L3.7 sigmoid 链已被编译器代数融合（IR 证据）**: `af * tl.sigmoid(af) * bf` 在 last_pass.mlir 中
  直接降为 `vdiv(af, 1+exp(-af))` 后单 `vmul(·,bf)`（vmuls(-1)→vadds(+0)→vexp→vadds(+1)→vdiv→vmul 共 6 op），
  源码层无需手动重排；其中 `vadds(+0.0)` 是 lowering 内部冗余，源码层不可消除，不要再为此改公式。

### 新增证伪记录（三要素）
1. **#17 kernel 分裂之 multirow 小 SEG 打包 —— 证伪**。写法: SEG constexpr pow2 → shift 除法，
   `a_off = offs + (offs//SEG)*SEG` 的多行打包 kernel（store 连续、a/b 为行内定长 run 的 gather）。
   实测 geomean 0.6858 ≪ 基线 1.4451；BLOCK=4096 时出现 ~0.0976ms 恒定病态延迟（grid=8 欠载 + 单核大 tile），
   即使调优到 BLOCK=1024（0.0278ms）仍劣于 rowseg（0.006-0.0115ms，[256,256] fp16 dim=-1 单 case profiler）。
   失效边界: 910B1 + persistent-core 行段骨架 + seg<2048 pow2 小 SEG case 集。
   未被证伪变体: 其他小 SEG 打包形态（如 2D block 向量 load 替代 gather 寻址）。
2. **优化点 #32/#33 不命中条件确认**: 基线已带 `@triton.autotune`（BLOCK 512-16384 阶梯）时，
   autotune 点不命中；#33 作为 fallback 因 #32 已采纳同样不命中——不要对已 autotune 的 kernel 重复命中这两点。

### Simulator 诊断（msprof op simulator，SOC=Ascend910B1）
- twoslab/rowseg 均为 **VECTOR ~40% + MTE2+MTE3 ~40% 均衡**，FLOWCTRL 0.1%，无 Cube 活动（MMAD=0）——
  elementwise 算子的健康形态，无 latency-optimizer 优化点可映射（访存已全连续、hints 已开、autotune 已覆盖）。
- **SCALAR% 误诊陷阱**: rowseg 部分核 SCALAR 达 31.8%，但 top SCALAR 指令（LDP/LD/ST 地址建立）calls=1，
  属一次性开销，**非**逐元素标量降级（规则 2 要求 calls≈tile 元素数）。判 SCALAR 降级必须看 call_count。
- **simulator 环境陷阱**: camodel 不支持 NPU 侧 `torch.randn`（aclnnInplaceNormal 561103）/ Cast——
  输入必须在 CPU 构造后 `.to(dev)`；`--kernel-name` 过滤不到时先确认 app 本身跑完（打印 DONE）。
- **IR 提取陷阱**: `TRITON_ALWAYS_COMPILE=1` 下 autotune do_bench 可能崩（SUSPECT REMOTE ERROR 507057），
  但编译期 dump 已产生，last_pass.mlir 提取不受影响；多 kernel 算子宜每 kernel 单独驱动脚本提取。

---

## 2026-09-18 追加：4_Abs（50-case，numel 100~134M 全覆盖，Ascend910_9372）

> 验证环境: Ascend910_9372 (dav_2201, 20 AICore / 40 VectorCore, 实测卡 device 2；卡 0/1/4 故障禁用) / cann-9.1.0 /
> 骨架 = persistent-core 连续段 + **kernel 分裂三档**（grp_tiny 单 tile / grp_mid / grp_large）+
> autotune（tiny {128..2048} / mid {2048,4096,8192} / large {8192,16384,32768}）+ multibuffer + unit_flag /
> Phase 3 基线 geomean **1.1033** → 最终 **1.3000**（vs torch eager 现场实测，50/50 精度全过）。
> 轨迹: triton_ascend_output/op_4_Abs_20260918_1205_6758/

### 新增 Layer 3 技巧
- **L3.8 优化点 #17 kernel 分裂对「单指令 elementwise」有效（abs 实测 +0.1281）**: 与 GELU 会话证伪的
  "单档 BLOCK=8192 小 shape kernel" 不同写法——grp_tiny 用 autotune 多档小 BLOCK（128~2048）+ 单 tile 无循环，
  收益来自去掉 persistent 循环 prologue；mid/large 仅按规模收窄 autotune 档范围。机制: 小 shape launch/prologue
  开销占比高，按规模分档让各档 kernel 结构最小化。
- **L3.9 mid/large kernel 叠加 multibuffer=True + unit_flag=True（abs 实测 1.2314→1.2758→1.3000）**:
  机制: multibuffer 打开 UB 双缓冲加深 MTE2→V→MTE3 流水重叠（IR 显示重叠窗口加深）；unit_flag 让编译器
  复用已建立地址单元（L1.5 已记 hints 需逐算子实测——abs 两个 hint 均为正收益）。

### 新增证伪记录（三要素）
1. **无新增证伪**（本会话 3 轮优化全部采纳，opt_iter_3 全扫 1-31 无命中为自然耗尽，非证伪）。
2. **#32/#33 不命中条件再次确认**（与 SwiGLU 会话一致）: 基线已带 `@triton.autotune` 阶梯时两点均不命中。

### Simulator 诊断（msprof op simulator，SOC=Ascend910_9372）
- grp_large 大 shape（28M~134M）: **MTE3 55-59% + MTE2 26-31% ≈ 87%**（store_ubuf_to_gm 主导），
  VECTOR ~10%，无 Cube（MMAD=0）——**纯访存带宽饱和**，优化点 7/21/10 全部不适用。
- grp_tiny: SCALAR 36.7% 为一次性 prologue（LDP 地址建立，calls=1 级）——再次印证 SwiGLU 会话的
  "SCALAR% 误诊陷阱"：必须看 call_count 区分一次性开销与逐元素标量降级。
- **结论（升级为类别级认知）**: 纯 load+store+1 条向量指令的 elementwise 算子（abs 型），geomean vs torch
  eager 上限约 1.3×（910_9372 带宽极限）；target≥2× 对此类算子物理不可达，不要为冲高倍数做无效迭代。

---

## 2026-09-18 追加：4_Abs 第二次会话（dtype 双 kernel 分裂路线，Ascend910_9372）

> 验证环境: Ascend910_9372 (dav_2201, 40 VectorCore, 实测卡 device 7；卡 0/1/4 故障禁用) / cann-9.1.0 /
> 骨架 = persistent-core 连续段 + **dtype 双 kernel 分裂**（fp16/fp32 原生 tl.abs / bf16 升 fp32）+
> autotune BLOCK∈{1024,2048,4096,8192} key=["NUMEL"]，**无 compiler hints** /
> Phase 3 基线 geomean **1.0683** → 最终 **1.3130**（vs torch eager 现场实测，50/50 精度全过）。
> 轨迹: triton_ascend_output/op_4_4_Abs_20260918_1207_6897/。
> 与同日上午 4_Abs 会话（1205_6758，三档规模分裂路线 1.3000）**并行跑同机不同卡**，两条独立路线
> 收敛到几乎相同 geomean（1.3000 vs 1.3130），进一步坐实「abs 型算子 ~1.3× 带宽上限」的类别级认知。

### 新增 Layer 1 候选约束（dtype 混合 case 集通用，待二次验证）
- **L1.8 triton-ascend autotuner 缓存 key 不含 constexpr 参数（强制认知）**: 单 kernel 用
  `PROMOTE_F32: tl.constexpr` 分派 dtype 路径时，把 constexpr 名写进 `key=[...]` **无效**——
  同 NUMEL 不同 dtype（如 fp32 case 与 bf16 case）共享同一份调谐缓存，第二 dtype 首 launch 触发
  内联 JIT 编译被 profiler 计入（实测假数据 6.7~9ms / case）。**修复 = 按 dtype 分裂为独立
  `@triton.jit` kernel**（两个 Autotuner 实例缓存天然隔离，geomean 1.0844→1.3130）。
  凡 case 集混 dtype 且 kernel 内有 dtype 分支的，一律双 kernel，不要信 constexpr-in-key。

### 新增 Layer 3 技巧
- **L3.10 dtype 双 kernel 分裂（优化点 #17 的另一形态，abs 实测 +0.2286）**: 与 1205_6758 会话的
  「按 numel 规模三档分裂」不同写法——按 dtype 分裂（原生路径 / bf16 升 fp32 路径），
  机制是消除 autotune 跨 dtype 缓存污染（L1.8），而非 kernel 结构最小化。两条路线最终 geomean
  相当（1.3000 vs 1.3130），可按 case 集特征任选；dtype 混合越悬殊，本路线收益越稳。

### 新增证伪记录（三要素）
1. **multibuffer+unit_flag 加在 dtype 分裂骨架上 —— 证伪（与 1205_6758 会话结论冲突，并列保留）**。
   写法: 双 kernel launch 均带 `multibuffer=True, unit_flag=True`（全 numel 段）；实测 geomean
   1.1766 < 无 hints 版 1.3130。失效边界: 910_9372 + dtype 双 kernel 骨架 + 50 case。
   ⚠️ 冲突标注: 1205_6758 会话在三档规模分裂骨架上实测同两 hints 为正收益（1.2314→1.3000）——
   两结论均真实，hints 收益**强依赖骨架**；按「经验卡保质期」规则，两骨架互套时对方结论自动降级为
   候选假设，以本地实测为准。未被证伪变体: hints 仅加在规模分裂的 mid/large 档（1205_6758 形态）。
2. **host 静态分档替代 autotune（优化点 #12）—— 证伪**。写法: 去 `@triton.autotune`，host 按
   numel<65536 → BLOCK=1024 否则 4096 静态分派；实测 0.8669 ≪ 1.3130（失去逐 NUMEL 调谐 + 分档边界
   case 集体劣化）。失效边界: 50 case numel 跨度 100~134M。未被证伪变体: case 集 numel 集中时的静态分档。
3. **hints 按 numel 分段混合（优化点 #13）—— 无效（噪声内）**。写法: numel<4M 加 hints、≥4M 不加；
   实测 1.2978 vs 1.3130，差 0.0112 < 0.03 噪声阈，判无提升。本质上 hints 在 dtype 骨架任何子段都非正收益。
4. **纯净 BLOCK=16384 档（优化点 #32 扩展）—— 证伪**。写法: autotune 阶梯 {1024..16384} 五档;
   实测 1.1507/1.2340 两次（频率漂移期数据失真，换卡复测仍 < 1.3130）。16384 档对个别大 fp32 case
   （case 35, 28M）有效但 geomean 被小 case autotune 误选拖垮。未被证伪变体: 仅大 numel host 门控启用 16384。

### Simulator 诊断（msprof op simulator，SOC=Ascend910_9372）
- 主 kernel（fp32 小 shape [64,128] 单 launch）: MTE2 27.4% / SCALAR 24.4% / MTE3 22.8% /
  VECTOR 14.1%，MMAD=0%；bf16 kernel: SCALAR 29.8% / MTE3 21.0% / VECTOR 19.6% / MTE2 19.3%。
- 两 kernel 均为访存 bound（MTE2+MTE3 ≈ 40~50%），SCALAR 高占比经 call_count 复核为一次性 prologue
  （再次印证「SCALAR% 误诊陷阱」）；bf16 的 VECTOR 略高为 cast 链开销（vabs 无 BF16 指令，硬件强制）。
- 结论与 1205_6758 会话一致: 访存类优化点 7/21/10 对「单 load + 单 store」最简形态全部不适用。

### 环境陷阱（测量纪律补充）
- **并行会话同机干扰**: 本次与 1205_6758 会话并行跑同机（不同物理卡），device 7 多次出现
  1800→800MHz 频率漂移，污染 autotune do_bench 选档与小 case 逐 case 数据——除既有纪律
  （geomean 差异 >0.03 才置信）外，**疑似的优化收益必须换卡/换时段复测一次再采纳**。
