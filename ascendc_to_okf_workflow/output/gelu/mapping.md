# P2 语义转化 — gelu（Ascend C 事实 → Triton 决策）

> 生成时间：2026-09-17｜依据：`extraction.md`（锚点复用不重复标注）｜**P2.5 skipped: low-value**（manifest 判定），后端论断经 backend_facts 检索取证，证据等级 [SRC]/[DOC]，不升 [EXP]。

## 0. 检索记录（backend_facts → zh → IR）

| 决策点 | 命中 | 出处 |
|---|---|---|
| `tl.exp` 可用性与 dtype | VExpOp vexp 仅 **F16/F32**（无 BF16）→ **exp 必须在 fp32 域**（与源码 DAG 天然一致） | backend_facts/dtype_support.md:157 [SRC] |
| `x / y` 除法 | vdiv 支持 f16/f32（bf16 见 §3 注不支持）→ fp32 域安全 | dtype_support.md:52 [SRC] |
| mask 写法 | where 内比较转 fp32；**load/store 的 mask 保持整型比较**（i32 `<` 标量化属"正确性优先"已知例外） | dtype_support.md §3 [DOC+SRC]、backend_contracts C-002 |
| 尾轴对齐 | VV 类 32B 整除 → BLOCK 取 32B/dtype 倍数（候选 1024~8192 全满足） | backend_contracts C-010 [DOC] |
| 大 numel 寻址 | numel ≥ 2³¹ 时 int32 偏移静默溢出 → **int64 仅限标量块基址**，块内偏移保 int32 | dtype_support.md 4e [SRC+EXP 先例] |
| 禁 `%` | 本算子一维线性寻址，天然无 `%`/`//` | backend_contracts C-001；tl_op_lowering.md:79-81 |
| multibuffer | bishengir EnableAutoMultiBuffer 默认 true → 不显式编码 | compiler_options.yaml:21-25 [SRC] |
| num_stages/num_warps | NPU 运行时不生效 → 不写 | GDN 骨架 C-006 既有结论 |
| persistent-core | `grid = min(cdiv(numel,BLOCK), vector_core_num)` + 核内**连续段**循环（交织 grid-stride 次优，禁） | ascendc_triton_semantic_mapping.md §1 行 2 [EXP 2026-09-07] |
| vector 核数查询 | `get_device_properties(...)["num_vectorcore"]`，禁写死 | output/add/triton_skeleton.py:31 先例 |

## 1. 分核策略 → grid

Ascend C 事实：展平一维、核数 = min(按 MIN_TILING_BITS 折算, AIV 核数)，每核 cache-line 对齐**连续段** blockFormer + 尾核 blockTail，核内 ubFormer 循环（elewise_tiling.cpp，extraction §4.1）。
Triton 决策：**persistent-core 忠实映射**——`grid = min(cdiv(NUMEL, BLOCK), NUM_VEC_CORE)`；每 program 连续段 `per_prog = cdiv(NUM_BLOCKS, num_progs)`，`for blk in range(pid*per_prog, min((pid+1)*per_prog, NUM_BLOCKS))`。尾块由 mask 统一处理（mapping §1 normal/tail 行：优先 mask 统一写法）。
等价论证：官方两级切分（核间连续段 + 核内 UB 块）= Triton 连续段循环内每 iteration 一个 BLOCK，负载等价；对齐约束由 BLOCK 候选保证（§4）。

## 2. dtype 分派 → 单 kernel

Ascend C 事实：3 路 constexpr 壳（TPL_FP16/BF16/FP32 → GeluDAG<U, float>，计算域恒 fp32）。
Triton 决策：**单 kernel 无 dtype 分支**——dtype 由指针类型推导（mapping §1 dtype 壳行 ✅）：`load → .to(tl.float32) → 计算 → .to(out_dtype) → store`。fp32 输入时两次 cast 恒等。13→1 实例简化同 add 先例。
tilingkey 两维（固定 1 × dtype）无需映射——dtype 差异已被指针推导吸收；schMode∈{0,1} 是 base 库调度内部键，Triton 无对应（编译器自行调度）。

## 3. Compute → Triton 计算骨架

Reg 链逐条映射（gelu_dag.h:55-65 → 骨架 kernel 体，全部 mapping §3 ✅ 行）：

| Reg 原语 | Triton | 可达性 |
|---|---|---|
| LoadAlign / StoreAlign | `tl.load / tl.store`（mask 尾块） | ✅ |
| `Mul` x²、x³ | `x*x`、`x2*x` | ✅ vmul f32 |
| `Axpy(cub, x, 1/0.044715)` | `x3 + x * TANH_APPROX_FACTOR`（编译期常数；mul+add 可能被 TRITON_DEFAULT_FP_FUSION 折成 fma，与 Axpy 同语义） | ✅ |
| `Muls(·, -1.595769121*0.044715)` | `* NEG_SQRT_EIGHT_OVER_PI`（常数表达式照抄，trace 期折叠） | ✅ |
| `Exp` | `tl.exp`（fp32 域，dtype_support:157 安全） | ✅ |
| `Adds(·, 1.0)` + `Div(x, ·)` | `x / (1.0 + t)` | ✅ vdiv f32 |
| Reg VL 循环 + UpdateMask | 无对应——BLOCK 内由编译器细分 VL（gelu_dag.h:37-39 表明硬件本就 VL 粒度流水，BLOCK 只是 UB 块） | ⚠️ 自动 |

**常数保真决策**：照抄源码两常数表达式 `1/0.044715`、`-1.595769121*0.044715`（而非化简的 1.5957691 直写）——与官方舍入路径逐位同构，消灭与 golden 比对时的常数来源差异。

## 4. 片上内存 → 预算校核

BLOCK=8192 上限情形：fp16 输入块 16KB + fp32 中间（编译器复用，峰值按 2 块计）64KB + 输出 16KB ≈ 96KB < A2 192KB / 950 248KB（hardware_platforms [SRC]）✅。BLOCK=8192 即安全上限，autotune 候选 {1024, 2048, 4096, 8192} 全部可行，无需分平台降档。
双缓冲：不写显式 pingpong（mapping §2 double buffer 行 ⚠️），由 multibuffer 默认 true 承接（§0）。

## 5. 大 numel 寻址（4e 防御）

`base = blk.to(tl.int64) * BLOCK`（int64 标量基址），`cols = tl.arange(0, BLOCK)`（int32 块内偏移），指针算术 `x_ptr + base + cols`。mask：`rem = tl.minimum(NUMEL - base, BLOCK).to(tl.int32)` 后 `cols < rem`——保持整型比较（dtype_support §3 规则 2，store/load mask 正确性优先）。
0~8 维全动态 shape（def.cpp:37-38）→ numel 可超 2³¹（fp32 8GB 输入在 950 上合法），此防御不可省。

## 6. 显式丢弃 / 不涉及清单

- ❌ 项 = 0（复核）：无寄存器直写 E2B/reshape、无 gather/scatter、无 pingpong、无 NZ——低价值判定自洽。
- aclnn 层 Contiguous/ViewCopy（aclnn_gelu.cpp:112,120）→ 丢弃，wrapper 约定 `.contiguous()`（mapping §1 多 kernel 分解行：结构性更优）。
- workspace=0 → Triton 天然满足。
- 无 approximate 属性、无 runtime 标量属性 → 无 constexpr 两态问题（与 fused_bias_leaky_relu 不同，本算子无任何属性）。

## 7. 精度策略 → 验收

- 中间域恒 fp32 + RTNE 回写（Cast RINT = Triton cast 默认舍入，mapping §3 Cast 行 ✅）→ 骨架 `.to(tl.float32)` / `.to(out_dtype)` 必须保留，写入卡片验收标准。
- golden = torch.nn.GELU(approximate="tanh") fp32 域计算（golden.py:36-44）——数学恒等式 0.5x(1+tanh(z)) = x/(1+e^{−2z}) 保证两形等价；fp32 域内差异 < 1 ulp 级。
- inf 支路数值一致性已论证（extraction §7）：大负 x → e^t→+inf → y→−0，与 tanh 形极限一致；ST 含 fp16 inf case 作背书。
- 精度异常排查旋钮：`TRITON_DEFAULT_FP_FUSION=0`（dtype_support §4 [DOC]）。

## 8. P2 质量门自查

- [x] 每个 tl op 有 lowering 依据（§0 检索记录，[SRC]/[DOC] 分级）
- [x] ❌=0、⚠️=2（VL 循环自动细分、multibuffer 承接）均有出处
- [x] UB 预算字节数校核（§4）
- [x] 无违禁写法：无 `%`、无 int64 向量 lane、mask 整型比较、常数非烘焙运行时标量（编译期字面量）
- [x] **P2.5 skipped: low-value**——confidence 保持 inferred，不产 p25_report.md
