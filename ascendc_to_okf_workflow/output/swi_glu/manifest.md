# P0 材料清单 — swi_glu

> 生成时间：2026-09-17｜来源目录：`src/ops-nn/activation/swi_glu/`
> **kernel 类型判定：AIV-only elementwise（split-view 二目）**——非 DAG 算子，**手写 AscendC kernel**（TPipe/TQue/DataCopy 全套），3 个实现类：`SwigluVector`（fp32 直通）、`SwigluVectorBF16`（fp16/bf16 升 fp32 计算）、`SwigluSingle`（单 GM 输入 + 父子类继承，负责 GM 寻址）。双 kernel 文件：`swi_glu.cpp`（910b/910_93，bf16 分支 `#if __CCE_AICORE__==220` 条件编译）与 `swi_glu_apt.cpp`（950 regbase，bf16 无守卫；def.cpp:55 `opFile.value=swi_glu_apt`）。
>
> **🔻 低价值预判：低价值——P2.5 skipped: low-value；P4 降单组冒烟。**
> - ❌=0：DataCopyPad→mask ✅；strided 2D DataCopy→stride 折算 ✅；TQue 双缓冲→multibuffer ⚠️；2D base-tile 优选/DataCopy 硬件上限（blockCount≤4095 等）= host 侧工程，Triton 由 autotune+编译器承接；无 gather/规约/广播。
> - 公式 `silu(A)·B`（sigmoid 形，fp32 域）在基座先验内；无 benchmark 语义超集（ST csv 为 fuzz 配置）。
> - **但有两个语义陷阱使本卡有真实价值**（非「增益≈0」白卡）：① `beta` 成员初值 +1.0 是诱饵，子类 InitGmBuffer 里 `this->beta = -1.0f`（sigmoid 正号靠它）；② fp16 **也升 fp32**（与 abs 的 fp16 原生相反）——照抄 abs 卡习惯会错。
> - 联动结论：①「P2.5 skipped: low-value」；② confidence=inferred；③ 证据上限 [DOC]/[SRC]；④ 不产 p25_report.md；P4 降单组冒烟。

## 材料归集表

| 类别 | 路径 | 存在 | 关键符号 / 行号锚点 | 对应详设章节 |
|---|---|---|---|---|
| 原型定义 | `op_host/swi_glu_def.cpp` | ✅ | x/y {F16,F32,BF16}（:23-30）；**Attr dim OPTIONAL 默认 −1**（:32）；平台分组：910b/910_93 全 dtype（:33-34）、310p/kirinx90/kirin9030 无 bf16（:36-49）、**950 regbase cfg + `opFile=swi_glu_apt`**（:51-55）；DynamicRank/Shape | §2、§3 |
| GEIR 原型 | `op_graph/swi_glu_proto.h` | ✅ | 登记用 | §3 |
| shape 推导 | `op_host/swi_glu_infershape.cpp` | ✅ | 输出 = 输入在 dim 轴减半 | §2.1 |
| tiling host 侧 | `op_host/swi_glu_tiling.cpp` + `.h` | ✅ | 常数：L2 cache line 512B、UB 32B 对齐、DEFAULT_BUFFER_NUM=2、**DataCopy 硬件上限 blockCount≤4095 / blockLen≤65535×32B / stride≤65535**（:11-21）、WORKSPACE=16MB（:31）、TILING_KEY_EMPTY=100（:32）；**SetTotalShape：shapeBefore=∏dims<splitDim、shapeAfter=∏dims>splitDim，负 dim 归一**（:172-190）；GluSingleTilingOptParam 2D base-tile 优选（baseRowLen×baseColLen、alignPackLen 512B/32B）（:56-64） | §4 |
| kernel 入口（910b） | `op_kernel/swi_glu.cpp` | ✅ | tilingkey = dtype 码 {0=fp32,1=fp16,27=bf16}；isDoubleBuffer→bufferNum 2/1；**bf16 分支 `#if __CCE_AICORE__==220` 守卫** | §4.2、§6 |
| kernel 入口（950 apt） | `op_kernel/swi_glu_apt.cpp` | ✅ | 同上但 bf16 无守卫；`TILING_KEY_IS(100)` 早退 | §4.2 |
| **计算实现（核心语义）** | `op_kernel/arch35/swi_glu_impl.hpp`（fp32）+ `swi_glu_bf16.hpp`（fp16/bf16→fp32）+ `swi_glu_single.hpp`（寻址） | ✅ | 计算链见下「语义速览」；**`InitGmBuffer: this->beta = -1.0f`**（single:45）；is32BAligned 双路径 DataCopy/DataCopyPad（single:52-119） | §1、§6、§7 |
| 文档 | `README.md`；`docs/aclnnSwiGlu.md` | ✅ | out=Swish(A)·B，A/B = x 按 dim 对半切；dim∈[-x.dim(), x.dim()-1] | §1.1 |
| golden | `tests/assets/golden.py` | ✅ | `torch.chunk(x,2,dim)` → `F.silu(a)*b`；**fp16 升 fp32 计算**（kernel golden）；aclnn golden 同 | §测试设计 |
| ST 用例 | `tests/st/arch35/ttk_kernel_swi_glu_st.csv`（79 case fuzz） | ✅ | **⚠️ 全 79 行 output_shapes == input_shapes，与 golden 语义（dim 轴减半）矛盾**——疑 fuzz 配置错误，以 golden 为准；dim 取值覆盖正负/中间轴/尾轴 | §测试设计 |
| UT | `tests/ut/op_kernel/test_swi_glu.cpp` + data gen | ✅ | kernel 级 UT 含 tiling gen | §测试设计 |

## 计算语义速览（P1 详抽对象）

**sigmoid 形 SiLU 门控**：`y = a·sigmoid(a)·b = a/(1+e^{−a})·b`，A/B = x 沿 dim 对半切。

两条计算链（数学等价、舍入路径不同）：
- fp32（impl.hpp:59-83）：`Muls(c,a,β=−1) → Exp → Adds(1) → Div(c, temp1.0, c) [=sigmoid(a)] → Mul(c,c,a) → Mul(c,c,b)`，全程 outType=float。
- fp16/bf16（bf16.hpp:62-88）：`Cast(a,NONE→fp32) → Muls(β=−1) → Exp → Adds(1) → Div(c, a, c) [=a·sigmoid(a) 直接] → Cast(b,NONE→fp32) → Mul(c,c,b) → Cast(out,RINT)`。**fp16 同样升 fp32**（`SwigluVectorBF16<half,float,half>`，入口 key=1）。

寻址（SwigluSingle）：单 GM 输入；A 块/B 块两次 DataCopy（splitVecGmOffset1/2），输出 numel = 输入 numel/2；32B 对齐与否双路径（DataCopy vs DataCopyPad+DataCopyPadExtParams）。

## 缺省项与注意点（P1 必须处理）

1. **beta 符号陷阱**：impl/bf16.hpp:29-30 成员初值 `beta = 1.0`，实际符号由 `SwigluSingle::InitGmBuffer` 覆写为 **−1.0f**（single.hpp:45，且留了 GM 读 beta 的注释掉代码 `/* * (((__gm__ float*)beta_gm)[0]) */`——历史上曾设计为 runtime 属性）。只读 impl 会得出 sigmoid(−a) 的错误结论。
2. **fp16 升 fp32**：与 abs（fp16 原生）相反、与 gelu 一致——卡片必须明示，防跨卡习惯错抄。
3. **ST csv 形状矛盾**（79/79 行 out==in shape）：以 golden（chunk+silu·mul）为语义权威，登记偏差。
4. **TILING_KEY_EMPTY=100 早退**（apt）——空/退化输入保护，P1 确认触发条件。
5. **DataCopy 硬件上限与 2D base-tile 优选**是 host 侧工程（AscendC API 约束），Triton 无需复刻；但 512B/32B 对齐常数与 C-010 互证，可引用。
6. **dim 任意轴**：SetTotalShape 折叠为 (before, dim, after) 三段乘积——Triton 侧对应 3D view 地址折算，dim=−1 时 after=1 直通（零整数除法）。
7. workspace 16MB 为系统预留（WORKSPACE_BUFFER），非算子数据面。

## P0 质量门结论

材料齐全（def/infershape/tiling/kernel×2 入口/impl×3/golden/ST/UT）→ **通过，进入 P1**。低价值判定成立（两语义陷阱提升卡片价值密度，但不改变 ❌=0 的实测豁免结论）：**P2.5 skipped: low-value**，P4 降单组冒烟，confidence=inferred。
