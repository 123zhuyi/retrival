---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: SwiGlu（SiLU 门控线性单元，sigmoid 形实现）Ascend C → Triton 语义转化卡
description: split-view 二目 elementwise 激活 y=SiLU(A)·B，A/B=x 沿 dim 对半切。AIV-only 手写 kernel（非 DAG 家族），950 走 swi_glu_apt（regbase），910b 走 swi_glu.cpp（bf16 受 __CCE_AICORE__==220 守卫）。卡片价值集中在两个语义陷阱：① beta 成员初值 +1.0 是诱饵，符号由子类 InitGmBuffer 覆写为 −1.0（只读 impl 层会得出 a·sigmoid(−a) 的错误语义）；② fp16 也升 fp32 计算（SwigluVectorBF16<half,float,half>，与 abs 卡 fp16 原生相反）。寻址设计：SetTotalShape 折叠 (before,2H,after)，Triton 侧代数化简为一次整数除法 decode（a_off=offs+i·HA），无 %、j/k 解码消除。2D base-tile 优选/DataCopy 硬件上限属 host 工程不复刻。P0 判低价值（❌=0），P2.5 skipped: low-value，P4 降单组冒烟。
tags: [swiglu, glu, activation, elementwise, silu, sigmoid_form, beta_sign_trap, fp16_promote, dim_fold, aiv_only, ascend950, ascend910b, ascendc_kernelwiki]
confidence: inferred
status: draft
resource: src/ops-nn/activation/swi_glu/（CANN ops-nn 本地快照，910b/910_93/310p/kirin/950）
created_at: "2026-09-17T00:00:00Z"
updated_at: "2026-09-17T00:00:00Z"
---

sources:
  - url: src/ops-nn/activation/swi_glu/
    role: primary

# SwiGlu（y = SiLU(A)·B）

Swish 门控线性单元：A/B = x 沿 dim（默认 −1）对半切，`out = SiLU(A)·B`（README；docs/aclnnSwiGlu.md）。**AIV-only 手写 kernel**（TPipe/TQue/DataCopy 全套，非 DAG 家族）。双入口：910b/910_93 → `swi_glu.cpp`（bf16 分支 `#if __CCE_AICORE__==220` 守卫）；**950 → `swi_glu_apt.cpp`**（regbase，bf16 无守卫，`opFile.value=swi_glu_apt`，def.cpp:55）。golden = `torch.chunk(x,2,dim)` → `F.silu(a)*b`（fp16 升 fp32，golden.py）。

## 1. 功能与公式（详设 §1）

- 语义：`y = a·sigmoid(a)·b = a/(1+e^{−a})·b`（sigmoid 形 SiLU）。
- **⚠️ 陷阱 1（beta 符号）**：计算类成员初值 `beta = 1.0`（impl.hpp:29 / bf16.hpp:30）是诱饵——`SwigluSingle::InitGmBuffer` 覆写 `this->beta = -1.0f`（single.hpp:45）。只读 impl 层会得出 `a/(1+e^{a}) = a·sigmoid(−a)` 的**错误语义**。同行注释掉的 `* beta_gm[0]` 表明 beta 曾设计为 GM runtime 属性（现硬编码 −1.0，勿复活）。
- 输出 numel = 输入 numel/2；splitDim 轴长度须为偶数。

## 2. 原型与约束（详设 §2）

### 输入输出（GEIR，swi_glu_def.cpp:23-32）
- x/y：{F16, F32, BF16}，ND，同型；**Attr dim OPTIONAL 默认 −1**，取值 [-rank, rank-1]。
- 平台分组：910b/910_93 全 dtype；**310p/kirinx90/kirin9030 无 bf16**；950 regbase。
- infershape：输出 = 输入在 splitDim 轴减半。

### aclnn 层
两段式直通；无其他属性。

## 3. 规格约束（详设 §3）

| 项 | 值 |
|---|---|
| dtype | fp16/fp32/bf16（310p/kirin 无 bf16） |
| shape | 动态 rank；splitDim 轴偶数 |
| dim | [-rank, rank-1]，默认 −1 |
| 平台 | 910b/910_93/310p/kirin9030/kirinx90/950 |
| ST 值域 | 多 (-1,1)，含 (-1000,-10) 大负支路 |
| ST 容差 | fp32 1e-4、fp16/bf16 1e-3 |

## 4. 分核策略（详设 §4 → Triton grid）

Ascend C 事实：host 侧 2D base-tile 优选（baseRowLen×baseColLen，UB 内枚举最优，tile 均分到核；tiling.cpp:56-64）；DataCopy 硬件上限 blockCount≤4095/blockLen≤65535×32B/stride≤65535（:16-18）。
Triton 映射：**输出展平 1D + persistent-core 连续段**（gelu/abs 家族范式）——`grid = min(cdiv(NUMEL_OUT, BLOCK), num_vectorcore)`，NUMEL_OUT = numel_x/2。2D tile 优选是 DataCopy 粒度工程，**不复刻**（编译器 MTE + autotune 承接）；32B/512B 对齐常数与 C-010 互证。

## 5. BLOCK / 地址折算（详设 §4.3/§6，本算子核心设计点）

### dim 折叠 → 一次除法 decode
wrapper 折三段 `before/H/after`（对齐 SetTotalShape，tiling.cpp:172-190）；kernel 内：
```python
HA = H * after                       # 每个 before 段的输出元素数
i  = offs // HA                      # ⚠️ 2026-09-18 实测修正：i32 向量除法在热路径
                                     # 是性能灾难（回放实测 44x：13.4ms→0.3ms）——
                                     # 「vdiv-i32 指令存在」≠「便宜」；Phase 4 已改用
                                     # M/S 双路径结构消除 decode 除法。新卡勿再推荐
a_off = offs + i * HA                # 恒等式：(i·2H+j)·after+k = offs + i·HA —— j/k 解码被代数消除
b_off = a_off + HA                   # B = A 沿 splitDim 后半
```
无 `%`（C-001）；`//` 不进 mask 表达式（tl_op_lowering rewrite 约束）。
### BLOCK
候选 {1024, 2048, 4096}，key=["NUMEL_OUT","HA"]；UB 最坏 ~104KB（含 multibuffer 余量）< 192/248KB [SRC]。

## 6. 片上内存预算（详设 §5）

3 TQue（inA/inB/outC，bufferNum=2）+ tmp：fp32 档 1 个 float tmp（预填 1.0）；低精度档 2 个 fp32 tmp（a/b 复用，bf16.hpp:35-37）。Triton 侧编译器自管，§5 已校核。

## 7. 流水编排与 Triton 可达性（详设 §6）

| Ascend C | Triton | 可达性 |
|---|---|---|
| A/B 两次 strided DataCopy | 两次 tl.load（a_off/b_off） | ✅ stride 折算 |
| DataCopyPad（非 32B 对齐路径） | mask + other=0.0 | ✅ |
| Muls/Exp/Adds/Div/Mul 链（fp32 域） | `a / (1.0 + tl.exp(-a)) * b` | ✅ vexp/vdiv f32 [SRC] |
| Cast NONE 进 / RINT 出 | `.to(tl.float32)` / `.to(out_dtype)` | ✅ 必须保留（**fp16 也升**） |
| TQue 双缓冲（bufferNum=2） | multibuffer 默认 true [SRC] | ⚠️ 不显式编码 |
| PipeBarrier<PIPE_V> 逐条同步 | 无 | ⚠️ 编译器自动 |
| 2D base-tile 优选 / DataCopy 上限 | autotune / 编译器 MTE | ⚠️ 不复刻（结构性无关） |
| TILING_KEY_EMPTY=100 早退 | wrapper numel==0 早退 | ✅ 等价 |

## 8. Triton 骨架（生成用伪码）

完整骨架 `output/swi_glu/triton_skeleton.py`（preflight 0 命中）。核心：
```python
a = tl.load(x_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
b = tl.load(x_ptr + b_off, mask=mask, other=0.0).to(tl.float32)
y = a / (1.0 + tl.exp(-a)) * b        # β=−1：e^{−a}；结合序取 bf16 链
tl.store(y_ptr + base + cols, y.to(y_ptr.dtype.element_ty), mask=mask)
```

## 9. 精度标杆与验收（详设 §7/§8）

- golden：`torch.chunk(x,2,dim)` → `F.silu(a)*b`，fp16/bf16 升 fp32 域。
- **必须保留**：① β=−1（e 的指数为 −a）；② 恒 fp32 中间域（含 fp16）+ RTNE 回写；③ dim 负值归一 + splitDim 偶数约束。
- 大负 a 支路：e^{−a}→+inf → a/inf→−0 ✓（ST (-1000,-10) 值域背书）。
- ST csv 79 行 output==input shape 与语义矛盾（疑 fuzz 配置错误）——**验收以 golden 为准**。
- 已知限制：numel_x ≥ 2³¹ 超 int32 安全域（wrapper 断言拦截；超限输入按 before 维切块循环调用）。
- P4 策略（低价值联动）：单组冒烟——三 dtype × 尾轴 dim + 一组中间轴 dim 对 golden；验收标准「无负引导」。

## 10. Agent actions（生成指引）

1. 识别类别：split-view 二目 elementwise，AIV-only，无 dot/规约/广播；单输入（内部两视图）单输出。
2. **先查符号再写公式**：sigmoid 指数必须是 **−a**（β=−1.0 在子类 InitGmBuffer，single.hpp:45）——这是本卡第一语义点。
3. **必须保留**：① 恒 fp32 计算域（**fp16 也升**——与 abs 卡相反，勿跨卡抄习惯）；② dim 折叠三段乘积 + 负 dim 归一；③ a_off=offs+i·HA 的化简 decode（勿写 j/k 双重 div/mod，禁 %）。
4. **不要复刻**：2D base-tile 优选、DataCopy 硬件上限、DataCopyPad 双路径、PipeBarrier 逐条同步、beta 的 GM 属性痕迹。
5. dtype 无分支：指针推导 + 恒 fp32 路径覆盖三型。
6. 分核：输出展平 1D + persistent-core 连续段；grid 核数运行时查询。
7. numel ≥ 2³¹ 超域：wrapper 断言 + 卡片 §9 处置（按 before 切块），勿在 kernel 引 int64 向量 decode（i64 div 标量化 [SRC]）。
8. 优化项（非语义）：dim=−1（after=1）时 HA=H，整数除法仍在——如需极致可加 AFTER_IS_ONE constexpr 直通，benchmark 通过后再做。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化（**confidence: inferred**；P2.5 skipped: low-value，证据上限 [SRC]/[DOC]）。锚点指向 `src/ops-nn/activation/swi_glu/`；后端论断经 backend_facts 检索取证（vexp/vdiv dtype、C-001/C-002/C-010、multibuffer 默认值）。未经实测的点：BLOCK 优选、整数除法 decode 的带宽开销（dim 非尾轴场景）、after>1 的访存局部性差异。残留不确定项：TILING_KEY_EMPTY=100 的 host 触发条件未定位（不影响正常语义）；ST csv 形状矛盾根因未追溯（以 golden 为准）；「todo tiling 中 BUFFER 大小刷新」注释（bf16.hpp:37）表明官方 buffer 计算有未完事项，不影响语义。回放验证：P4 降单组冒烟，报告见 `output/swi_glu/validation_report.md`。
