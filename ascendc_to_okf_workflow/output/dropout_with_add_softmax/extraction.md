# P1 语义抽取 — dropout_with_add_softmax（三组件合成）

> 抽取对象：融合算子本体闭源，按「组件 1 AxpyV2 → 组件 2 SoftmaxV2 → 组件 3a GenMask + 3b DoMaskV3」分层抽取，章节号对齐详设模板。
> 数据流 wiring（推断，manifest §数据流推断）：`t = x1 + alpha·self` → `s = softmax(t, -1)` → `result_softmax = s`；`result_dropout = s·mask/keep_prob`；`mask` 同输出。

## 1.1 功能与公式

- `t = x1 + alpha · self`（axpy_v2 README 公式；PTA 文档 "y=x1+self*alpha"）
- `s_i = exp(t_i) / Σ_j exp(t_j)`，沿 dim=-1（softmax_v2 README；PTA 文档）
- `result_dropout_i = mask_i=1 ? s_i/keep_prob : 0`，keep_prob = 1 − prob（drop_out_do_mask_v3 README 公式 + kernel 锚点；PTA 文档"1/prob"为错误表述，见 manifest 裁决 2）
- 三输出：mask（保留/丢弃记录）、result_softmax（= s）、result_dropout（PTA 文档返回值说明）

## 2. 原型与约束（PTA 文档契约）

| 参数 | 方向 | 类型 | 约束 |
|---|---|---|---|
| self | 输入 | Tensor fp16/fp32（bf16 待确认） | 4D (N,C,H,W) |
| x1 | 输入 | Tensor 同 self | 4D 同 shape |
| alpha | 属性 | float | 正负浮点 |
| prob | 属性 | float | 丢弃概率 [0,1)；≥1 参数非法 |
| dim | 属性 | int | **仅支持 -1** |
| mask | 输出 | Tensor | 文档称 shape=(N,C,H,W)（⚠️ 与组件 bit-packed 布局矛盾，缺省项 2） |
| result_softmax | 输出 | Tensor | (N,C,H,W)，dim 维和为 1 |
| result_dropout | 输出 | Tensor | (N,C,H,W) |

约束（PTA 文档）：H,W ∈ {128,256,384,512}；(N·C)%32==0；self/x1 同 shape。

## 3. 规格约束

- 支持平台：组件 3a 仅 ascend950（stateless_drop_out_gen_mask README 产品表）；组件 1/2/3b 均含 950 及 A2/A3
- dtype：fp32/fp16（bf16 组件均支持；PTA 文档未列 bf16，标注待确认）
- 确定性：result_softmax 确定；mask/result_dropout 依赖 philox 随机源（seed/offset 由 PTA 闭源层生成）

## 4. 组件语义分层抽取

### 4.1 组件 1：AxpyV2（ops-math/math/axpy_v2/）

- kernel 壳结构同 add：`BroadcastSch<Mode, OpDag>` 4+ 路 dtype 分派（axpy_v2_apt.cpp:31-48）
- **fp16/bf16 精度链**：Cast<fp32,NONE> 三输入 → **`Vec::FusedMulAdd<fp32>`**（x2·alpha+x1 单舍入 FMA，axpy_v2_dag.h:48-53）→ Cast<T,RINT> 回写
- fp32 路径：FusedMulAdd 直出 fp32（dag.h:63-66）
- alpha 是**张量输入**（GEIR 侧 (1,)）；PTA 侧为 float 标量属性

### 4.2 组件 2：SoftmaxV2（ops-nn/activation/softmax_v2/）

- 内部全 fp32：输入 CastToFp32From（softmax_v2_base.h:68-77），计算 fp32，回写按输出 dtype
- 6 个 arch35 变体：ar/ara × {full_load, recompute, small_r}，按 A（外积行数）/R（轴长）形态由 tiling 选择（op_host/arch35/ 6 个 tiling cpp）
- R≤512（本算子 W 上限）⇒ 全载路径：减行最大值 → exp → ReduceSum（LastReduceSum，base.h:493-572）→ 除
- axes 仅支持 1 维（def.cpp），PTA dim=-1 对应

### 4.3 组件 3a：stateless_drop_out_gen_mask（ops-math/random/）

- **philox 布局**（stateless_drop_out_gen_mask_pt.h）：
  - `PhiloxRandom<10>`：10 轮 philox4x32（pt.h:209）
  - key = (seed_lo32, seed_hi32)（pt.h:99-102）；counter = 4×u32 ← 2×u64 offset（pt.h:104-119）
  - 每 counter 产 4 个 u32（RESULT_ELEMENT_CNT=4，pt.h:28）；tile 间 counter 推进 `Skip(ceil(n/4))`（pt.h:218-235；Skip 进位链 random_kernel_base.h:346-371）
  - ⇒ **元素 i 的随机数 = philox10(key, counter₀ + ⌊i/4⌋)[i%4]**
- **归一化**：`u = x·2⁻³² + 2⁻³³`（CURAND_2POW32_INV，开区间 (0,1)，pt.h:29, 183-184）
- **保留判定**：`mask_bit = (u < keep_prob)`（CompareScalar CMPMODE::LT，pt.h:201）
- **输出**：bit-packed uint8，8 元素/字节，字节偏移 = 元素偏移/8（pt.h:32, 129, 229-231）

### 4.4 组件 3b：DropOutDoMaskV3（ops-math/random/）

- `prob_reciprocal = 1.0f / keep_prob`（drop_out_do_mask_v3.h:194）
- 一般路径：`y = mask ? x·prob_reciprocal : 0`（Muls + Select(select0=0)，L97-98）
- 边界：keep_prob==0 → Duplicate 全零（L256-265）；keep_prob==1 → 直通拷贝（L267-280）；判定用 float eps 比较（IsProbEqual L80-84）
- b16：升 fp32 乘、CAST_RINT 回写（L111-118, 75-77）
- mask 输入 bit-packed → MaskDeInterleave 解包（L213-217）

## 5/6. 片上内存与流水（组件侧，不进卡片主体）

- genmask/DoMask：纯 elementwise，UB 双缓冲队列（BUFFER_NUM=2 / DOUBLE_BUFFER=2），分核 = 元素均分（tiling normalCoreProNum/tailCoreProNum，domask L131-142）
- softmax：A×R 分核，R 全载或分切
- **P2 丢弃理由**：融合后 Triton 单 kernel 行式分核（W≤512 一行全载），组件各自的 UB 切分/双缓冲设计不需要翻译

## 精度策略汇总（卡片 §9 依据）

| 环节 | 精度事实 | 锚点 |
|---|---|---|
| axpy | fp16/bf16 → fp32 FMA 单舍入 → RINT 回写 | axpy_v2_dag.h:48-53 |
| softmax | 内部 fp32；减最大值；fp32 除 | softmax_v2_base.h |
| domask | b16 → fp32 乘 → RINT；fp32 直算 | domask L97-118 |
| genmask | u ∈ (0,1) fp32 与 fp32 keep_prob 比较 | pt.h:183-201 |

## P1 质量门

- 全部论断带锚点 ✅；文档与实现偏差 2 处已裁决（缩放系数、mask 布局——后者留待 device 实测）
- 数据流 wiring 为推断（唯一无源码环节），验证预案见 manifest 缺省项 1
