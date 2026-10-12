# P1 语义抽取 — flash_attention_score（v2 重新提取，服务 level4 SelfAttention）

> 抽取对象：`src/flash_attention_score_v2/`（CANN ops-transformer @ d28d8f99a；arch32=A2 多模板主线 + arch35=950 PR/DT 单模板）。本目录为 2026-10-10 重新提取版，独立于旧 `src/flash_attention_score/` 与 `output/flash_attention_score/`；本文件命名沿用 `<op>_v2`。
> 本抽取出**注意力核**语义；level4 自带的 4 段 nn.Linear 投影 = CANN MatMul，在 §2.2 显式拆出（P2 需补回）。

## 1.1 数学语义

level4 **SelfAttention**（`ascendc-kernelgen-data/npu-kernelbench/data/kernel_generator/level4/SelfAttention/`，golden=`ascendc-kernelgen-data/npu_benchmark/level4/18_SelfAttention.py`）整体 = **4×投影(MatMul) + 缩放点积注意力**：

1. 投影：`q=Wq·x, k=Wk·x, v=Wv·x, out=Wo·z`（`nn.Linear(D→D, bias=False)`，seed=42 权重，作用于 `x[B,S,D]`）。
2. `scores = (q·kᵀ)/√head_dim`，head_dim=D/n_heads，q/k/v reshape `(B,H,S,hd)` 再 `transpose(1,2)`。
3. `weights = softmax(scores, dim=-1)`（golden 用 online softmax，见下）。
4. `z = weights·v`（fp32 累加），`z→(B,S,D)`。
5. `output = Wo·z`。

CANN **FlashAttentionScore kernel 只含 2~4 的注意力核**：`softmax(QKᵀ·scale)·V`，online softmax、fp32 累加。

**golden 实现**（`tests/pytest/cpu_impl.py:40-70`，验收唯一真值源）：
- `qk = matmul(q, k.permute(0,1,3,2))`（cpu_impl.py:45）
- scale/pse 顺序随 `pse_type`：`qk=qk.add(pse).mul(scale)`（pse_type 分支 :50）或 `qk=qk.mul(scale).add(pse)`（:52）
- `tsoftmax`（:20-26）：`x_max=max(x,-1)` → `x_sub=x-x_max` → `y=exp(x_sub)` → `x_sum=sum(y)` → `res=y/x_sum`，返回 `(res, x_max, x_sum)`
- `skip_invalid_row`（:29-37）：若某行 `x_max==-40000`（稀疏/band 遮蔽断言），整行 softmax=0、`x_max=-inf`、`x_sum=+inf`
- dropout（:62）：`drop_res = softmax_res * drop_mask * (1/keep_prob)`
- `out = matmul(drop_res, v)`（:66）
- **`x_max/x_sum` broadcast 到尾维 ×8**（:68-69）——对应 kernel 每行 softmaxReduceSize=8（`s1s2_bn2gs1.h:207`）
- scale 缺省 `1/√d`（:107）

**kernel 侧五步**（`docs/FA算子设计介绍.md:7-18`）：
1. `bmm1 = Q·Kᵀ` → 加 pse → `×scale_value` → atten_mask select 遮蔽（被遮位设为负极小值，softmax 后≈0）（:9）
2. FlashSoftmax 沿 **Skv(S2) 方向切分**，逐 SkvSplit 在线更新（:11-17）
3. `i=0`：`MM[PV]` 直接存 `ub_attention_out[0]`
4. `i≥1`：先 `×exp[i]`（`exp[i]=e^{max_{i-1}-max_i}`，:14）再加本次 `MM[PV]` 存 `ub_attention_out[1]`（刷新流程，:17）
5. 除 sum 后移到输出前：末尾 `ub_attention_out / softmax_sum` 写回 `attention_out`（:18）

## 2. GEIR kernel 原型（`op_host/flash_attention_score_def.cpp`）

`REGISTER_OP FlashAttentionScore`（def.cpp:20-22, :337 `OP_ADD`）。

### 2.1 输入/输出/属性清单（def.cpp:23-169）

**REQUIRED 输入（def.cpp:23-40）**：`query`、`key`、`value`——`{FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16,FP32,FP32, HIFP8,HIFP8,E5M2,E5M2,E4M3,E4M3, HLFP8,HLFP8,E5M2,E5M2,E4M3,E4M3}`，FORMAT_ND，AutoContiguous。

**OPTIONAL 输入**：
- `real_shift`(pse)（:41）`{FP16,BF16,FP16,BF16,FP32,FP32,FP32,FP32,FP32,FP32,FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16}`
- `drop_mask`（:47）`UINT8`；`padding_mask`（:53）`{FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16,FP32,FP32,...}`
- `atten_mask`（:59）`{UINT8,UINT8,BOOL,BOOL,UINT8,UINT8,BOOL,BOOL,UINT8,BOOL,...}`（位数=22 平台档）
- `prefix`/`actual_seq_qlen`/`actual_seq_kvlen`/`q_start_idx`/`kv_start_idx`（:65-94）均 `INT64` + `ValueDepend(OPTIONAL)`
- `d_scale_q/k/v`（:95-112）、`sink`（:125）、`p_scale`（:131）均 `FP32`
- `query_rope`/`key_rope`（:113-124）`{FP16,BF16,FP16,BF16,FP16,BF16,FP16,BF16,FP32,FP32,HIFP8,E5M2,E4M3,...}`

**REQUIRED 输出（:137-156）**：`softmax_max`(FP32)、`softmax_sum`(FP32)、`softmax_out`(`{FP16,BF16,...,FP32}`)、`attention_out`(`{FP16,BF16,...,FP32}`)。

**属性（:157-169）**：
| 属性 | 缺省 | 说明 |
|---|---|---|
| `scale_value` | 1.0 | QKᵀ 缩放（def:157） |
| `keep_prob` | 1.0 | dropout keep 概率（def:158） |
| `pre_tockens`/`next_tockens` | 2147483647 | sparse band 前后窗（def:159-160） |
| `head_num` | **REQUIRED** | 头数（def:161） |
| `input_layout` | **REQUIRED** | `NSD/BSH/SBH/BNSD/TND/...`（def:162） |
| `inner_precise` | 0 | 精度策略档（def:163） |
| `sparse_mode` | 0 | 见 §3（def:164） |
| `pse_type` | 1 | pse 与 scale 组合顺序（def:165；cpu_impl:50/52） |
| `seed`/`offset` | 0 | dropout 随机种子（def:166-167） |
| `out_dtype` | 0 | 输出位宽覆盖（def:168） |
| `softmax_out_layout` | "" | softmax_out 布局（def:169） |

### 2.2 与 level4 的接口差（投影拆分指令）

- level4 接口 `run(x, n_heads)`：**先做 4 段 nn.Linear**，注意力核才吃 q/k/v。CANN FAS 直接吃 q/k/v、head_num 来自属性。
- → **P2 卡片必须补「前投影四段 = 4×MatMul(D,D × B,S,D)」**；`head_dim=√ 缩放`对应 FAS `scale_value = 1/√head_dim`（golden :107）。
- level4 `output` = FAS `attention_out` 再经 `Wo` 投影；FAS 的 `softmax_max/sum/out` 是额外中间产出（level4 golden 内部重算 softmax，不建模，新版 golden 用重算 lse 自洽）。

## 2.1 原型约束

- 支持静态/动态 shape、动态 rank、动态 format（def:172-176 `DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)`）。
- `PrecisionReduceFlag(true)`（def:177）——允许精度降低档（对应 inner_precise）。
- Tiling 输入数据依赖项 `{7,8,9,10,11}` = prefix/actual_seq_qlen/actual_seq_kvlen/q_start_idx/kv_start_idx（`flash_attention_score_tiling.cpp:400`）——变长 TND 场景依赖这些 host 需读取的索引张量。
- 变长约束：VarLen 场景要求 q 与 kv 等 S 维度约束（`arch35/flash_attention_score_tiling_regbase.cpp:314`）。

## 3. 规格约束

**平台**（def.cpp AddConfig）：
- `ascend950`（def:183）→ 独立专属 config（含 fp8/HIFP8/E5M2/E4M3 全 dtype 面, def:23-25）
- `ascend910b`、`ascend910_93`（def:332-333）→ 共享 config（仅 FP16/BF16/FP16/FP16/FP32/FP32 10 档, def:186-188）

**dtype 面**：FP16/BF16/FP32 + **预缩放 fp8 量化双 scale（d_scale_q/k/v 生效, def:95-112, docs）**样例为 fp8 通路。level4 三 dtype（fp16/bf16/fp32）全覆盖。

**sparse_mode 语义**（tilingkey Sparse 位, `arch32/flash_attention_score_template_tiling_key.h`）：
`0=ALL, 1=NONE, 2=ANY, 3=CAUSAL, 4=BAND, 5=PREFIX, 6=BAND_COMPRESS, 7=RIGHT_DOWN_CAUSAL, 8=RIGHT_DOWN_CAUSAL_BAND, 9=BAND_LEFT_UP_CAUSAL`。**因果 = sparse_mode=3**（对角线带宽即自注意力掩蔽）。

**架构差异（arch32 A2 vs arch35 950）**：
- arch32 (=A2 训练) 走高阶 API，多模板（见 §4.1.1）
- arch35 (=Ascend 950PR/DT, `NpuArch::DAV_3510`）走**低阶 API**，一套模板支持全量 shape（`docs/FA算子设计介绍.md:166-168, 285`）；`tiling.cpp:358` 若平台==DAV_3510 分支转 regbase tiling
- arch32 模板不切 S2 到多核（docs :198）；950 一律兼顾

## 4.1.1 分核策略

**切分轴全集**：B、N2(kv 头)、G(query/kv 头编号)、S1(query S)、S2(key S) 共 5 轴；**先核内再核间**，核内按基本块大小选轴，核间把核内剩余轴合并后按 AI Core 核数切分（`docs/FA算子设计介绍.md:105`）。

**arch32 六模板（按优先级，序号小者先匹配, docs:107-163）**：
| 模板 | 切分轴 | 进入条件 |
|---|---|---|
| TNDSameAB（docs:126） | UB 切 S1 | `(B≥4 & accumS1≥8192 & accumS2≥8192 & maxS1≥512 & maxS2≥512) or (maxS2≥5120 & maxS1≥5120)` |
| TND（:133） | UB 切 S1S2 | TND 且不满足 TNDSameAB |
| SameAB（:139） | UB 切 S1 | 非 FP32, `S2≥512 & D%16!=0 or D=96`, or `S2>1024 & 128<D<196` |
| S1S2（:145） | UB 切 S1S2 | 不满足上述 & `S2>1024` |
| S1（:151） | UB 切 S1D | 大 shape 地板表达式（:153） |
| B（:155） | 核间切 B | 兜底 |

**三类主实现对应的「核间切 B/N2/G/S1 + 核内切」模板（docs:198-244）**：
1. **通用模板 S1S2**（`s1s2_bn2gs1.h`）：核间切 B/N2/G/S1、核内切 S1+S2；S2>1024 用（docs:199-212）；CV 基本块 `S1.i=64/128, S2.i=1024`。
2. **S1 模板**（`s1_bn2gs1.h`）：核间切 B/N2/G/S1、核内切 S1；`128<S2≤1024` 或大 shape（docs:215-232）；S2 不切。
3. **B 模板**（`bn2gs1s2_b.h`）：核间切 B、Cube 不切 S1/S2 把 B.i·N2·G 开循环 batch matmul，Vector 把 B.i·N2·G·S1 综合切分（docs:235-244）。

**arch35 950 模板（docs:246-257）**：核间切 B/N2/G/S1 + 核内切 S1S2（`kernel_train.h`，regbase 低阶）；`S1.i=128（少量 64）, S2.i=128（少量 256）`。

**持core 循环结构**（s1s2_bn2gs1.h Process）：每个 program 处理**连续** `totalSize/splitFactorSize` 的压平任务块（`multiCoreInnerOffset = blockIdx × splitFactorSize`, :773），核内对 S1（外层多轴复合 idx:986-994 兼 B/N2/G/S1 笛卡尔序）再对 S2 开内层循环（:861-965）——即 **task-idx 经 `(bo,n2o,go,s1o)` 解压取轴**，S2 不进多核。尾块由 `notSecondLast/notLast` 两哨兵标记（:908-912），尾块只做 S2 收尾、不再 IterateBmm1。

## 4.1.2 单核内切分（tilingdata）

**基本块尺寸常量**（`arch32/flash_attention_score_tiling_general.cpp:56-126`）：
- BMM1 基本块 `M=128,N=128,K=64/128`（:82-86）；BMM2 `M=128,N=128` 或 `M=64,N=64`（:91-94）
- HIGH_PERF 档 `S1=128,S2=128`（:56-59）；UB 基本块上限 `8×1024`（:98）
- TND S1 基本块 128（:120）；S1 L1 携带上限 128（:126）

**CV 基本块设计**（docs:38-53）：Vector fp32 → 基本块 `8×1024`；Cube f16→`128×128`；`nRatio=8` 配比出 `128×1024`（C 侧 1 块对 V 侧 16 份）。UB 内再按 S1 配比 `S1_c/S1_v=128/8` 多次 Vector：`copy_gm_to_ub(S1_v_i×S2_v_i)` → Vector → `copy_ub_to_gm`（docs:47-50）。Cube Buffer 有限，CV 基本块进核内再切（docs:186-190，核内块 Cube 32KB、Vector 32KB）。

## 4.1.3 tilingdata / tilingkey

**tilingkey 位布局**（`arch32/flash_attention_score_template_tiling_key.h`，64-bit）：
| bit | 字段 | 取值 |
|---|---|---|
| 3-0 | KernelTypeKey（空tensor:1） | |
| 7-4 | UB0 | B/N2/G/S1/S2/D/NONE |
| 11-8 | UB1 | 同上 |
| 15-12 | Block（核间轴） | 同上 |
| 17-16 | ImplMode | `0=AA_HIGH_PRECISION,1=AA_HIGH_PERFORMANCE,2=AA_INVALID_LINE_HIGH_PRECISION` |
| 19-18 | DataType | `0=FP16,1=FP32,2=BF16,3=FP16_PRECISION` |
| 22-20 | Layout | `0=NONE,1=BSH,2=SBH,3=BNSD,4=TND` |
| 23 | Bmm1Format | 0=ND/1=NZ |
| 24 | Bmm2Source | 0=GM/1=L1 |
| 28-25 | Sparse | 0..9（见 §3） |
| 30-29 | BigDoubleBuffer | NONE/BIG_BUFFER/BIG_DOUBLE_BUFFER |
| 31 | HasDropOut | bool |
| 32 | HasAttenMask | bool |
| 33 | HasPse | bool |
| 34 | EnableL1Reuse | bool |
| 35 | HasRope | bool |
| 36 | MatmulPolicyType | NORMAL/UNSPLITK |
| 40-37 | S1TemplateType | ALIGNED_16..128 |
| 44-41 | S2TemplateType | 同上 |
| 48-45 | DTemplateType | UNKNOW/80/96/128 |

**S1S2 模板默认组合**（template_tiling_key.h `ASCENDC_TPL_SEL`）：`UB0=S1, UB1=S2, Block=NONE, ImplMode∈{0,2}, DataType=3`, Lay=BSH/SBH/BNSD, Sparse=ANY, BigDoubleBuffer, HasDrop/Atten/Pse∈{0,1}, Bmm2Source=GM。

**模板注册优先级**：`REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, Class, DAV_3510, prio)` ——arch35 已见 `DropMask=81`, `Basic=83("S1S2Const")`（`arch35/flash_attention_score_tiling_dropmask.cpp:84`、`_basic.cpp:213`），VarLen=`"VarLenConst"`（`_varlen.cpp:21`）；**数字小者优先**。

## 5. 片上内存预算（s1s2_bn2gs1.h InitBuffer :503-522）

| TBuf | 容量 | 语义 |
|---|---|---|
| maskTBufPing/Pong | atten(≈9k)/drop(16k) | mask 双缓冲（:507-510） |
| pseTBuf | 16384B | pse（:511） |
| stage1PingBuf/PongBuf | — | bmm1 结果双缓冲（stage1, :212-213） |
| stage2TBuf | stage2Size×4B≈32K | t.c（bmm2 出，:513） |
| commonTBuf | 32K | t.b 复用空间（:514, :221） |
| softmaxSumBuf[2] | s1Base×4×8 | online sum 双缓冲（:516-517） |
| softmaxMaxBuf | s1Base×4×8 | online max（:518） |
| softmaxExpBuf[2] | s1Base×4×8 | 刷新 exp（:519-520） |
| softmaxTempBuf | s1Base×32 | softmaxReduceSize=1 临时（:521-522） |

softmaxReduceSize 常量=8（:207）——每行一个硬件「8 元素 reduce 单元」，与 golden ×8 broadcast 对齐。950 用 UB pathway：AIC 直接输出到 UB 供 AIV vec 运算（docs:59），D_v>128 时 mm2 先存 GM。

## 6. 流水编排（CV 融合）

**Compose 阶段**（docs:20-28）：`Process` → `IterateBmm1`(Cube1) → `ProcessVec1`(Vector1) → `IterateBmm2`(Cube2) → `ProcessVec2`(Vector2)。

**3-slot 任务流水**（s1s2_bn2gs1.h Process :861-965）：`extraInfo[taskId%3]` 环形 3 槽，每轮：
1. `WaitBmm1Result(extraInfo[(taskId+2)%3])`（:864/929）
2. `IterateBmm1`（:869-877, NZ 分支 needNz2Nd:870）
3. `ProcessVec1(extraInfo[(taskId+2)%3])`+`SetFlag<MTE3_MTE2>`（:881/947）
4. `WaitBmm2Result()`（:887/953）
5. `WaitFlag<MTE3_MTE2>` → `IterateBmm2`（:891/957）
6. `ProcessVec2(extraInfo[(taskId+1)%3])`（:895/962）

**Cube 双发/preload**：A2 → Cube1/Cube2 衔接达成 **Cube bound**，MTE2 bound（docs:80-84）；950 preload=3（完成 3 次 mm1 才开 mm2, docs:88）。Vector 侧 ping-pong double buffer（docs:65-74）。

**特殊模板**：`_sab` 后缀=**Cube 为主核**模板（V0 统一发起 Matmul，结果 V0/V1 各半，复用左矩阵+L0C 累加）（docs:263-273）；regbase 低阶模板完全 Cube 发起、同步通知 Vector（docs:279-283）。

## 7. 精度策略

- 全部**中间量 fp32**（`INPUT_T/T` 双模板参数，输入 T 位宽、计算 T=fp32；`:767-769`）。
- `inner_precise` 档切换 `ImplMode`（HighPerformance 允许 f16 中间 / HighPrecision 全程 fp32）。
- online softmax 用**减 max + exp + 统计 sum** 全按行（row），除 sum 后移（docs:17-18）。
- 预缩放 fp8：`d_scale_q/k/v × query/key/value` 入 Cube、输出 `p_scale` 还原（def:95-112,131）。
- bmm1 可走 NZ 布局并需 NZ→ND 置换：kernel 依 tilingkey `Bmm1Format` 分支 `this->bmm1Nz` vs `this->bmm1`，`extraInfo.needNz2Nd` 决定（`s1s2_bn2gs1.h:869-877,936-940`）；`Bmm2Source∈{GM,L1}` 决定 PV 源（template_tiling_key.h Bmm2Source 位）。

## 8. 测试设计

- **CPU-SIM unit**（`tests/pytest/`）：`cpu_impl.py` 为 golden（见 §1.1），`npu_impl.py` 调用，`test_flash_attn.py`/`test_case.py` 汇总，覆盖 normal/TND/varlen/空 tensor/稀疏。
- **UT**（`tests/ut/`）：`op_api/test_aclnn_flash_attention_score.cpp` + `_v4.cpp`；`op_host/arch32//arch35/` tiling 单测（**期望 tilingkey 断言**）。
- **examples**（`examples/`）：test_aclnn_flash_attention_score.cpp / `_v2_fp32.cpp` / `_varlen_v4_fp32.cpp`（对外验收入口，fp32 标杆）。
- **open-offline golden 事件**：level4 SelfAttention golden 在 `npu_benchmark/level4/18_SelfAttention.py`（Model 重算 softmax，接受 lse 自洽）。相关工作流已验证：flash 系 Model 须避免 `_apply` 等 nn.Module 保留名、golden 需与被测 kernel 同舍入序列（凡 fp32 计算域一律全 dtype 升 fp32）。

## 9. 材料清单（P0 五类核对）

| 类别 | 位置 | 状态 |
|---|---|---|
| 原型/kernel 入口 | `op_host/flash_attention_score_def.cpp`、`op_kernel/flash_attention_score.cpp`/`_apt.cpp` | ✅ |
| kernel 模板 | `op_kernel/arch32/*.h`（s1s2/s1/b/sab/var_len）、`arch35/kernel_train.h`+entry_regbase.h | ✅ |
| host tiling | `op_host/flash_attention_score_tiling.cpp`+`_common.h`、`arch32/*general.cpp`、`arch35/{regbase,dropmask,basic,varlen}.cpp` | ✅ |
| aclnn API | `op_api/aclnn_flash_attention_score.{h,cpp}`、`flash_attention_score.{h,cpp}` | ✅ |
| 文档/测试 | `docs/FA算子设计介绍.md`+aclnn*.md、`tests/`、`examples/` | ✅ |

**缺省说明**：无（五类齐全）。shell/Triton 不可达项判归由 P2 完成（本文件不做后端论断）。

## 来源承诺

所有 `file:line` 锚点基于 `src/flash_attention_score_v2/` 与 `ascendc_kernelgen_data/npu(-kernel)bench/level4/SelfAttention/`、`npu_benchmark/level4/18_SelfAttention.py`。若后续以 arch35 950 为主纲（level4 生成目标多为开源 NPU），P2 建议以 regbase + kernel_train.h 为主参考链、arch32 多模板佐证计算数学等价性（四 layout 归一化）。