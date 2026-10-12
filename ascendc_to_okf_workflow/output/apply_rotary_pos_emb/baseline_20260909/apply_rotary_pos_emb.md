---
schema_version: okf.v1
kind: operator
type: operator_spec
source_family: golden
title: ApplyRotaryPosEmb（RoPE 旋转位置编码 q/k 双路融合）Ascend C → Triton 语义转化卡
description: out = x·cos + rotate(x)·sin 的 q/k 双路融合 AIV-only 向量算子，支持 half/interleave/quarter 三种旋转模式 × BSND/SBND/BNSD/TND 四种布局 × cos/sin 批量广播 × partial rope。arch35 主线为 4 tilingkey/3 kernel 类（计算数学相同、循环结构不同），Triton 侧收敛为单 kernel + ROTARY_MODE constexpr + strides 布局归一化；关键决策点：行空间 persistent-core 分核、fp32 计算域、cos/sin (b,s) 组级循环不变量外提（实测 +100%）、interleave 连续 load + tl.split/tl.interleave 寄存器重排（strided-2 实测有后端正确性缺陷）、partial rope 直通段处理。
tags: [rope, rotary_pos_emb, position_embedding, elementwise, fused_qk, layout_stride, broadcast, fp16, bf16, fp32, arch35, ascend950, aiv_only, apply_rotary_pos_emb]
confidence: verified
status: verified
resource: src/ops-transformer/posembedding/apply_rotary_pos_emb/（zmm_source_retrieval 本地快照，cann/ops-transformer @ d28d8f99a8858b71caa27a535da32bb5ba68ce2a）
created_at: "2026-09-08T00:00:00+08:00"
updated_at: "2026-09-09T00:00:00+08:00"
---

sources:
  - url: src/ops-transformer/posembedding/apply_rotary_pos_emb/
    role: primary

# ApplyRotaryPosEmb（旋转位置编码，q/k 双路融合原地更新）

对 query、key 两路执行 RoPE：`out = x·cos + rotate(x)·sin`，rotate 结构由 rotary_mode 决定（half/interleave/quarter）；推理网络为省一次 launch 将两路融合（README.md:18）。**AIV-only 纯向量**（arch35 入口 AIC 直接 return，apply_rotary_pos_emb_apt.cpp:31；legacy 入口 `KERNEL_TYPE_AIV_ONLY`，apply_rotary_pos_emb.cpp:24），无 Cube/规约/atomic。**非壳算子**：tiling×5 + kernel×7 全部本体在仓内。arch35（ascend950）为主线：4 个 tilingkey（20010/20011/20020/20030）对应 3 个 kernel 类，**计算数学完全相同**（共享 common.h 的 Half/Quarter/Interleave VF），差异仅在循环结构与 cos/sin 广播处理——Triton 侧据此收敛为**单 kernel + constexpr**。legacy（910b/910_93/310p/kirin）为另一套（key 1/3/4，layout 仅 BSND/TND、D∈{64,128}、rotary 仅 half），见 §3 架构差异。语义分层：aclnn V1 = V2 锁 rotary_mode="half"（aclnn_apply_rotary_pos_emb.cpp:27-29），无其他参数消化；benchmark 已有精确 case（ascendc-migration/level2/5_ApplyRotaryPosEmb，§9）。

## 1. 功能与公式（详设 §1.1）

- 功能：RoPE 旋转位置编码，q/k 双路融合、原地更新语义（输出与输入同名同形）。
- 计算公式（README.md:19-136）：`q_embed = query·cos + rotate(query)·sin`，`k_embed = key·cos + rotate(key)·sin`；cos/sin 的 N 维恒为 1（同层所有 head 共享一组旋转角），B 维可广播（CosB∈{qB,1}），末维 realDim ≤ D（partial rope：仅前 realDim 维旋转，[realDim, D) 直通，tiling_base_arch35.cpp:319）。

  | rotary_mode | rotate(x) 定义 | 逐元素展开（pair 索引 i） |
  |---|---|---|
  | half（默认） | cat(−x₂, x₁)，x₁/x₂ 为前后半场 | out[i]=x[i]·c[i]−x[i+h]·s[i]；out[i+h]=x[i+h]·c[i+h]+x[i]·s[i+h]，h=RD/2 |
  | interleave | 奇偶对 (x[2i],x[2i+1])→(−x[2i+1],x[2i]) | out[2i]=xe·ce−xo·se；out[2i+1]=xo·co+xe·so |
  | quarter | cat(−x₂,x₁,−x₄,x₃)，四等分 | 段内同 half：段 1↔2、3↔4，q=RD/4 |

  展开式已与 kernel VF 逐行核对一致（half：common.h:111-119；quarter：common.h:193-209；interleave：common.h:270-280）。
- **语义分层**：GEIR kernel 本体 = 上式（layout/rotary_mode 是编译期分派维度，不进数据流）；aclnn 层 V1=V2 锁 half，无参数消化；benchmark 层已有精确 case（45 case，定义域 ⊆ 算子定义域，无超集偏差，§9）。
- 计算流程概述（以 ABAAndBA 为例，aba_and_ba.h:168-244）：S 轴 ubFactorS 步进 → B 轴循环（BA 广播时 cos/sin 载一次复用）→ 先 QN 全循环后 KN → 每块 CopyIn（LoopModeParams 多维跨步 DMA）→ Compute（VF，fp32 Reg 域）→ CopyOut。
- 锚点：`op_kernel/arch35/apply_rotary_pos_emb_common.h:66-328`（三模式 VF）、`apply_rotary_pos_emb_apt.cpp:35-59`（key 分派）。

## 2. 原型与约束（详设 §2 / §2.1）

### 输入输出（来源：`op_host/apply_rotary_pos_emb_def.cpp:27-52`）

| 参数名 | 入参类型 | 必选性 | dtype | format | shape | 语义 |
|---|---|---|---|---|---|---|
| query | 输入 | 必选 | FP16 / FP32 / BF16（310p/kirin 仅 FP16/FP32） | ND | 4 维（TND 3 维） | 主输入 |
| key | 输入 | 必选 | 同 query | ND | 与 query 仅 N 维可不同（QN vs KN） | 主输入 |
| cos | 输入 | 必选 | 同 query | ND | N 维=1、B 维∈{qB,1}、末维 realDim≤qD | 旋转角余弦 |
| sin | 输入 | 必选 | 同 query | ND | 与 cos 完全相同 | 旋转角正弦 |
| query | 输出 | 必选 | 同输入 | ND | 与输入 query 相同 | 原地语义输出 |
| key | 输出 | 必选 | 同输入 | ND | 与输入 key 相同 | 原地语义输出 |

四输入均 AutoContiguous（def.cpp:32-50）。

### 属性

| 属性 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| layout | int 可选 | 1 | 1=BSND、2=SBND、3=BNSD、4=TND（def.cpp:53；tiling.h:139-144） |
| rotary_mode | string 可选 | "half" | "half"/"interleave"/"quarter"（def.cpp:54；校验 tiling_base_arch35.cpp:236-238） |

### 原型约束（arch35，tiling_base_arch35.cpp）

- shape 关系：cos==sin shape；cos/sin N 维==1；q/k/cos/sin S 维一致；cosD≤qD、kD≤qD；非 TND 时 kB==qB 且 cosB∈{qB,1}；q/k 输出与输入同形（L158-207）；所有维度为正（L120-140）。
- rotary 约束：**D≤1024**；half/interleave 要 D%2==0；quarter 要 D%4==0（L142-156，D_LIMIT=1024 L33）。
- dtype：六 tensor 全同型，全集 {F32, BF16, F16}（L233-259）。
- infershape：输出 = 输入与 cos/sin 广播对齐；BroadcastDim 放行 dim1>dim2 即 partial rope 包容（infershape.cpp:52-54, 116-125）。
- **文档/实现偏差**：docs V2 称支持空 tensor，但 arch35 tiling 拒绝非正维度（L120-140）——以实现为准。docs V2 内部亦自相矛盾：q/k 行写「不支持空 Tensor」而 cos/sin 行写「支持空 Tensor」（aclnnApplyRotaryPosEmbV2.md 参数表），进一步印证以实现为准。
- **legacy D∈{64,128} 的根因**（docs V2 `ub_required` 公式，aclnnApplyRotaryPosEmbV2.md 约束节）：legacy kernel 把 q/k 全部 head 的整行同时驻留 UB，`ub_required ≈ (q_n+k_n)·lastDim·castSize·2 + …`；tiling.cpp:195-200 的硬编码检查是该公式的保守上界。arch35 改按 ubFactor 循环后此约束消失（D≤1024）。

## 3. 规格约束（详设 §3）

| 类别 | 取值 | 来源锚点 |
|---|---|---|
| 支持 NPU 型号 | ascend950（arch35 主线）；910b/910_93（legacy）；310p/kirinx90/kirin9030（legacy，dtype 裁剪 FP16/FP32） | def.cpp:62-109；config/ 7 平台目录 |
| dtype 全集 | FP16/FP32/BF16（310p/kirin 无 BF16） | def.cpp:20-21 vs 76-106 |
| format | ND（AutoContiguous） | def.cpp:27-50 |
| 确定性计算 | 是（无规约/atomic；16MB workspace 仅系统保留，kernel 数据流不使用） | tiling_ab_arch35.cpp:21、apt.cpp 未读 workspace |
| 交付形式 | aclnn V1/V2 + GEIR | op_api/；examples/ |

### 架构差异（arch35 vs legacy）

| 维度 | arch35（ascend950，本卡主线） | legacy（910b/910_93/310p/kirin） |
|---|---|---|
| kernel 文件 | apt.cpp + arch35/{common,ab,bab,aba_and_ba}.h | apply_rotary_pos_emb.cpp + {base,small,compute_ab,compute_ab_cast}.h |
| tiling 体系 | Regbase 模板注册（优先级 10000 AB → 20000 ABA/BA → 30000 BAB → 40000 legacy 兜底，小者先尝试） | 单 tiling 类（tiling.cpp，注册优先级 40000；非 regbase 平台实际命中它） |
| tilingkey | 20010(ABA)/20011(BA)/20020(BAB)/20030(AB) | 1(SMALL)/3(AB)/4(AB_CAST) |
| layout | 全 4 种 | 仅 BSND/TND（tiling.cpp:188） |
| D 约束 | ≤1024，%2/%4 | **必须 ∈{64,128}**（tiling.cpp:195-200） |
| rotary_mode | half/interleave/quarter | 仅 half（kernel 无模式分支） |
| BF16 | 原生（VF Load/Store 转 fp32） | 计算类型 float 模板参 + CAST_ROUND 回写（apply_rotary_pos_emb.cpp:52-62；compute_ab_cast.h:250-284） |
| 其他 | qkcNum 无上限 | qkcNum≤255（Mul repeatTimes uint8，tiling.cpp:306-308） |

## 4. 分核策略（详设 §4.1.1 → Triton grid 设计）

- Ascend C 事实：三模板分核轴不同——AB 把 BS 打平 `blockFactor=CeilDiv(bs,coreNum)`（tiling_ab_arch35.cpp:69-71）；ABA/BA 按 B×S 二维分核（整除优先单轴、否则双方案比选，tiling_aba_and_ba_arch35.cpp:80-135）；BAB 同构但比选乘 ubFactorS（tiling_bab_arch35.cpp:105-149）。kernel 内均以 ubFactor 步进循环（如 aba_and_ba.h:168-178）。
- **Triton 决策**：
  - 任务轴选择：**行（row = 一个 (b,s,n) 位置的 D 维向量）**。rope 只耦合行内 D 维配对，行间完全独立——三种源码分核轴都是行空间的不同切法，行是公共子结构。rows = B×S×N（每路）。
  - 任务块总量公式：`NUM_BLOCKS = CeilDiv(rows, BLOCK_ROWS)`；**grid = min(NUM_BLOCKS, vector_core_num)**（persistent-core，NPU 上超额 block 串行排队，勿用裸 NUM_BLOCKS grid）。
  - 核分配范式（persistent-core + 核内连续段循环，与源码「每核 blockFactor 连续段 + 核内 ubFactor 循环」同构）：

    ```python
    pid = tl.program_id(0); num_progs = tl.num_programs(0)
    total_blocks = tl.cdiv(rows, BLOCK_ROWS)
    per_prog = tl.cdiv(total_blocks, num_progs)
    for blk in range(pid * per_prog, tl.minimum((pid + 1) * per_prog, total_blocks)):
        row = blk * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)   # 连续行段
        row_m = row < rows                                   # 尾块统一 mask
        n = row % N; s = (row // N) % S; b = row // (N * S)  # 行号→(b,s,n)
    ```

  - 行号解码的物理布局由 strides 吸收（§5），与 layout 无关；核数动态读 `torch_npu.npu.npu_config`，禁硬编码。
  - Q/K 两路：**两次 launch**（共享 kernel）——P4 实测融合单 launch 反而劣化（4.2060→3.8252，验证记录见 validation_report.md）；cos/sin 体量 S×RD（KB 级），第二次 launch 全命中 L2，替代了源码「先 Q 后 K 复用 cos/sin UB」（aba_and_ba.h:214-234）与「Q/K 跨界块拼接」（ab.h:142-161）两个机制——两者均勿复刻。
  - **核内循环结构（P4 实测收敛，优于逐行打平）**：以 (b,s) 组为外循环、组内 n 行展开为 tile——`for bs in ...: cos/sin 载一次（标量 c_base）；x tile [BLOCK_N, BLOCK_DH]`。理由：cos/sin 在组内恒等，逐行重载实测慢一倍（循环不变量外提单点 +100%，2.0956→4.2060）；且 bs 为**标量**循环使行解码退化为标量 div/mod，规避了「向量 div/mod × 特定访存模式」的后端缺陷面（§7 ⚠️）。BLOCK_N = next_pow2(N)（N≤128 时单 tile 覆盖整组）。

## 5. BLOCK / Tiling 参数（详设 §4.1.2 / §4.1.3）

### Ascend C tilingdata → Triton 超参映射

| tilingdata 字段 | 语义 | host 侧计算式 | Triton 对应 | 候选值/推导式 |
|---|---|---|---|---|
| realDim / D | 旋转维 / 头维 | SetBsnd 按 layout 抽取（tiling_base L284-320） | `RD` 形参；D 不进 kernel | RD≤1024 |
| dSplitCoef∈{2,1,4} | 模式切分系数 | ConvertRotaryMode（L270-282） | 配对宽 `pair`：half/interleave=RD/2、quarter=RD/4 | wrapper 计算传入 |
| dAlign | 32B 对齐行宽 | CeilAlign(RD/coef, 32B/sizeof(T))×coef（tiling_ab L68） | `BLOCK_DH: tl.constexpr` | `next_pow2(pair)` |
| ubFactorS/B/QN/KN | 核内 UB 切分 | UB 预算反解（tiling_aba L137-185；tiling_bab L151-181；tiling_ab L73-85） | `BLOCK_N: tl.constexpr`（(b,s) 组内 n 行 tile 宽） | `next_pow2(N)`；N>128 时固定 128 + 内层循环 |
| blockNumB/S 等 | 分核 | 见 §4 | 不需要（persistent-core 隐式覆盖） | — |
| isPartialRope | realDim≠D | L319 | wrapper：`out=x.clone()` 保底直通段，kernel 只写 [0,RD) | RD==D 走 empty_like 快路径 |
| layout | 属性 | — | 不进 kernel；wrapper 按 layout 取 (stride_b, stride_s, stride_n)（TND 时 B=1、stride_b=0） | 直接读 tensor.stride() |
| CosB | cos/sin B 维 | — | `stride_cb`：CosB==1 → 0（广播退化为寻址） | wrapper 判定 |

### tilingkey / 模板分支 → constexpr

| tilingkey / 分支 | 分支语义 | Triton 处理 |
|---|---|---|
| 20010(ABA)/20011(BA)/20020(BAB)/20030(AB) | 循环结构 + cos/sin 广播策略（计算数学相同） | **合并为单 kernel**：广播 → stride_cb=0；循环结构差异 → persistent-core 行循环吸收 |
| rotaryMode∈{1,2,3} → Half/Interleave/QuarterVF | 三种配对数学 | `ROTARY_MODE: tl.constexpr` 三分支 |
| dtype 模板 DTYPE_QUERY | fp16/bf16/fp32 实例化 | 单 kernel 指针 dtype 自动分派 + `.to(tl.float32)` 统一计算域 |
| 队列双缓冲（4 队列 ×2） | 流水掩盖 | 不需要 constexpr——**勿用 num_stages/autotune**（实测：num_stages 在 Triton-Ascend 运行时不支持、仅 WARNING 不生效；µs 级 kernel 上 autotune 的 do_bench 噪声会误选配置，P4 三组一致拒绝） | 编译器自动流水 |
| legacy key 1/3/4 | 老平台小 shape/聚集搬出/cast | 丢弃（目标平台 arch35；BF16 由统一 fp32 计算域覆盖） |

## 6. 片上内存预算（详设 §5）

| Ascend C buffer（ABAAndBA，aba_and_ba.h:139-148） | 位置 | 容量表达式 | Triton 等效 tile | 字节数校核（BLOCK_ROWS=8, RD=128 half, fp16） |
|---|---|---|---|---|
| qInQueue_/qOutQueue_ | VECIN/VECOUT ×2 | ubFactorB×ubFactorS×ubFactorN×dAlign×T×2 | [BLOCK_ROWS, BLOCK_DH] 输入×2 半 + 输出 tile | 8×64×2B×2×2 ≈ 4KB |
| cosInQueue_/sinInQueue_ | VECIN ×2 | BA: ubFactorS×dAlign×T；ABA: ×ubFactorB | [BLOCK_DH]×4（c1/c2/s1/s2），全行共享 | 64×2B×4 = 512B |
| fp32 Reg 计算域 | 寄存器 | — | 计算 tile（编译器分配） | 8×64×4B×~6 ≈ 12KB |

- UB 预算合计：每迭代活跃数据量级 **<32KB** vs UB 192KB/核（Ascend950：AIV 40 核、UB=196608B，tiling UT 实参锚点 `tests/ut/op_host/test_apply_rotary_pos_emb_tiling.cpp:1478`）——富余近两个数量级；BLOCK_N=next_pow2(N) 在 N≤128 时单 tile 覆盖 (b,s) 整组，容量无压力。D≤1024 保证单行 rope 区域 ≤4KB（fp32），整行单次载入无压力。
- double buffer：源码 4 队列全双缓冲；Triton 侧**勿用 num_stages/autotune**（实测不支持/噪声误选，§5），行循环的 GM 往返由编译器自动流水掩盖；cos/sin 小且经 (b,s) 组级外提后每组只载一次。

## 7. 流水编排与 Triton 可达性（详设 §6 kernel 方案）

### 流水结构

源码三 kernel 同构：外层（S 或 BS 打平）→ 中层（B / N）→ 内层单块 CopyIn→Compute→CopyOut；cos/sin 在能复用的层级载一次（BA 广播：S 段级复用，aba_and_ba.h:185-196；BAB：S 段级，bab.h:135-154）；Compute 按 rotaryMode 分派 VF，fp32 RegTensor 域。Triton 对应（P4 实测收敛）：persistent-core 以 **(b,s) 组为外循环**，组内 load cos/sin 一次 + n 行 tile load x → fp32 计算 → store——**cos/sin 组级外提是单点 +100% 的决定性杠杆，必须显式写**（不能依赖 L1/L2 自然驻留：驻留只省带宽，不省重复 load 的指令与地址计算开销）。

### 可达性判定表

| Ascend C 手段 | 锚点 | 可达性 | Triton 写法 / 降级方案 | 预期影响 |
|---|---|---|---|---|
| CopyIn→Compute→CopyOut 核内循环 | aba_and_ba.h:236-244 | ✅ | 行循环内 load→compute→store | 无 |
| LoopModeParams 多维跨步 DMA（BNSD B→N→S） | aba_and_ba.h:284-291 | ✅ | strides 参数化寻址 `b*sxb+s*sxs+n*sxn` | 无 |
| DataCopyPad 越界保护 | 各 kernel | ✅ | `tl.load/tl.store(mask=...)` | 无 |
| TQue 双缓冲 ×4 | aba_and_ba.h:139-148 等 | ⚠️ 编译器自动 | **勿用 num_stages/autotune**：num_stages 在 Triton-Ascend 运行时不支持（WARNING 不生效）；µs 级 kernel autotune do_bench 噪声误选（P4 三组实测一致拒绝） | 交给编译器注流 |
| MicroAPI DeInterleave/Interleave（interleave 模式奇偶重排） | common.h:272-274, 681-683 | ✅ 已实测定局 | **连续 load + `tl.split(tl.reshape(x,(…,2)))` / `tl.interleave` 寄存器重排**（P4 采纳路径，interleave 7 case 达 4.5~7.0x）。**禁用 strided-2 访存**（de=2i/do=2i+1 两次半宽 load/store）：实测触发两个后端正确性缺陷——① 向量 div/mod 行解码 × strided-2 → 静默错值；② 并发下 masked strided-2 store → 非确定性相邻行污染（grid=1 不复现，并发才出）；另 `InterleaveStatusWithMaskOptimization` pass 对带 mask 的重叠/移位 load 静默丢 mask → 越界。契约细节见 runbooks/memory-sram/triton-ascend-strided2-interleave-contract.md | 无（连续路径带宽与 half 相当） |
| MicroAPI UpdateMask 逐 VL 尾处理 | common.h:101 等 | ✅ | `mask = i < pair` 统一 | 无 |
| RegTensor\<float\> 全程 fp32 计算 | common.h 全部 VF | ✅ | `.to(tl.float32)` + 末段 `.to(out_dtype)`（RTNE） | 必须保留（精度） |
| partial rope 双段搬运（CopyIn/OutHalf/QuarterMode） | common.h:781-877 | ✅ | wrapper `out=x.clone()` + kernel 只写 [0,RD)；或 kernel 内补写直通段 | clone 多一次全量拷贝（RD≈D 时可忽略；RD≪D 时直通占比大，可选 kernel 内直通） |
| BA 广播 cos/sin 载一次复用 | aba_and_ba.h:185-196 | ✅ **必须显式复刻** | **(b,s) 组级循环不变量外提**：bs 标量外循环 + 组内 cos/sin 只载一次（实测单点 +100%，2.0956→4.2060）；stride_cb=0 处理 B 广播。**勿依赖 L1/L2 驻留自然复用**（只省带宽不省指令，P2 初版判断错误已更正） | 决定性性能杠杆 |
| Q/K 融合单 launch（UB 复用 cos/sin） | apt.cpp 单入口 | ✅ 不需要 | 两次 launch，L2 覆盖复用 | 省跨界拼接逻辑 |
| Q/K 跨界块拼接 | ab.h:142-161 | 丢弃 | 两次 launch 后无跨界 | 无 |
| workspace 16MB / 4095 blockCount 上限 | tiling 各文件 | 丢弃 | Triton 无对应概念 | 无 |

**❌ 不可达项：0 个**。interleave 的 Reg 级重排已由 `tl.split`/`tl.interleave` 寄存器路径实测定局（P4：连续 load + 寄存器重排正确且高效；strided-2 访存路径有后端正确性缺陷，已禁）。

## 8. Triton 骨架（生成用伪码）

完整可运行版（含 wrapper 与自建标杆）见 `output/apply_rotary_pos_emb/triton_skeleton.py`（已同步为下述 P4 收敛结构）。以下为 **P4 实测收敛后的结构**（2026-09-08 三组回放优胜方形态：(b,s) 组外循环 + cos/sin 组级外提 + interleave 寄存器重排；几何均值 4.2060x）：

```python
@triton.jit  # 无 autotune/num_stages：µs 级 kernel 上实测有害/不生效（§5）
def apply_rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr,
                      BS, S, N, RD,                # BS = B*S（TND 时 B=1）
                      sxb, sxs, sxn,               # x 的 (b,s,n) 元素 stride，layout 归一化
                      scb, scs,                    # cos/sin 的 (b,s) stride；CosB==1 → scb=0
                      SEG,                         # half/interleave=RD//2；quarter=RD//4
                      ROTARY_MODE: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_DH: tl.constexpr):
    pid = tl.program_id(0); num_progs = tl.num_programs(0)
    per_prog = tl.cdiv(BS, num_progs)            # grid = min(BS, vector_core_num)，persistent-core
    bs_start, bs_end = pid * per_prog, tl.minimum((pid + 1) * per_prog, BS)
    nn_idx = tl.arange(0, BLOCK_N)               # BLOCK_N = next_pow2(N)，单 tile 覆盖组内全部 n 行
    nm = nn_idx < N                              # mask 保持 int32 比较（fp32 导出 mask 实测触发后端污染）
    i = tl.arange(0, BLOCK_DH)
    for bs in range(bs_start, bs_end):           # (b,s) 标量外循环：解码退化为标量 div/mod，规避后端缺陷面
        b = bs // S
        s = bs - b * S                           # kernel 内禁用 % 运算符，写 a-(a//b)*b（MLIR 常量折叠陷阱）
        x_base = b * sxb + s * sxs + nn_idx * sxn      # [BLOCK_N]
        c_base = b * scb + s * scs                     # 标量
        if ROTARY_MODE == HALF or ROTARY_MODE == QUARTER:
            m = nm[:, None] & (i < SEG)[None, :]
            # cos/sin 循环不变量外提：每 (b,s) 组只载一次（决定性杠杆，实测 +100%）
            c1 = tl.load(cos_ptr + c_base + i,       mask=i < SEG, other=0).to(tl.float32)
            c2 = tl.load(cos_ptr + c_base + SEG + i, mask=i < SEG, other=0).to(tl.float32)
            s1 = tl.load(sin_ptr + c_base + i,       mask=i < SEG, other=0).to(tl.float32)
            s2 = tl.load(sin_ptr + c_base + SEG + i, mask=i < SEG, other=0).to(tl.float32)
            p1 = x_base[:, None] + i[None, :]; p2 = p1 + SEG
            x1 = tl.load(x_ptr + p1, mask=m, other=0).to(tl.float32)
            x2 = tl.load(x_ptr + p2, mask=m, other=0).to(tl.float32)
            tl.store(out_ptr + p1, (x1 * c1[None, :] - x2 * s1[None, :]).to(out_ptr.dtype.element_ty), mask=m)
            tl.store(out_ptr + p2, (x2 * c2[None, :] + x1 * s2[None, :]).to(out_ptr.dtype.element_ty), mask=m)
            if ROTARY_MODE == QUARTER:
                # 第二对段 (i+2SEG, i+3SEG)，同构：o3 = x3*c3 - x4*s3；o4 = x4*c4 + x3*s4
                ...  # 四段 cos/sin 同组外提；完整代码见 triton_skeleton.py
        else:  # INTERLEAVE：全程连续 load/store + 寄存器重排（strided-2 已禁用，§7）
            j = tl.arange(0, 2 * BLOCK_DH)         # 连续覆盖 [0, RD)
            m = nm[:, None] & (j < RD)[None, :]
            p = x_base[:, None] + j[None, :]
            xrow = tl.load(x_ptr + p, mask=m, other=0).to(tl.float32)          # 连续整段
            crow = tl.load(cos_ptr + c_base + j, mask=j < RD, other=0).to(tl.float32)
            srow = tl.load(sin_ptr + c_base + j, mask=j < RD, other=0).to(tl.float32)
            xe, xo = tl.split(tl.reshape(xrow, (BLOCK_N, BLOCK_DH, 2)))        # 寄存器级 deinterleave
            ce, co = tl.split(tl.reshape(crow, (BLOCK_DH, 2)))
            se, so = tl.split(tl.reshape(srow, (BLOCK_DH, 2)))
            oe = xe * ce[None, :] - xo * se[None, :]
            oo = xo * co[None, :] + xe * so[None, :]
            o = tl.interleave(oe, oo)                    # 寄存器重组回连续布局
            tl.store(out_ptr + p, o.to(out_ptr.dtype.element_ty), mask=m)      # 单次连续 store
```

wrapper 要点：① 按 layout 取 (b,s,n) 轴 stride（`_LAYOUT_DIMS = {1:(0,1,2), 2:(1,0,2), 3:(0,2,1), 4:(None,0,1)}`，TND 时 B=1/stride_b=0），直接读 `tensor.stride()`；② cos/sin B 广播 → scb=0；③ `SEG = RD//4 if quarter else RD//2`、`BLOCK_DH = next_pow2(SEG)`、`BLOCK_N = next_pow2(N)`；④ partial rope：`RD<D → out = x.clone()`，否则 `empty_like`；⑤ q/k 各一次 launch（融合单 launch 实测劣化，§4）；⑥ grid = `min(B*S, vector_core_num)`，核数动态读禁硬编码；⑦ 全部 `.contiguous()`（对应 AutoContiguous）；⑧ **形状/stride 全部 constexpr 化**（单次 launch 内不变，编译期折叠 mask 与地址强度削减，实测 +2.2%）；⑨ **勿加 @triton.autotune**（µs 级 kernel do_bench 噪声误选，三组实测一致）。

## 9. 精度标杆与验收（详设 §测试设计）

- 标杆类别：**benchmark 精确 case 已补建**（ascendc-kernelgen-data `npu_benchmark/ascendc-migration/level2/5_ApplyRotaryPosEmb/`，45 case：{3 模式}×{4 layout}×{fp16,bf16,fp32}×GQA/MQA×{CosB∈{1,B}}×{RD=D, RD<D partial} + zero/inf/nan 特殊值；其 Model golden 与本节自建标杆同舍入序列——全 dtype fp32 计算 + 末段单次 RTNE；2026-09-08 修复其 fp16 原生计算缺陷后，45/45 与「fp32+RTNE 完美 kernel 代理」在 evaluator 真实判定逻辑下互验通过，含 partial 直通段 bitwise 校验）。自建 torch golden 保留用于补边界（D=1024 上界、D 非 2 幂 half 等）。
- 标杆实现要点（torch golden，fp32 计算域 + 末段单次 RTNE，与 kernel 舍入序列对齐）：

  ```python
  rot = x[..., :RD].float()
  emb = rot * cos.float() + rotate(rot, rotary_mode) * sin.float()   # 三模式参数化 rotate
  out = torch.cat((emb, x.float()[..., RD:]), -1) if RD < D else emb # partial 直通
  out = out.to(x.dtype)
  ```

- **算子级验收标准**：
  - fp32：rtol=1e-5 / atol=1e-5（一乘一加无累加，应接近 bitwise）；
  - fp16/bf16：对 fp32 标杆 ≤1 ulp（唯一舍入点为末段 RTNE）——**本卡自控线，严于官方**：官方 ST（tests/st/op_api/aclnnApplyRotaryPosEmb/，300 case、V1 锁 half、D∈{64,128}、输入填充 [-1,1]）统一 `fp16_error = 2⁻⁸ = 0.00390625`（high_precision 标准，≈4 ulp@1.0）；验收对线时以官方 2⁻⁸ 为准，1 ulp 用于开发自验；
  - **官方 golden 的坑**：官方 ST executor（executor_aclnnApplyRotaryPosEmb.py `single_golden`）只把 bf16 升 fp32、fp16 原生计算——它只在 [-1,1] 填充 + 2⁻⁸ 容差组合下安全；自定义 benchmark 若换大量级填充（如 N(±100,25²)）或更严 cap，fp16 原生 golden 会把完美 kernel 误判 FAIL。golden 一律全 dtype 升 fp32（kernel 是 RegTensor\<float\> 全域 + 单次 RTNE）；
  - **模式专项**：half 验前后半场交叉配对；interleave 验奇偶位（构造 x[2i]=1, x[2i+1]=0 类探针）；quarter 验四段；
  - **partial rope 专项**：[RD, D) 区间输出必须 bitwise 等于输入；
  - **广播专项**：CosB=1 与 CosB=B 各一遍，结果逐点一致（同一份 cos/sin 内容广播）；
  - **layout 专项**：同一逻辑 tensor 按 4 种 layout 物化，结果逐点一致。
- **填充纪律**：对齐官方 ST 实践，输入填充幅度控制在 [-1,1] 量级；fp16/bf16 大量级填充（|v|>~100）会撞 evaluator max_abs_error cap（fp16=0.1/bf16=1.0），即使相对误差达标也误判。
- **benchmark 语义超集检查**：case 定义域 ⊆ 算子定义域，无超集偏差；覆盖 {3 模式}×{4 layout}×{CosB∈{1,B}}×{RD=D, RD<D}×{3 dtype}（官方 ST 只有 V1 half × D∈{64,128}，interleave/quarter 官方零覆盖，benchmark 是必要补位）；未覆盖项：D=1024 上界、BNSD×CosB=1（tilingkey 20011 路径）、SBND×interleave、BNSD×quarter——由自建标杆补齐。
- 典型用例 shape：LLM 推理典型值——B∈{1,8}、S∈{1, 2048}、QN∈{32}、KN∈{8}（GQA）、D∈{64,128}、RD∈{D, D/2}；examples/ 下 aclnn V1/V2 样例。**测试资产锚点**（P4 前补锚点已完成）：① tiling UT `tests/ut/op_host/test_apply_rotary_pos_emb_tiling.cpp:1459-1607` 含 6 个 Ascend950 用例，期望 key 与本卡分派表完全吻合（BNSD→20010、BSND→20030；小 shape 在 950 也走 AB 20030 而非 legacy key 1，证实优先级链小者先命中），compileInfo 实参锚定 AIV=40 核、UB=196608B；② ST `tests/st/op_api/all_aclnnApplyRotaryPosEmb.json` 300 官方案例（V1 half-only）；③ regbase kernel CPU-sim UT（test_apply_rotary_pos_emb_regbase.cpp）在仓内但**全部注释**，arch35 kernel 无活跃 UT——属已知空白，非提取遗漏。
- 边界/特殊值用例：D=1024（D_LIMIT 上界）、D 非 2 幂（如 96，half，验证 mask 正确性）、rows=1（单行）、rows 非 BLOCK_ROWS 整数倍、QN≠KN。

## 10. Agent actions（生成指引）

1. 识别算子类别：行独立 elementwise（行内 D 维配对耦合），kernel 类型 **AIV-only**；单 kernel + `ROTARY_MODE: tl.constexpr`，**不要**按源码 4 tilingkey 拆 4 个 kernel（差异仅是循环结构与广播寻址，§5 已给出吸收方式）。
2. 切分轴：**(b,s) 组为 persistent-core 外循环**（grid = min(B*S, 物理核数)，连续块划分——相邻 bs 共享 cos 行，L2 友好），组内 n 行展开为 [BLOCK_N, BLOCK_DH] tile（BLOCK_N=next_pow2(N)）；**勿用一块一 program 裸 grid，也勿逐行 (b,s,n) 打平循环**（后者丢失 cos/sin 组级复用，实测差一倍）。
3. layout 处理：wrapper 读 stride 传参，kernel 内零 layout 分支；TND → B=1。
4. BLOCK 静态规则：BLOCK_N = next_pow2(N)（N>128 时 128 + 内层循环）、BLOCK_DH = next_pow2(SEG) 固定（D≤1024 → SEG≤512，§6 校核富余）。**勿挂 @triton.autotune、勿设 num_stages**——µs 级 kernel 上 autotune 的 do_bench 噪声会误选配置（P4 实测：autotune 轮 4.1228 < 静态轮 4.2060；对照组另证实 µs 级 kernel 误选）；num_stages 在 Triton-Ascend 运行时不支持（WARNING 不生效）。
5. 精度提升点：**任何 dtype 都 `.to(tl.float32)` 计算、末段单次 `.to(out_dtype)` RTNE 回写**——多一次中间舍入即可能超 1 ulp 验收线（对齐 common.h RegTensor\<float\> 全域）。
6. 属性/形态两态指引（不给绝对禁令）：rotary_mode、layout 在 GEIR 语境是编译期常量，Triton 侧 ROTARY_MODE 用 constexpr 顺理成章；layout 的 stride 参数化是运行时值（同一 kernel 服务多 layout），无需 constexpr 化。interleave 的访存形态**已定局（非两态）**：连续 load + tl.split/tl.interleave 寄存器重排；strided-2 两次半宽 load/store 因后端正确性缺陷**禁用**（§7）。in-place 两态：源码是原地更新语义（图级证据：op_graph/fusion_pass 的 tensormove pass 专门消除喂给本算子的 TensorMove 拷贝）——Triton wrapper 可选 `out=x` 真原地写（RD<D 时 kernel 只写 [0,RD)，直通段天然不动）；默认仍用 clone/empty_like 出参，避免 benchmark/调试语境的输入变异副作用。
7. 不可达点：无 ❌；勿在 Triton 中复刻 LoopModeParams 多级 DMA 配置、4095 blockCount 分片、workspace 管理——strides/mask/编译器已覆盖。
8. 「勿复刻」清单：① 4 kernel 类分派；② Q/K 跨界块拼接（ab.h:142-161）与 N 合并轴（tiling_ab L73-82）——两次 launch 后不存在；③ legacy key 1/3/4 与 BF16 cast 模板参。**（已移出）**BA/BAB 的 cos/sin「载一次复用」——P2 初版误判为「stride_cb=0 + cache 驻留等价」列入勿复刻，P4 实测显式 (b,s) 组级外提是 +100% 决定性杠杆，**必须复刻其语义**（仅不必复刻其显式队列管理形式）。
9. partial rope：默认 wrapper clone 保底直通段；RD≪D 且性能敏感时改 kernel 内直通写回，注意两写区域不重叠。
10. 验收按 §9 标准执行（自建标杆 + 模式/广播/layout/partial 四专项 + 填充幅度 [-1,1] 纪律）。

## 11. 复现性声明

本卡由 Ascend C 源码静态分析转化，过程产物：`output/apply_rotary_pos_emb/{manifest,extraction,mapping}.md` + `triton_skeleton.py`。**P4 双盲回放已完成（2026-09-08，confidence 升 verified）**：三组对照全 45/45 通过，终点单调——直给卡组 **4.2060x** > KB 检索组 3.5789x > 无知识对照组 3.1749x（详情 `validation_report.md`）。P4 实测核销/更正项：① interleave 访存形态——strided-2 有后端正确性缺陷（静默错值/并发 store 污染），连续 load + tl.split/tl.interleave 寄存器重排正确且高效，§7/§8 已定局；② BLOCK autotune/num_stages——µs 级 kernel 上实测有害/不生效，已改静态规则；③ 奇偶交织 store 的 DMA 合并问题——随 strided-2 禁用而消解；④ bf16 `.to(tl.float32)` 精度路径——45 case 含 bf16 全过，与 golden 互验一致；⑤ cos/sin 复用——P2 初版「cache 驻留等价」判断错误，实测显式组级外提 +100%，§7/§8/§10 已更正。仍未实测：legacy 平台差异节为源码对照，未在 910b 实测。
