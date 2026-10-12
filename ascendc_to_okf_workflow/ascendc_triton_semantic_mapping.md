# Ascend C → Triton 语义映射速查表

> P2 阶段的判定依据。每条映射给出：Ascend C 形态 → Triton 对应 → 可达性 → 备注。
> 可达性三态：✅ 直接可达 / ⚠️ 编译器或运行时自动完成（只写约束）/ ❌ 不可达（必须写降级方案）。
> v1.1（2026-09-04 add 经验）：新增广播折算、壳算子分派、多 kernel 分解、host 均分等价四组条目。
> v1.2（2026-09-07 fused_bias_leaky_relu P4 实测回流）：新增 persistent-core 分核（含「一块一 program」次优警告）、属性 constexpr 两态两组条目。

## 1. 分核与任务划分

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| `usedCoreNum` + `GetBlockIdx()` 核内 for 循环 | `pid = tl.program_id(0)`; `for i in range(pid, NUM_BLOCKS, tl.num_programs(0))` | ✅ | Triton-Ascend 固定范式（grouped_matmul、Lighting Indexer 均用）；grid 直接取物理核数 |
| **每核连续段 blockFactor + 核内 ubFactor 循环**（elementwise 家族标准两级切分） | **persistent-core + 核内连续段循环**：`grid = min(cdiv(numel,BLOCK), vector_core_num)`；kernel 内 `per_prog=cdiv(total_blocks,num_progs)`、`for blk in range(pid*per_prog, min((pid+1)*per_prog, total_blocks))` | ✅ | 2026-09-07 P4 实测：这才是两级切分的**忠实映射**。⚠️「一块一 program、grid=NUM_BLOCKS」是次优——超额 block 串行排队（grid=123 vs 40 核实测仅 0.69x）；连续段划分优于交织 grid-stride（生成器 checklist 禁交织，DMA 连续性更好）。对照组+两实验组最终版全部收敛于此（+35%） |
| normal core / tail core 二分 | 同一份 kernel 内用 mask 处理尾块；或 constexpr 分支 | ✅ | 优先 mask 统一写法；尾核逻辑复杂时才拆 constexpr |
| host 侧 tiling 标量计算 | wrapper（python）中预计算后传参 | ✅ | tilingdata 标量一律上移到 wrapper，不进 kernel |
| 多维 blockIdx (AIC+核间) | 单维 grid + 手工解码 `block_idx // / %` | ✅ | Triton-Ascend 只有一维有效 grid，多维必须手工解码 |
| tilingkey 模板分支（TEMPLATE_MODE） | `tl.constexpr` 开关；差异大则独立 kernel + wrapper 分发 | ✅ | 分支选择逻辑放 wrapper（对应 host 侧 tilingkey 计算） |
| host 侧 `blockFormer = ceil(total/coreNum)` 均分 + 尾核 `blockTail`（base 库两级切分） | grid-stride 循环 + 尾块 mask：与静态均分**负载等价**，mapping 须写明等价论证（add mapping.md §1 范例） | ✅ | 卡片据此标「官方机制已对照」，不再写「惯例推断」 |
| base 库回调类策略（preferMultiCore：核占不满时按 CACHE_LINE 缩 BLOCK 铺核，broadcast_tiling.cpp:569-586） | autotune 的 BLOCK 候选空间覆盖该语义 | ⚠️ | 卡片 §10 记「BLOCK 初选偏大须优先降 BLOCK」；手工定 BLOCK 时遵守 |
| 连续输入合轴（DimensionCollapse ≤8 维）/ 非连续合轴计划（KEEP-MERGE-ABSORB） | wrapper 侧 `expand` + stride 折算天然覆盖合轴语义，Triton 无需显式合轴 | ✅ | 等形 case 合轴后即直通路径 |
| kernel 入口 N 路 dtype 分派壳（`add.cpp` 4 路 constexpr → 统一 DAG） | **单 kernel**：dtype 由指针类型推导天然分派，`.to(tl.float32)` 统一精度路径 | ✅ | 13 个 binary 实例 → 1 个 Triton kernel，Triton 侧大幅简化 |
| aclnn 层多 kernel 策略分解（alpha 四策略：Add / Axpy 融合 / AxpyV2 / Mul+Add 两 kernel + Contiguous/Cast/ViewCopy 中转） | **一律单 kernel 融合**：参数融为 runtime 标量（如 alpha），直写 out | ❌→不需要 | 标「结构性更优，少一次 GM 往返」；生成 Agent 不应复刻兜底分解 |

## 2. 数据搬运（MTE）

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| `DataCopy` / `DataCopyPad`（GM→UB） | `tl.load(ptr + offsets, mask=..., other=...)` | ✅ | DataCopyPad 的补零语义 = `other=0.0`；非对齐由 mask 处理 |
| `DataCopy`（UB→GM） | `tl.store(ptr + offsets, val, mask=...)` | ✅ | |
| 广播 DMA 展开（CopyInBrc，按广播形态把输入搬进 UB） | wrapper：`out_shape = broadcast_shapes(x,y)`；`xe = x.expand(out_shape)` 取 stride（广播维=0）；kernel 内输出线性 offs 逐维 div/mod 分解坐标 → `in_off = Σ coord_d·stride_d` | ✅ | 广播维 stride=0 自动复用；等形 case 走直通路径（SAME_SHAPE constexpr，零整数除法，大 2D 性能关键）；最内维 stride=0 的重复 load 是带宽浪费风险点，记卡片 §10 优化项 |
| 离散/跨步搬运（stride≠1 的 DataCopyExtParams） | 多维 offsets 算术表达 stride | ✅ | 用 `offs_i[:,None]*stride_i + offs_j[None,:]` 展开 |
| `Reg::DataCopyGather` / `DataCopyScatter` | 下标向量 `tl.load(ptr + idx_vec)` / `tl.store(ptr + idx_vec, v)` | ⚠️ | 编译器生成离散访存，**无 gather/scatter 指令直达**；性能劣化需在卡片预警 |
| GM→L1→UB 分级搬运 | 无对应（Triton 单层抽象） | ⚠️ | 编译器自动安排；卡片只记录 tile 大小决策 |
| double buffer（TQue buffer_num=2） | `multibuffer=True`（编译期）/ pipeline num_stages | ⚠️ | 不显式编码，由编译器注流；卡片写「需要双缓冲掩盖」约束即可 |
| 显式 pingpong（手写 ping/pong 切换，如 FAV2 的 `p_l1_ping/pong`） | 无对应 | ❌ | 降级：依赖编译器 multibuffer；或手工拆两段循环。预期损失需注明 |

## 3. 计算（CUBE / VEC）

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| `Mmad` / Cube matmul | `tl.dot(a, b)` | ✅ | 转置用 `tl.trans(k)`；累加清零对应每次循环新建 acc |
| `AscendC::Add/Mul/Sub/Max/Min` 等 Vector 指令 | 对应 `+ - * / tl.maximum/tl.minimum` | ✅ | |
| `AscendC::Exp/Log/Sqrt/Rsqrt` | `tl.exp/tl.log/tl.sqrt/tl.rsqrt`（或 `tl.math.*`） | ✅ | |
| `ReduceSum/ReduceMax`（axis 规约） | `tl.sum/tl.max(x, axis=k)` | ✅ | |
| `Cast`（bf16↔fp32） | `.to(tl.float32)` / `.to(tl.bfloat16)` | ✅ | **精度提升点必须保留**：bf16 输入 fp32 累加是双标杆达标前提；DAG 中 `Cast<T, RINT>` 回写 = `.to(out_dtype)`（RTNE 为 Triton cast 默认舍入） |
| `Muls`（标量乘向量） | 标量广播 `x * scalar` | ✅ | |
| `Compare+Select` / mask 计算 | `tl.where(cond, a, b)` | ✅ | |
| Reg 级寄存器直写（`Reg::DataCopy` E2B、寄存器 reshape） | 无对应 | ❌ | Triton 无寄存器级控制；仅作语义翻译（等价 load/store/reshape），性能差异写入复现性声明 |
| NZ/ND 格式转换、`al.insert_slice/extract_slice`（UB 内切片拼 NZ） | `tl.reshape` + 常规 store | ⚠️ | Triton-Ascend 的 `al.*` 扩展部分可用；标准 Triton 用 reshape 近似，nz 拼接逻辑通常可省略 |
| online softmax 等融合流水（FA 类） | 同结构循环 + `tl.dot` 内联 | ✅ | 参考 FAV2 详设：mask/scale 在 softmax 内完成 |

## 4. 同步与流水编排

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| `TPipe` + `TQue EnQue/DeQue` 事件同步 | 无对应 | ⚠️ | 编译器自动插同步；卡片不翻译，只保留「掩盖关系」描述 |
| MTE2/CUBE/VEC 流水掩盖设计 | 计算顺序 + multibuffer/num_stages 提示 | ⚠️ | 卡片记录「哪个循环值得掩盖」作为 autotune 搜索方向 |
| `SetFlag/WaitFlag` 手工同步 | 无 | ❌ | 丢弃，依赖编译器正确性 |

## 5. 标量与控制流

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| tilingdata 字段读（`GET_TILING_DATA_WITH_STRUCT`） | kernel 形参 | ✅ | |
| **float 属性经 tilingdata 透传**（如 negative_slope/scale，host `attrs->GetFloat` → tiling 字段 → kernel 标量） | **两态**：生产语义 → runtime 标量形参；benchmark/属性编译期稳定场景 → `tl.constexpr` | ✅ | 2026-09-07 P4 实测 constexpr 单项 **+17.3%**（4.4187→5.1818）：常量折叠消指令（scale=1.0 消 Muls、ns 已知化简 where）。注意 GEIR 语境属性本就是编译期常量（属性变→重走 tiling+编译），constexpr **更贴近**源码执行模型；runtime 标量仅服务"属性频繁变且编译时间敏感"的动态场景。卡片须给两态权衡，**禁止只写「勿烘焙常量」这类绝对禁令**（实验 2 优化器据此推理拒绝、未实测，丢掉 17%） |
| `if constexpr` 模板分支 | `if <CONST>:（tl.constexpr 条件）` | ✅ | python 层 if，编译期裁剪 |
| int64 地址/索引算术 | 注意：尽量转 fp32/int32 比较，避免标量降级 | ⚠️ | 对应 runbook 卡 int-compare-cast-float32 |
| while/动态循环（数据相关） | 有限支持，优先改静态循环 | ⚠️ | 动态 Loop 是已知性能坑，卡片应给出静态化改写 |

## 6. 精度与标杆

| Ascend C | Triton | 可达性 | 备注 |
|---|---|---|---|
| 中间 fp32 累加（`Cast` 提升） | acc 用 `tl.float32`，store 前 `.to(out_dtype)` | ✅ | 必须保留，写入验收标准 |
| 确定性计算约束 | 规约顺序固定化；atomic 场景需声明 | ⚠️ | Triton 不保证跨块规约顺序，确定性需求需在卡片显式标注风险 |
| 双标杆验收（CPU 高精度 + GPU） | torch fp32 标杆 + 算子级判定准则 | ✅ | 卡片写「算子级判定」（如 topk 集合一致），不写裸 rtol |

## 7. 平台特性边界

| Ascend C / 硬件特性 | Triton-Ascend | 可达性 | 备注 |
|---|---|---|---|
| 950 CV 直通数据通路 / RegBase | 无 | ❌ | 属 AscendC 重写期收益，Triton 移植阶段不承诺 |
| AIC/AIV 异构分工（CV 融合） | 由 `tl.dot` 自动走 Cube，其余走 Vector | ⚠️ | 不显式控制；卡片只需判定「是否含 tl.dot」 |
| UB 192KB 上限、bank 冲突 | UB 上限需人工校核；bank 冲突无对应 | 上限 ✅ / 冲突 ❌ | 卡片 §片上内存预算必须做字节数校核 |
| FP8（A2A3 不支持） | 同不支持，改 bf16/fp16 | ✅ | 参考 Lighting Indexer：量化逻辑整体删除并注明 |
