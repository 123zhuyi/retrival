# 设计草图 → 性能上限与瓶颈推导（通用流程）

适用：任何 Triton-Ascend 算子，推导其**理论性能上限**与**瓶颈归属**。产出两件：
1. `{workdir}/kernel_sketch.json`（agent 草图，过 `pipe_time_model.py` 三道机械闸）
2. 报告 §2.5 的上限表 + 瓶颈归属表

推导材料只允许三类：**算子语义与输出契约、设计草图、芯片理论速率（chip_spec）**。
实测数据（board 占空比、仿真指令数）不进入推导——它们用于第 7 步之后的量化对账。

---

## S1 设计锚定

**有草图（Phase 2 `sketch.txt` 或等价设计描述）→ 读 sketch 逐条提取，缺一不可**：
- **架构决策**（@architecture_decision 每一条）：状态驻留、维度切分、映射选择
- **grid 分解式**：programs 数的显式公式——**搬运复制因子藏在这里**（每 program
  各自需要的全宽 tile 会被 programs 数放大）
- **tile 常量**（BT/BV/BK…）：进 kernel_sketch.json 的 constants
- **数据流**：每迭代的 load/store/dot 序列与依赖方向

⚠ 陷阱（201_GDN 实例）：漏读 "v-dim-program-split" 决策 → k/w 搬运少算 V/BV=2×。
grid 分解式与搬运量必须互相推导核对：`搬运总量 = 每 program 步量 × programs`。

**S1' 无草图 → 按原结构重建 sketch.txt（sketch-optional，2026-10-27 增；2026-10-27 修订为重建制）**：

sketch.txt 缺失（外部 kernel / 草图遗失 / 未走生成管线）时**不再停推，也不绕开 sketch 锚**——
而是**遵循原 designer 流程的结构重建一份 sketch.txt**，之后 S1-S7 原条款零改动执行：

1. **素材三源**（同三方锚，次序倒转——先证据后"意图"）：
   - IR 实现事实：last_pass.mlir（循环结构/调用点/静态形状）；
     `resource_audit.py --emit-sketch-draft` 机械提取 NT/programs/MMA 形状/实测闭合锚
   - 任务语义：任务 json/py 输入输出 + verify 输出集（= S2 契约素材）
   - 芯片速率：chip_spec（同原流程）
2. **结构同构**：重建稿必须含原稿全部字段——`@architecture_decision` 逐条
   （每条附 `ir_refs` 真实行号，措辞用「实现事实=IR 所见」，禁止臆造模板/KKB 出处）、
   `symbols`、`tensors`（形状以任务 json 为权威）、`constexpr`（从 IR 静态形状还原，
   标注来源行）、Kernel 列表（grid 分解式公式）、每迭代数据流伪代码（调用点顺序还原）
3. **落盘**：`{workdir}/sketch.txt`（仅当缺失时允许创建；**存在原稿时禁止覆盖**）；
   文件头第一行 `# @provenance ir_reconstructed <日期>` + 差异声明（无意图层）
4. **下游零改动**：S1 照常读取；kernel_sketch.json `design_refs` 指向重建稿真实行号；
   三道机械闸不变（白名单 / gm_in≥任务下限 / md5 指纹）
5. **口径声明**：报告标注「sketch 为 IR 重建稿」；冗余度 = 重建量/任务下限，
   描述"当前实现搬多少"；设计意图级上限（"本可搬多少"）无原稿即无解，显式声明
6. Worked example：`references/sketch_reconstructed_example.txt`（GDN c1 仅凭 IR 证据重建，
   可与同工作目录原稿 diff 对照保真度）

## S2 语义契约（读任务 json/py + verify 输出集）

- 列出全部输入/输出张量与 dtype
- **逐输出判定"语义必需 / 可选"**：verify 比较哪些输出，哪些是训练下游消费的
  中间态。语义必需的写不可标"冗余"（201_GDN 实例：h 逐 chunk 是训练必需输出，
  把它标成"可省的无效搬运"会高估优化空间）
- chunk/分组粒度若是算子 attr（如 chunk_size=64），标记为不可改

## S3 逐步枚举（每 (迭代单元, program)）

从 S1 数据流枚举每步的字节与 FLOP，**尺寸以 ir_dump/*_ttir.mlir 的实测张量形状为准**
（sketch 是意图，IR 是现实；两者不一致时以 IR 为准并记录）：
- `tt.load`/`tt.store` 逐条列出形状×dtype→字节；区分循环内（逐迭代）与循环外（每链一次）
- `tt.dot` 逐条 2×Πdims MAC→FLOP
- 向量算术按 op 数×宽度粗估（标注"粗估"）
- 对照：循环内 load 字节和 × programs × 链长 应 ≈ 草图 gm_in（对账由 pipe_time_model
  的 `--ir` 机械完成，偏差>25% 强制人工复核）

## S4 语义下限（任何正确实现的刚性组成）

- GM 必读 = Σ输入元素字节（恰读一次，零复制因子）
- GM 必写 = Σ语义必需输出元素字节（恰写一次）
- 固有 FLOP = 算子数学的最小 dot 集
- 串行链 = 状态递推依赖（RAW）决定的每 program 链长：迭代 i+1 消费 i 的驻留状态
  → 链不可切分进 grid

## S5 结构时间（波次公式）

```
墙钟下限 = max( GM必读/聚合MTE2, GM必写/聚合MTE3, 固有FLOP/聚合CUBE,
               ceil(programs/核数) × 每程序链长 × 每迭代临界路径 )
每迭代临界路径(核) = max( 该迭代各步在单核上的时间 )     # 完美流水口径
                    ≥ 单核双 dot 时间                     # RAW 链的硬底
```

⚠ 波次陷阱：programs > 核数时 ceil 放大（链减半被波次抵消的 case 要显式算出，
201_GDN c4 vs c7：128×1波 = 64×2波）。

## S6 速率代入

- **聚合速率**只用于流量下限（S5 前两条）；**每核速率**用于链预算——两层勿混
- 每核速率 = 聚合 ÷ 核数（AIC=24 / AIV=48，从 chip_spec 读，不硬编码）

## S7 判定与归因

产出三张表：
1. **上限表**：逐 case 四条刚性线 + max = 上限 + 绑定者
2. **草图理想 vs 上限**：差距比 + 归因（逐项列出草图多出的搬运/计算）
3. **瓶颈归属**：语义层（结构性天花板）/ 草图层（Triton 可达上限）/ 草图外
   （编排/同步——草图不建模，量化指回 IR 与 board）

### 不可去性证明模板（对每个"冗余"必须做）
构造合并/重排反证：若重排后每核带宽仍先于聚合带宽饱和 → 该冗余在当前抽象内不可去。
（201_GDN 实例：k/w NVB 复制——合并两 v-half 到一个程序，GM 流量减半但每核链
0.85µs/chunk×NT 不变 → 不可去，属数据供给层。）

---

## 产出物 schema（kernel_sketch.json）

```json
{
  "code_md5_prefix": "<最终代码 md5 前 10 位>",
  "constants":  {"BT": 64},
  "aliases":    {"K": "in0_d3", "NT": "nseg*cdiv(in0_d2,BT)", ...},
  "kernels": {"<保留kernel名>": {
    "design_refs": ["sketch.txt L行号 引用"],
    "ir_refs":     ["ttir L行号 引用——必须真实行号，禁止待补占位"],
    "exprs": {"c_flop": "...", "v_flop": "...", "gm_in": "...", "gm_out": "...", "fixp_bytes": "..."},
    "redundancy_notes": ["逐项冗余判断 + 不可去性结论"]}}}
```

机械闸（agent 不可绕过）：表达式白名单 / `gm_in ≥ 语义必读` / md5 指纹 / IR 对账
偏差>25% 强制人工复核。schema 参考实例见 `references/kernel_sketch_example.json`。

---

## Worked example（201_GDN 全链路数字）

- S1: v-dim-program-split → programs = nseg×8×2；BT=64；每 (chunk,program) 读
  k/w 整宽 + u/g 切片
- S2: h 逐 chunk = 必需输出；chunk_size=64 不可改
- S3: 循环内 load 41,216B（w 16384+k 16384+u 8192+g 256）、双 dot 4.19MFLOP、
  store 24,576B；循环外 hi 16,384B
- S4: 必读 3.42MB / 必写 4.14MB / 固有 0.34 GFLOP（c1 基准）
- S5: c1 = 10步×1波；c4 = 128×1；c5-c7 = 64×2波
- S6: 每核 MTE2 0.860µs、CUBE 0.171µs、MTE3 0.513µs
- S7: 上限 c1 3.6 / c4-c7 44µs（MTE3 必写绑定，几何平均 25.5µs）；
  草图理想 c1 8.6 / c4-c7 110.1（几何平均 63.2µs）；恒定 2.48× = k/w NVB 复制，
  合并反证不可去 → 数据供给层。瓶颈归属：语义层 = MTE3 必写 + RAW 链；
  草图层 = 每核 MTE2 关键路径（已达可达上限）；草图外 = 编排/同步。

## 两层实现量口径（2026-10-26 固化）

- **GM 层用 ttir**：`pipe_time_model.ir_traffic()`（--ir 闸门）——形状清晰、
  循环内外可判（scf.for 分割），输出量对账与下限冗余度
- **片上层用 last_pass**：`scripts/last_pass_inventory.py`——intrinsic call 站点
  清单是编译器结构的唯一证据源：set_l1（UB→L1 预载）/fixpipe（L0C→UB 排流）/
  nd2nz（转置读通道）/vcast（数值重铸）均 ttir 不可见（201_GDN 实测：片上搬移
  = GM 口径 3.0×；MTE1 2 站点→14 条指令 = 分形馈送子 tile 化）
- 速率归因（pipe 达成率 gap）必须以 last_pass 清单为输入；速率实测 = trace busy
  （pipe_overlap_analysis）÷ IR 量

## 诚实边界

本流程推得的是**理想带**（完美流水假设）。真实执行落在带之上，超出部分 = 草图
不建模的标量编排/同步/气泡——其量化不属于本流程，指回 SKILL §2（board 占空比）
与 §3A.1（IR 标量降级）。
