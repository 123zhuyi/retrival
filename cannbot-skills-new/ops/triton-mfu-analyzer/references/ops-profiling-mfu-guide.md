# MFU 采集指南（官方 operator_mfu 等价口径）

## MFU 是什么

MFU（Model FLOPs Utilization）= 模型的**理论计算量**在芯片**峰值算力**下的利用率：

```
actual_tflops = FLOPs / (kernelDuration(ns) × 1e-9) / 1e12
mfu           = FLOPs / (kernelDuration(ns) × 1e-9) / chipPeakFLOPS
```

- **分子 FLOPs**：模型完成该算子语义所必需的理论计算量，与实现无关；
- **分母**：kernel 实际耗时 × 芯片 cube 峰值算力；
- **MFU 越接近 100%**，说明实现越接近硬件极限。MFU 不随"实现多算了多少冗余 FLOPs"而虚高——
  冗余计算会同时拉长耗时，MFU 反而下降，因此它是比 PMU 占空比更真实的"有效算力"指标。

## 与 PMU 口径（aic_mac_fp16_ratio）的区别

| | 官方口径 MFU（本 skill） | PMU 口径 mac_fp16_ratio（--compare 采集） |
|---|---|---|
| 分子 | 理论 FLOPs（语义必需） | 实际执行的 MAC 数（含冗余计算、tile padding） |
| 归一 | 全部 cube 核 × 标称峰值 | **活跃核** × **运行时频率**峰值 |
| 小 shape | 正确反映核利用率不足 | 按活跃核归一导致**严重虚高**（实测可虚高 ~12×） |
| 用途 | 跨算子/跨实现横向对比"离极限多远" | cube 阵列占空比、流水瓶颈定位 |

两者换算关系：`官方MFU ≈ PMU-MFU × 理论FLOPs/实际执行FLOPs × PMU峰值/标称峰值`。
做"还能优化多少"的判断用官方口径；做"瓶颈在哪条流水线"的定位用 PMU 口径。

## 理论 FLOPs 统一口径（官方文档第 6 节）

只统计核心 GEMM 类计算，后处理一律不计：

- **GEMM / MatMul**: `2 × M × K × N`
- **Attention 类**: `2 × score_elems × q_dim + 2 × score_elems × value_dim`
  （仅 Q@K^T 与 P@V 两次 GEMM；softmax / mask / scale / transpose / gather **不计**）
- **Conv**: `2 × Cout × Hout × Wout × Cin × KH × KW × batch`（按 im2col 等价 GEMM）
- Elementwise / Norm 等 memory-bound 算子：MFU 意义不大（分子太小），不建议采集，
  这类算子应看 MemoryUtilization（--compare 模式）。

## 使用方式

```bash
python3 $SKILL_DIR/scripts/ops_profiling/msprof_perf_summary.py --quick --mfu \
    --output-dir <op_dir> [--impl asc] [--mfu-top-k 3] [--mfu-peak 353.89]
```

- 前置条件：算子目录下存在 `flops.py`（见下节）；不存在时只采集耗时并提示跳过 MFU。
- 默认行为：按理论 FLOPs 降序**自动选取 top-K 大 shape**（默认 K=3）——MFU 随规模饱和，
  最大几个 shape 即代表峰值水平，且省去小 case 的采集时间。如需全量，显式 `--cases` 或 `--max-cases`。
- MFU 只对 AscendC 实现（asc 侧）计算；`--impl asc` 可省掉参考实现的采集时间。
- 芯片峰值自动推导（platform_config ini / npu-smi），推导失败时用 `--mfu-peak` 显式指定。
- 输出：`perf_report.md` 增加 "MFU（官方 operator_mfu 等价口径）" 小节，
  `performance.json` 增加 `mfu` 字段及 per-case 的 `theory_gflop/actual_tflops/mfu_pct`。

## flops.py 编写约定

在算子输出目录放 `flops.py`，暴露一个函数：

```python
def flops(case: dict) -> float:
    """返回该 case 的理论 FLOPs。case 为 jsonl/json 中的一行，
    含 inputs 列表：tensor 项有 name/shape/dtype，标量项有 name/value。"""
```

示例（MatMul）：

```python
def flops(case) -> float:
    a, b = case["inputs"][0], case["inputs"][1]
    m, k = a["shape"][-2], a["shape"][-1]
    n = b["shape"][-1]
    batch = 1
    for d in a["shape"][:-2]:
        batch *= d
    return float(2 * batch * m * k * n)
```

示例（FlashAttention 类，q/k/v 形状可得时）：

```python
def flops(case) -> float:
    q, v = case["inputs"][0], case["inputs"][2]
    *lead, s, d = q["shape"]           # 末两维为 seq_len × head_dim
    rows = 1
    for x in lead:
        rows *= x                      # batch × heads
    score_elems = rows * s * s         # 稠密注意力；稀疏算子用实际选中的 KV 行数
    return float(2 * score_elems * d + 2 * score_elems * v["shape"][-1])
```

注意：
- 分子必须是**语义必需**的计算量，不要把实现里的冗余（如 split-P 双份 GEMM）算进去；
- 稀疏/变长算子按 case 实际选中元素数计算（如 SFA: 选中 KV 行数 × Q 行数）；
- shape 维度数随 case 泛化时，用 `prod(shape[:-k])` 之类的写法保持通用。

## 参考：经验水位

- 大 shape GEMM/Attention 类优化良好的 kernel：MFU 40%~60%+（910B 实测 CANN SFA 大 shape ≈ 40%）；
- 小 shape（行数少、核铺不满）：MFU 个位数很正常，此时应关注核切分策略而非 cube 流水；
- 官方口径 MFU 与 PMU 占空比的差距揭示冗余计算量（实际执行 FLOPs / 理论 FLOPs）。
