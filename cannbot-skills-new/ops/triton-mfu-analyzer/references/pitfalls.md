# 坑与解读

下面每一条都是在 `20_FlashAttentionV2`（Triton-Ascend，910B2/B2C）上实测踩到的，
不是推测。

---

## 坑 1：峰值不区分 dtype，fp32 的 MFU 被低估约 4 倍

ops-profiling 的 `_resolve_peak_tflops` 从 cube 规格推峰值：

```
peak = 2 × cube_m × cube_k × cube_n × ai_core_cnt × cube_freq / 1e6
910B2C: 2 × 16³ × 24 × 1800 / 1e6 = 353.89 TFLOPS
```

式子里**没有 dtype**，推出来的是 fp16/bf16 峰值。本机 `peak_calib.py` 实测
（4096³ GEMM，两次测量）：

| dtype | 实测上限 | 占标称 353.89 |
|---|---|---|
| float16 | 314~320 TFLOPS | ~89% |
| bfloat16 | 317~320 TFLOPS | ~90% |
| **float32** | **85 TFLOPS** | **24%**（约 1/4 速率） |

后果：同一个算子的 fp32 case 用 353.89 当分母，报出来 10.18%，按 fp32 实测上限
修正后是 **42.5%**——从"很差"变成"全场最好"。

**做法**：`--mfu-peak` 显式指定，或用 `mfu_decompose.py --peak-json` 按 dtype 修正。
标称峰值只在同 dtype 内横向对比时才可以直接用。

注意 `peak_calib.py` 测的是**实测可达上限**（fp16 约为标称的 90%），
和标称峰值是两个口径，报数时要说清用的哪个。它还有 ~2% 的测量波动，别当常数抄。

---

## 坑 2：自动 top-K 选出的三个 case 会混 dtype

`--mfu` 未指定 `--cases`/`--max-cases` 时，`_select_mfu_top_cases` 按 `flops.py`
的返回值降序取 top-K（默认 3）。

**这个启发式本身是有效的，只测三个就够。** 50 case 全量实测验证过：自动选出的
`[36, 46, 48]` 在两种口径下**都命中了全局最大值**（标称口径 max = case 46，
修正口径 max = case 48），而尾部小 shape 全部 ≤0.2%。所以默认流程只跑 top-3，
不必为了"保险"去跑全量。

要注意的只有一点：它**只比 FLOPs 数值，不看 dtype**。本例 top-3 里混进两个 fp32，
默认输出的 9%/10% 因此失真。**这个由第 4/5 步的 dtype 修正解决**
（`MFU@ach` 列），不需要为此改选 case。

另一个次要影响：排序键是你写的 `flops.py`。本例 94.7% 的 FLOPs 在投影上，
所以实际是按 `D` 排序而非按 `S`，最长序列（S=641）的 case 38 排第 4 被截掉——
它恰恰是 attention 占比最高的那个。**只有当你要的是"某一段的代表性 shape"
而非"整体峰值 MFU"时**，才需要自己 `--cases` 挑。

---

## 坑 3：`__file__` 派生的用例路径（静默失效）

参考实现常这样找用例：

```python
json_path = os.path.splitext(__file__)[0] + ".json"
```

改名成 `model.py` 之后它会去找 `model.json`；被测实现里还可能**硬编码**原始文件名。
找不到时通常不抛异常，而是走兜底路径（返回空 cache / 用随机合成输入），
**采集能跑完，数据是错的**。

`prepare_op_dir.py` 会自动补齐：复制成 `model.json`、原始文件名，再扫描两个实现
源码里所有 `*.json` 字面量各复制一份。代价是目录里有多份同名用例，**改用例时要全部同步**。

---

## 坑 4：解析用的 warmup/repeats 必须和采集时一致

`mfu_decompose.py` 按 `warmup + repeats` 把 kernel 行等分成若干次迭代、丢掉
warmup 份。数字对不上就切不出整数份。

默认它从 `performance.json` 读这两个值（连同每个 case 的 PROF 目录绝对路径），
所以正常流程不会踩到。只有手工用 `--warmup/--repeats` 覆盖时才需要自己对齐——
对不上会直接报错而不是给出错误数字：

```
(skip: 50 kernel 行不能被 warmup+repeats=6 整除，无法可靠切分迭代)
```

---

## 坑 5：quick 模式对同名 kernel 的多次发射是**求和**

`_parse_msprof_duration_quick` 把一次迭代内所有 kernel 的 `task_time` 相加。
本例每次迭代发射 5 个 kernel（4 次 `linear_kernel` + 1 次 `flash_attention_kernel`），
4 次 linear 全部计入。这对 MFU 是**正确**的口径（分母要覆盖算子的全部执行时间）。

注意别和某些 benchmark 工具"按 kernel 名分组、同名只计一次"的口径混起来——
那种口径下这个算子的时间会少算 3 次投影。

---

## 坑 6：CANN 环境

编译器/profiler 相关结论对 CANN 版本敏感。采集前确认：

```bash
echo $ASCEND_HOME_PATH        # 应指向你以为的那个 CANN
which msprof
```

非交互 shell 可能读不到 `set_env.sh`，会静默跑在旧 CANN 上，整片结论作废。

---

## 坑 7：放大 shape 时用例文件名不统一（静默失效）

各算子 `model.py` 读用例的方式不一样，有的 `os.path.splitext(__file__)[0]+".json"`
（即 `model.json`），有的硬编码 `"<OpName>.json"`。只写其中一个，后一类算子会
**继续跑原始小 shape 且完全不报错**。

症状：**耗时不随数据增长**。实测 43_CoordAtt 数据 ×6561、耗时 ×1.06。

对策：把 case 写进目录下所有候选用例文件（`max_shape_probe.py: case_files()`）。

## 坑 8：只验被测实现，不验参考实现

`ModelNew` 能跑通不代表 shape 合法。等比缩放常破坏参考实现里的形状约束
（卷积权重尺寸、固定 reshape），实测 94 个算子里 26 个出现
"参考实现拒绝、被测实现照跑"，同时 FLOPs 也算不出来。

对策：把"参考实现 FakeTensor 追踪成功且 FLOPs>0"作为硬闸。纯 CPU，很便宜。

## 坑 9：混合实现只统计 triton kernel → MFU 虚高

很多 `ModelNew` 是 triton kernel + torch/aclnn 的混合实现。按 `aclnn` 前缀过滤框架算子，
会导致分子算整个算子、分母只算 triton 部分。实测 26_ViTAttention 因此得到 **102.9%**，
超过 90.4% 的物理上限。

对策：按 `Task Start Time(us)` 排序后做周期检测取最后 N 个周期，
setup 的输入生成算子自然排除，混合实现的 torch 算子被正确计入。

**MFU > 90.4% 一律视为口径错误**，不要当成好结果。

## 坑 10：采集残缺时静默出数

算子探测阶段跑通、加 PMU 后仍可能崩（aicore 异常 507015）。若只采到 1 次前向
却仍按 N 次迭代平均，MFU 会大 N 倍。

对策：runner 跑完打印哨兵行，汇总时校验；缺失则不出 MFU 并标注"采集残缺"。

## 坑 11：PMU 流水占比 > 1

占比本应 ≤1。实测 1233 行中有 1 个算子的 5 个 `aic_*` 项同时 >1
（`aic_mte2` 加权到 3.170），是 MIX kernel 的 `aicore_time` 被低估、计数不自洽。

对策：任一原始比值 >1.05 时标记"PMU 计数不自洽"，**不做 bound 判定**。


## 坑 12：分子是"参考实现怎么写的"，不是"算子语义要算多少"

`FlopCounterMode` 对参考实现做 aten 级计数，得到的是**参考实现的写法**产生的 FLOPs。
当参考实现用**稠密表达来实现稀疏/滑窗/topk 语义**时（先做完整矩阵乘再套 mask），
分子会被系统性放大，而被测 kernel 利用了稀疏结构、实际只做很小一部分工作。

实测 `28_FlashAttentionFwd` 的一个 case（`window_left=23, window_right=3`）：

```
tracer（稠密矩阵乘 + mask）   8,624,885,760
算子语义（仅窗口内）            354,447,360     差 24.3 倍
```

放大 seq 后偏差被放大到 **389 倍**，直接产出 MFU 5771% 与 11735% 这种荒谬值。

**检测手段（零成本，纯 PMU）**：cube 在这段时间里能产出的 FLOPs 有硬上限

```
cube 上限 = Σ(该 kernel 耗时 × 该 kernel 的 aic_mac_ratio) × 芯片峰值
分子 > cube 上限  ⇒  分子不等于 kernel 实际做的工作，MFU 虚高
```

`mfu_grid_search.py` 已内置此检查，超限时在结果里写 `suspect` 并在输出中告警。

**更强的判据：直接读硬件计数的 FLOPs。** 换 `--aic-metrics=ArithmeticUtilization`
指标组，`op_summary` 会给出 `aic_cube_fops` 与 `aiv_vector_fops` ——
**kernel 真正执行了多少 FLOPs，由硬件计数器直接给出**：

```
硬件实测 FLOPs = aic_cube_fops + aiv_vector_fops
分子 / 硬件实测 > 1.2  ⇒  分子不等于 kernel 实际做的工作
按硬件计数的 MFU = 原 MFU / 该比值
```

这条**不依赖任何峰值假设，对 cube 与纯 vector 算子同样成立**，
补上了 cube 占比判据在纯 vector 实现上失效的空白（`aic_mac_ratio` 恒为 0）。

实测某 ParallelPolarizedSelfAttention 实现：tracer 声称 **9.08e10**，
硬件实测 **3.56e8**（cube 0 + vector 3.56e8），差 **255 倍** ——
报出的 42.3% 按实际工作量应为 **0.17%**。

代价是要为选中的点多跑一次 msprof（换指标组），
所以只在最终选出的极限点上做一次，不在整个网格上做。

## 分子该取哪个：语义分子 = min(参考实现追踪值, 硬件实测值)

MFU 的 **M 是 Model FLOPs —— 算子语义上必需的量**。手上有两个可测的分子，
**都不等于语义值，而且偏差方向相反**，所以要取两者的较小值：

| 分子 | 计的是什么 | 偏差方向 |
|---|---|---|
| 参考实现追踪（`FlopCounterMode`） | 参考实现的 aten 算子加起来多少 | 参考实现用稠密表达稀疏/滑窗语义时**偏大** |
| 硬件实测（`aic_cube_fops+aiv_vector_fops`） | kernel 实际执行了多少 | 含 tiling 补齐、被 mask 丢弃的无效算力时**偏大** |

```
参考实现 > 硬件  ⇒  参考实现写稠密、kernel 走稀疏，kernel 没做那么多  ⇒  取硬件值
硬件 > 参考实现  ⇒  kernel 在做补齐/丢弃的无用功，那不是 Model FLOPs  ⇒  取参考实现值
```

某批 93 个算子实测：39 个取硬件值、54 个取参考实现值，两个方向都大量出现。

**两个方向的实例：**

```
28_FlashAttentionFwd   参考 8.62e9  硬件 3.03e9   参考/硬件 2.85x
  → window_left=23/right=3 的滑窗被参考实现写成"稠密矩阵乘 + mask"
  → 取硬件值，MFU 19.62% 修正为 6.80%

37_BigBirdAttention    参考小       硬件 30x      硬件/参考 30x
  → 块稀疏语义，但 kernel 按稠密块算完再 mask
  → 取参考实现值，MFU 保持 1.98%（硬件口径的 59.92% 是"cube 占用率"不是 MFU，
     其中约 97% 的矩阵乘结果被丢弃）
```

**不要只用 `aic_cube_fops` 当分子。** 它测的是 cube 单元吐出的所有 FLOPs，
是**算力利用率**而非 MFU。两者的比值本身是有价值的优化线索：
比值大说明参考实现表达方式浪费，反过来大说明 kernel 在做无用功。

**另注**：`aic_cube_fops` 与 `aiv_vector_fops` 应分开看。MFU 的分母是 cube 峰值
（`2×16³×24×1800MHz`），把 vector 的 FLOPs 塞进分子是量纲错误——
纯 vector 实现（`aic_mac_ratio ≡ 0`）用 cube 峰值算出的任何非零 MFU 都没有物理意义。
实测 `57_ParallelPolarizedSelfAttention` 报 42.29%，而它 cube 一次都没启动。

## 坑 13：CANN 的 [Warning] 不带换行，会和程序输出粘成一行

```
[Warning]: tiling struct [TileInfo] is conflict with one in file hccl_formulaic_t[WALL] 0.101376
```

用 `line.startswith("[WALL]")` 之类的行首匹配会**永远匹配不上**，
导致每个采集点都被误判成"没有返回"。必须用 `re.search` 全文搜索 stdout+stderr。

同一类错误在解析 msprof / runner 输出时会反复出现，凡是从子进程捞标记行的地方都要注意。

## 坑 16：FLOPs 追踪的真实张量回退路径会爆内存

`trace_flops` 类工具通常是「先 FakeTensor，失败则回退真实张量」。回退路径上
参考实现会**实例化稠密中间张量**（`scores = q @ k^T` 的 `[B,H,Sq,Sk]`），
其规模可以远超输入。

**按输入规模设的护栏挡不住**：`Sq=Sk=8192`、40 heads 时输入仅约 100 MB，
score 矩阵却是 `40×8192×8192 = 2.7e9` 元素 ×4B = **10.7 GB**。
护栏检查的是输入，爆的是中间量。

**对策：直接砍掉回退路径 —— FakeTensor 失败就跳过，不在 CPU 分配真实张量。**
真实张量算出的 FLOPs 与 fake 完全一致，回退没有收益，只有风险。
子进程另设 `MFU_FLOPS_NO_REAL=1` 与 `RLIMIT_AS` 硬上限兜底。
文档里不要写「纯 CPU，很便宜」——那只在 FakeTensor 成功时成立。

## 坑 15：msprof 采集不能高并发

采集阶段的并发度必须**显著低于设备数**，否则两种后果二选一：

- **数据失真**：同一 shape 同一 kernel，16 路并发下 Task Duration 666 ms、核活跃 52.6%；
  4 路并发下 339.5 ms、核活跃 99.2%，而 `aiv_time` 两边都是 336.6 ms 纹丝不动
  —— 多出来的时间全是采集开销，会被误读成"核外空转"。
- **直接采不到**：12 路队列 × 每算子 3 个精测点时，10 个算子的精测全部返回
  "msprof 无有效 kernel 数据"；降到 3 路并发后 9/10 立即恢复。

经验值：**并发数 ≤ 设备数 / 4**。初筛（不开 msprof）可以高并发，精测必须压低。

## 坑 14：跳过候选点时必须记录原因

初筛/精测里每个 `continue` 都要把原因写进结果，否则全部失败时只能得到一句
"网格内无有效点"，无从判断断在哪一环——坑 13 之所以难定位就是因为这个。
全军覆没时还应额外打印原因分布（如 `runner 未返回 [WALL]×50`）。

## 解读：怎么判断一个 kernel 是不是 cube-bound

**手法：比同一 kernel 在 fp32 与 fp16 下的绝对吞吐（不除峰值）。**

cube 在 fp32 下每周期算力只有 fp16 的约 1/4。所以：

- 真的 cube-bound → fp32 吞吐应该是 fp16 的 ~1/4
- fp32 和 fp16 **跑出同样的 TFLOPS** → cube 大部分时间在空等，瓶颈在别处
  （vector 计算、MTE2 搬运、CV 跨核同步）

本例实测（attention kernel 绝对吞吐，`mfu_decompose.json` 的 `parts.attn`）：

```
非 causal:  15.2(bf16) 14.0(fp16) 13.5(fp32) 12.0(fp16) 10.4(bf16) 9.5(fp16)  TFLOPS
causal:      6.9(fp32)  6.6(fp32)  5.9(bf16)  5.7(fp16)  5.1(fp16)            TFLOPS
```

按 causal 分层，**完全不按 dtype 分层**，fp32 甚至排在 causal 组最前 → attention
kernel 根本不受 cube 限制。同一批 case 的投影 kernel 则分层干净（fp32 29~61，
fp16/bf16 59~163 TFLOPS），是正常的 cube-bound 行为。

结论：优化 attention 侧的 cube 利用（更大 tile、更好的 dot 排布）不会有收益，
该去查 vector / 同步。

### 另一个信号：causal 让吞吐减半

理论 FLOPs 已按下三角缩减过，同等效率下 causal 应该跑出**相同**的 TFLOPS。
实测 causal 组只有非 causal 组的一半，说明 causal 路径有额外浪费。本例可归因到：

- **块粒度 mask**：`kv_end = min(sq_start + BLOCK_SQ, S)` 只跳过整块，对角块内的
  上三角照算。case 46（S=317, BLOCK=128）实算 6×128×128 = 98,304 个 (q,k) 对，
  语义只需 317×318/2 = 50,403，浪费 1.95×；非 causal 的 case 49 只浪费 1.49×（纯 pad）
- **两趟 QK dot**（online softmax 求 m/l 一趟、求 w 一趟）—— 两种路径都有

---

## 经验水位

- 大 shape GEMM / Attention 类优化良好的 kernel：MFU 40%~60%+
- 小 shape（行数少、核铺不满）：个位数很正常，此时该看核切分策略而非 cube 流水
- 本例全 50 case 标称口径 mean 5.02% / median 3.74% / max 16.57%，
  尾部 10 个小 shape（S≤23）全部 ≤0.2%——**MFU 随规模饱和**这条是成立的，
  所以只测 top-K 大 shape 的策略是合理的
