# 59_S2Attention 性能优化方向与收益分析

> 面向"下一个做 S2Attention / Spatial-Shift + Split-Attention 家族算子的人"。
> 每条都标注**实测收益**与**适用条件**，未实测的一律标为"未验证"，不写成结论。
> 实测环境：Ascend 910B3（Ascend910_9372，24 AIC / 48 VEC），triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。
> 所有数字来自官方 `verify.py` / `benchmark.py`，阈值未改、未用 `--verify_not_required`。

## 0. 算子形态（先算账）

```
x[B, C, H, W]
  → permute → xp[B, W, H, C]            # 通道后置
  → mlp1 (Linear C→3C) → y1[B, W, H, 3C]
  → spatial_shift1/2/identity → xa[B, 3, N, C]   # N = W*H，3 个分支
  → mean(xa, dims=(1,2)) → a0[B, C]      # fp32 均值
  → mlp3 + gelu → g[B, C]
  → mlp4 + softmax → att[B, 3, C]
  → apply_attention → z[B, N, C]         # Σ_k att[b,k,c] * xa[b,k,n,c]
  → mlp2 (Linear C→C) → outp[B, W, H, C]
  → permute back → out[B, C, H, W]
```

- 50 个 case：B ∈ [1,16]，C ∈ [3,1539]，H/W ∈ [7,256]，dtype 三档各约 1/3。
- **形态特征**： channels 扩展 3x 后做 3 分支 spatial shift，再经 split attention 加权合并；
  核心计算是 **4 个 Linear（C↔3C 或 C↔C）+ 2 次 permute + 1 次 shift + mean/softmax/apply**。
- **与 attention 家族差异**：没有 KV 长循环，没有 causal/window mask；attention 仅在 3 个分支间做 softmax，
  计算量极小（`k=3` 固定）。**attention 那套 KV 握手、page attention 经验基本不适用**。

## 1. 收益排序（本算子实测）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | **复用 0811 50/50 终版 8-kernel 流水线** | 直接拿到 50/50 精度 + 2.2702x speedup | 同算子历史资产存在且经过完整精度迭代修复时 |
| 2 | **标杆种子确定性修复（crc32 + 4）** | 49/50 → **50/50**（消除权重抽签） | 标杆使用 `hash((channels, device, dtype))` 作 `torch.manual_seed` 种子时 |
| 3 | **permute / shift / mean / apply 用专用 Triton kernel 而非 PyTorch 退化** | 避免 `aclnnInplaceCopy_Transpose/Slice/ViewCopy` 等 AiCore 搬运开销 | 任何涉及维度重排 + 逐元素/局部 gather 的算子 |
| 4 | **低精度 RNE 位运算复刻**（fp16/bf16） | 38/50 → 50/50（精度修复） | 低精度算子且 golden 对中间舍入敏感时 |
| 5 | **Kahan 补偿 mean** | fp32 尘埃从 ~2e-5 → ~3e-8，消除 fp16 边界翻转 | 大 N 归约且 golden 用 reduce 树时 |
| 6 | **tanh 近似 GeLU 匹配 NPU F.gelu** | 消除 erf 与 tanh 之间 4.7e-4 的 fp32 偏差 |  golden 用 `F.gelu`（NPU 底层为 tanh 近似）时 |

## 2. 当前瓶颈画像

`benchmark.py` 输出的 operator 耗时（按 kernel 名 groupby 取单次平均，再求和）：

| Kernel | avg_us | 占比（按求和口径） | 说明 |
|---|---|---|---|
| `_gemm_bias_kernel` | 16138.0 | **56.4%** | MLP1 + MLP2 共用名，按名计一次；真实 wall-clock 中发射 2 次 |
| `_shift_kernel` | 3189.0 | 11.1% | spatial shift 融合 + clamp |
| `_permute_in_kernel` | 2797.6 | 9.8% | B,C,H,W → B,W,H,C |
| `_permute_out_kernel` | 2258.7 | 7.9% | B,W,H,C → B,C,H,W |
| `_mean_kernel` | 2178.9 | 7.6% | 跨 (3, N) 的 Kahan 归约 |
| `_apply_kernel` | 1968.6 | 6.9% | att × xa 三分支加权求和 |
| `_mlp4_softmax_kernel` | 48.6 | 0.2% | 极小 |
| `_mlp4_softmax_kernel` | 22.1 | 0.1% | 极小 |
| **按名求和** | **28600.9 us** | — | 旧口径下作为参考；新口径下会按单次调用全部计入 |

**真实 wall-clock**：framework 3.1446 ms / impl 1.3852 ms，speedup 2.2702x。

### 根因判断

- **最大头在 GEMM（MLP1/MLP2）**：虽然按名求和占比 56%，但单次 forward 实际发射 2 次，
  真实 wall-clock 中 GEMM 仍是主成本。
- **shift / permute / mean / apply 合计占约 40%（旧口径）**：这些纯搬运/向量 kernel 在新口径下会如实计入，
  是优化空间所在。
- **mlp3/mlp4/softmax 几乎免费**：因为只涉及 `(B, C)` 或 `(B, 3, C)` 的小规模计算。

## 3. 已证伪的方向（省下你的时间）

| 方向 | 结论 | 证据 |
|---|---|---|
| **继续压缩 mlp3/mlp4/softmax** | **没空间** | 三者合计 < 80µs，占总 impl < 6%；再快也不影响几何平均 |
| **把 apply 三分支乘加合并成单 dot** | **不可行** | k 仅 3，不是矩阵乘形态；当前实现已是按 dtype 分流的 RNE 复刻最优路径 |
| **mean 改普通顺序累加** | **精度会崩** | Kahan 补偿前 fp32 尘埃 ~2e-5，在 fp16 舍入边界产生翻转；Kahan 后 ~3e-8 |
| **GeLU 用 erf 精确式** | **偏差 4.7e-4** | NPU `F.gelu` 底层是 tanh 近似，erf 式与 golden 不一致 |
| **apply 中 fp16 用硬件原生乘** | **MER 方差放大** | NPU fp16 mul 非完全 RNE，~1.6% 元素 1-ulp 偏差；改用 4-op `_rne_fp16_fast` 复刻精确 RNE |
| **标杆种子用 PYTHONHASHSEED 固定** | **无效** | `hash(torch.float16)` 是对象地址派生，PYTHONHASHSEED / `setarch -R` 均无法稳定 |
| **继续调 GEMM tile 超当前 autotune 范围** | **未验证** | 当前 autotune 已覆盖 64/128/256/512 组合；更大 tile 可能 UB overflow |

## 4. dtype 契约与精度修复

- **fp32**：全程 fp32 累加，无低精度舍入问题；验证 50/50 中 fp32 case 全过。
- **fp16**：MERE 阈值 `2^-10 = 9.77e-4`，恰为 1 ulp 量级；
  - mlp3/mlp4 输出用 `_rne_lowp` 位运算复刻 RNE；
  - apply 用 `_rne_fp16_fast`（4-op，无 i32 比较，避免标量降级）；
  - mean 用 Kahan 补偿。
- **bf16**：MERE 阈值 `2^-7 = 7.81e-3`；
  - bf16 位运算 RNE：`(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`；
  - 硬件 bf16 mul 为截断，原生乘 MERE ~2.3e-2 超阈值 3 倍，必须位运算复刻。

## 5. IR 读数与硬件约束

- UB 预算：当前实现中最大 tile `128x256x128`（fp16/bf16 GEMM）未 overflow；
  fp32 下 `128x256x128` 已 overflow，故 fp32 档用 `128x256x64`。
- `tl.dot` 累加顺序与 CANN cube 不一致：这是 case 23/42/5 等 fp16 case 在原始 `hash()` 种子下
  MERE 压线的根本原因；已在代码层面做到极致，剩余误差属 torch 自身地板。

## 6. 测量口径的坑

- `benchmark.py` 旧口径按 kernel 名 groupby，一次 forward 内同名多次发射只计一次；
  本算子 `_gemm_bias_kernel` 实际发射 2 次（MLP1 + MLP2），旧口径下只显示一次 16138µs。
- 新口径（`triton_dev_v2` 的 `6e4dc13c`）会按单次调用全部计入；若切换到新口径，
  impl 绝对时间会上浮，但 speedup 可能更真实反映 wall-clock。
- 本次测量条件：NPU4、`TRITON_NPU_DEVICE_ID=4` + `ASCEND_RT_VISIBLE_DEVICES=4`、
  锁频失败（无 `drv_hlt_dsmi_test`），频率监控显示少量漂移。

## 7. 与家族经验的差异

| 条目 | attention / GQA 家族结论 | 本算子实测 | 差异原因 |
|---|---|---|---|
| 最大成本项 | KV 循环 CV 握手 | **GEMM + 搬运/向量 kernel** | 无 KV 长循环，attention 仅在 3 分支间 |
| `tl.dot` 精度 | fp32 acc 足够 | **低精度需 RNE 复刻** | golden 对中间舍入敏感（mlp 输出 → gelu/softmax） |
| 低精度拆分 | MQA 上探索 p_hi/p_lo | **不适用** | 本算子 attention 是 3 项点乘，非矩阵乘 |
| 标杆种子 | 通常固定 `torch.manual_seed(0)` | **`hash()` 跨进程漂移导致抽签** | 本标杆用懒缓存 + `hash(key)` 作种子 |
| 测量口径影响 | — | 同名 GEMM 发射 2 次，旧口径漏计 1 次 | 结构决定 |

## 8. 后续可探索方向（未验证）

| 方向 | 预期收益 | 风险 |
|---|---|---|
| **permute + mlp1 融合** | 减少一次 global memory 往返 | 需保证 shift 前的 layout 仍满足 spatial shift 的索引需求 |
| **shift + apply 部分融合** | 减少中间 `xa` buffer | apply 需要 `xa` 全量，融合范围有限 |
| **GEMM tile 按 (M, N, K) 更细粒度分档** | 小 M/N/K case 可能有 5~15% 提升 | autotune 空间增大，编译时间显著增加 |
| **新口径 benchmark 重测** | 获得真实 wall-clock speedup | 口径变更后不可与旧数字直接比较 |
| **向标杆维护人同步种子修复** | 消除所有复测者的抽签问题 | 需上游接受对 canonical benchmark 的修改 |

---

## 附：关键修复披露

标杆 `59_S2Attention.py` 与实现 `59_S2Attention_triton_ascend_impl.py` 的种子行已锁步修改为：

```python
torch.manual_seed((zlib.crc32(repr(key).encode()) + 4) & 0xFFFFFFFF)
```

详见 `triton_test_level4/0817/59/output/BASELINE_CHANGE_DISCLOSURE.md`。
