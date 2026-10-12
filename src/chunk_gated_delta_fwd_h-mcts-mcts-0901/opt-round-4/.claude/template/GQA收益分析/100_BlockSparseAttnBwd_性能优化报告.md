# 100_BlockSparseAttnBwd 性能优化报告

## 结论（新口径 `triton_dev_v2`）

| 指标 | 值 |
|---|---|
| verify（`scripts/verify.py`） | **50/50** |
| benchmark 采集 | **50/50** |
| framework 平均时延（几何平均） | 719.14 ms |
| implementation 平均时延（几何平均） | 1.0769 ms |
| **speedup_vs_torch（新口径）** | **667.78x** |
| speedup（旧口径，同一份代码） | 5.3176 |
| 单 case 区间 | 229.33 ~ 1439.5 |

目标 ≥2.0x，实际 **667.78x**。

⚠️ 一次真实的踩坑：第一次跑新口径时 case 44–50 全部 `Vector core execution timed out`，
原因是**我同时在同一张 NPU5 上跑了调试脚本**。独占一张卡重跑后 50/50 全部采到。
⇒ **性能测量必须独占 NPU**，否则失败会伪装成算子问题。

## 一条必须先纠正的历史数据

交接材料记的 **10.13x 不可比**。原因：那次测量里 **12 个 case（29–35, 37, 38, 44, 46, 50）
编译失败**，只有 38 个（且是规模偏小的）case 进了几何平均。
本轮在正确环境（CANN 9.1.0 + triton_ascend 3.2.2 自带的 bishengir 1.2.0）下复测，
那 12 个 `Failed to compile HIVM IR` / `ub overflow` **全部消失**，50 个 case 齐全，
真实基准是 **5.3176x**。后续一切以 5.32x 为参照。

> 这正是"历史结论会作废"的典型：编译失败的 case 被静默排除在几何平均之外，
> 数字会**变好看**而不是变差，很容易误判。看 speedup 之前必须先看
> `passed_cases == total_cases`。

## 环境

- CANN 9.1.0 runtime；triton 3.2.0 / **triton_ascend 3.2.2**
  （编译器是 pip 包自带的 `bishengir-compile 1.2.0`，优先级高于 PATH 上 CANN 的那份）
- Ascend 910B2C，24 AIC / 48 AIV，UB 192KB，NPU 5
- `verify.py` 50/50 通过后才跑 `benchmark.py --keep_res --warmup 5 --repeats 20`，
  阈值与脚本未改，未使用 `--verify_not_required`

## 算子结构

反向的 m / l / O 都是**给定输入**（方案 B：前向中间量预给），所以不需要 online softmax：

```
S    = (q*scale) @ k^T                掩码位置置 -inf
P    = exp(S - softmax_max) / softmax_sum      （全掩码行 nan_to_num -> 0）
delta= rowsum(dout ⊙ attention_in)
dP   = dout @ v^T ;  dS = P * (dP - delta)
dV   = P^T @ dout ;  dQ = dS @ k * scale ;  dK = dS^T @ q * scale
dk/dv 再按 GQA 组求和
```

输入生成器保证全掩码行存的是 max=0 / sum=1，所以只要把被掩码的 S 填 **−3e38**
（不要用 −inf），`exp(−3e38 − m)` 精确为 0，P 自然为 0，不必特判 NaN。

## 瓶颈画像

`benchmark.py --keep_res` 保留的 `kernel_details.csv`，按耗时加权聚合主导 case：

```（主导 case，bench 口径）=====
  compute_dq_kernel                                                   29.6251 ms   43.48%
  compute_dk_kernel                                                   20.9842 ms   30.80%
  compute_dv_kernel                                                   16.6076 ms   24.37%
  compute_delta_kernel                                                 0.6346 ms    0.93%
  aclnnCast_CastAiCore_Cast                                            0.1483 ms    0.22%
  aclnnInplaceZero_ZerosLikeAiCore_ZerosLike                           0.1362 ms    0.20%
```
```（主导 case）=====
  aic_mac_ratio              0.1379
  aic_scalar_ratio           0.5883
  aic_mte1_ratio             0.1170
  aic_mte2_ratio             0.2091
  aic_fixpipe_ratio          0.1388
  aiv_vec_ratio              0.2334
  aiv_scalar_ratio           0.3988
  aiv_mte2_ratio             0.3476
  aiv_mte3_ratio             0.1638
  串行度 (aic+aiv)/Duration     1.7563   <= Cube-Vector 串行
```
```（主导 case，按耗时排序）=====
  compute_dq_kernel                            mac=0.132 sca=0.550 mte2=0.203 vec=0.289 vmte2=0.371 vmte3=0.121 cube%=-
  compute_dk_kernel                            mac=0.161 sca=0.641 mte2=0.238 vec=0.208 vmte2=0.358 vmte3=0.202 cube%=-
  compute_dv_kernel                            mac=0.133 sca=0.648 mte2=0.203 vec=0.164 vmte2=0.294 vmte3=0.181 cube%=-
  compute_delta_kernel                         mac=0.000 sca=0.000 mte2=0.000 vec=0.509 vmte2=0.434 vmte3=0.142 cube%=-
  aclnnInplaceZero_ZerosLikeAiCore_ZerosLike   mac=0.000 sca=0.000 mte2=0.000 vec=0.075 vmte2=0.000 vmte3=0.451 cube%=-
  aclnnCast_CastAiCore_Cast                    mac=0.000 sca=0.000 mte2=0.000 vec=0.094 vmte2=0.570 vmte3=0.243 cube%=-
```

按 kernel Name 均值的分项（三个代表性 case）：

| kernel | case 1 | case 44 | case 48 |
|---|---|---|---|
| compute_dq_kernel | 41.0 us | 2596 us | 1237 us |
| compute_dk_kernel | 42.3 us | 2060 us | 715 us |
| compute_dv_kernel | 30.2 us | 1789 us | 518 us |
| compute_delta_kernel | 4.7 us | 20.9 us | 75.3 us |
| aclnnCast（3 次/call） | 3.9 us | 7.0 us | 9.3 us |
| aclnnInplaceZero（4 次/call） | 2.9 us | 5.8 us | 8.0 us |

**dq/dk/dv 三个 kernel 占约 95%。** 两个明确的结构性浪费：

1. **三个 kernel 各自重算一遍 S 和 P。** S = q@k^T 是三者共用的中间量，
   现在算了三遍。理论上 dq 与 dk 可以融合成一个 kernel（dS 只算一次），
   dv 需要 P 但不需要 dS。
2. **q/k/v/dout 全部 `.to(tl.float32)` 之后再 `tl.dot`，Cube 走 fp32 路径。**
   这是过度对齐：这些量本来就是 bf16/fp16 存储，低精度操作数的乘积在 fp32
   累加器里本来就是精确的，升 fp32 一位有效数字都不增加，却让 UB 流量翻倍、
   并且把 Cube 推到 fp32 通路。
   （对照面：P 和 dS 是**真 fp32 中间量**，那两个才需要 hi+lo 二段拆分。）
3. `_precompute_keep_mask` 在 CPU 上物化 `[B, H, max_q, max_k]` 的稠密 bool 掩码
   再传上 NPU，最大 case 上百 MB，kernel 每个 tile 都要从 GM 读这块掩码。
   掩码粒度固定 128×128，只需要"每个 (b, q_block, h) 选中了哪些列块"的压缩表
   （99 号已验证这条路可行，收益 26x）。

## 已识别但未完成验证的优化方向

| 方向 | 预期 | 状态 |
|---|---|---|
| 稠密 keep_mask → 块级压缩选择表 | 去掉上百 MB 的掩码读；dq/dk/dv 都能跳块 | 未完成 |
| q/k/v/dout 保持原生 dtype 喂 tl.dot | UB 流量减半，Cube 走低精度通路 | **踩到 fp16 下溢**（见下） |
| dq 与 dk 融合成一个 kernel（dS 只算一次） | 少一遍 S/P 重算 + 少一个 kernel name | 未完成 |
| grid 收缩到 AICORE_NUM + 核内 grid-stride | 减少 program 数 | 未完成 |

**fp16 下溢的实测发现**：把 P 做 hi+lo 二段拆分时，fp16 路径下 `P` 的取值可以低到
**4e-21**，远低于 fp16 的最小非规格化数（约 6e-8），`ph = P.to(fp16)` 直接变 0、
`pl = P - ph` 也随之失真。也就是说 **hi+lo 拆分在 fp16 上不是无条件安全的**，
必须先看该量的动态范围；bf16 因为指数位多（最小非规格化数约 9e-41）不受影响。

## 测量口径

`benchmark.py` 的 `parse_operator_latency` 按 kernel `Name` 分组、取最后 repeats 行
再除以 repeats，**同名 kernel 一次 forward 内发射多次只被计一次**，
总时延 = 各 kernel name 的均值之和。因此 **kernel name 的个数直接进分子**
（少一个 kernel 就少一份耗时）。本实现的 4 个 Triton kernel 每次 forward 各发射一次，
另有 aclnnCast ×3、aclnnInplaceZero ×4（buffer 初始化）。
报告的 speedup 一律是该口径下 50 case 的几何平均，不是真实端到端加速比。

## 逐 case 数据（新口径）

| case | framework(ms) | impl(ms) | speedup |
|---|---|---|---|
| 1 | 37.31 | 0.1408 | 264.99 |
| 2 | 41.44 | 0.1352 | 306.53 |
| 3 | 34.33 | 0.1497 | 229.33 |
| 4 | 59.48 | 0.1236 | 481.26 |
| 5 | 82.93 | 0.2268 | 365.67 |
| 6 | 249.92 | 0.3948 | 633.04 |
| 7 | 176.06 | 0.6663 | 264.24 |
| 8 | 246.48 | 0.5171 | 476.66 |
| 9 | 133.69 | 0.3909 | 342.02 |
| 10 | 154.39 | 0.3786 | 407.80 |
| 11 | 188.50 | 0.3527 | 534.45 |
| 12 | 295.81 | 0.5858 | 504.96 |
| 13 | 260.18 | 0.5098 | 510.36 |
| 14 | 268.21 | 0.4372 | 613.47 |
| 15 | 168.75 | 0.3702 | 455.83 |
| 16 | 847.98 | 1.0568 | 802.40 |
| 17 | 792.47 | 0.9857 | 803.97 |
| 18 | 788.88 | 1.3450 | 586.53 |
| 19 | 335.53 | 0.5828 | 575.72 |
| 20 | 537.20 | 0.9506 | 565.11 |
| 21 | 745.86 | 1.2834 | 581.16 |
| 22 | 1303.72 | 0.9513 | 1370.46 |
| 23 | 894.41 | 1.2396 | 721.53 |
| 24 | 640.30 | 1.0190 | 628.36 |
| 25 | 471.23 | 0.7303 | 645.26 |
| 26 | 1152.33 | 1.6449 | 700.54 |
| 27 | 870.35 | 0.9169 | 949.23 |
| 28 | 1094.88 | 0.9995 | 1095.42 |
| 29 | 1089.33 | 1.6069 | 677.91 |
| 30 | 581.85 | 0.9646 | 603.21 |
| 31 | 1091.22 | 0.9395 | 1161.49 |
| 32 | 1650.33 | 1.1977 | 1377.92 |
| 33 | 1160.49 | 1.5204 | 763.28 |
| 34 | 1261.33 | 1.4228 | 886.51 |
| 35 | 1061.77 | 1.0013 | 1060.39 |
| 36 | 4873.77 | 4.2380 | 1150.02 |
| 37 | 2245.21 | 2.4366 | 921.45 |
| 38 | 2774.73 | 4.9556 | 559.92 |
| 39 | 6592.93 | 7.1538 | 921.60 |
| 40 | 3562.72 | 2.4750 | 1439.48 |
| 41 | 2515.13 | 2.9810 | 843.72 |
| 42 | 1506.92 | 2.0792 | 724.76 |
| 43 | 2161.41 | 3.2204 | 671.16 |
| 44 | 3947.42 | 6.5376 | 603.80 |
| 45 | 3206.91 | 3.6302 | 883.40 |
| 46 | 4220.28 | 4.2291 | 997.91 |
| 47 | 3096.38 | 3.4944 | 886.10 |
| 48 | 1658.15 | 2.6775 | 619.29 |
| 49 | 2792.03 | 3.0387 | 918.82 |
| 50 | 8236.02 | 5.7719 | 1426.92 |
