# 100_BlockSparseAttnBwd 优化日志

环境统一：CANN 9.1.0 / bishengir-compile 1.2.0 / triton 3.2.0 / triton_ascend 3.2.2
（`source $BASE/_infra/env.sh`；非交互 shell 默认会落到 8.5.1+0.1.0，结论不可比）。
设备：NPU 5（`ASCEND_RT_VISIBLE_DEVICES=5`）。

时延口径：benchmark.py 按 kernel `Name` 分组、取最后 repeats 行再除以 repeats，
**同名 kernel 一次 forward 内发射多次只计一次**；总时延 = 各 kernel name 的均值之和。
因此 **kernel name 的个数直接进分子**（少一个 kernel 就少一份耗时），
本文所有 speedup 均为该口径下 50 case 的几何平均，非真实端到端加速比。

---

## 轮次记录

### base —— 历史实现在正确环境下的复测（保底存档）
- 改动：无，就是 `100_BlockSparseAttnBwd_generated.py` 原样。
- 依据：任务交接说它在 CANN 8.5.1 下 12/50 编译失败、历史 speedup 10.13x；
  需要先确认 9.1.0 下的真实基准。
- verify：**50/50**（8.5.1 下的 12 个 `Failed to compile HIVM IR / ub overflow`
  在 9.1.0 + bishengir 1.2.0 上全部消失）。
- speedup：**5.3176x**（fw 5.5098ms / impl 1.0362ms，50 case 几何平均）。
- 判定：**存档为 best 保底**（已 ≥3x）。
- 重要结论：历史那个 **10.13x 不可比** —— 它是在 12 个 case 编译失败、
  只有 38 个（且是偏小的）case 进几何平均的情况下得到的；补齐 50 个 case 后
  真实基准是 5.32x。后续都以 5.32x 为参照。
- 瓶颈画像（impl 侧，按 kernel Name 均值）：
  | kernel | case1 | case44 | case48 |
  |---|---|---|---|
  | compute_dq_kernel | 41.0us | 2596us | 1237us |
  | compute_dk_kernel | 42.3us | 2060us | 715us |
  | compute_dv_kernel | 30.2us | 1789us | 518us |
  | compute_delta_kernel | 4.7us | 20.9us | 75.3us |
  | aclnnCast（3 次/call） | 3.9us | 7.0us | 9.3us |
  | aclnnInplaceZero（4 次/call） | 2.9us | 5.8us | 8.0us |
  dq/dk/dv 三个 kernel 占 ~95%，且三者都各自重算一遍 S/P；
  q/k/v/dout 全部 `.to(tl.float32)` 后再 tl.dot，Cube 走 fp32 路径。

