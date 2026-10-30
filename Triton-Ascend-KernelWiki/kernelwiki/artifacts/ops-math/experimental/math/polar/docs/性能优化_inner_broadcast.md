# Polar inner-broadcast 性能优化（方案 A）

> 2026-05-21。本轮针对 acceptance 50-case 中**唯一性能不达 baseline 的两个 case**（Test_036 / Test_044）
> 做 angle inner-broadcast 专用优化，从输基线 0.8x 反超到 1.7-3.5x，并使所有 angle 周期广播 case 普遍 ~2x 提速。

## 1. 问题定位

测试口径：pybind + EXEC_NPU_CMD(aclnnPolar)，msprof op_summary Task Duration（每调用设备时，50 次取中位）。
baseline = `torch.polar`（NPU 内置，Sin/Cos/Mul/Complex 基础算子拼接）。

优化前全 50 case 仅这两个输基线：

| Case | input × angle | out 元素 | 优化前 ours | baseline | vs base |
|---|---|---:|---:|---:|---:|
| Test_036 | [256,256,64] × **[64]** | 4M | 168.8 µs | 147.2 µs | **0.87x** ❌ |
| Test_044 | [128,128,512] × **[512]** | 8M | 331.4 µs | 276.6 µs | **0.83x** ❌ |

**根因**：这两个是 "angle 为 input trailing-dim 的周期重复" 广播。原实现在 op_api 层用 `BroadcastTo`
把 angle 物理展开到 out shape（同 shape kernel 要求），于是：

- Test_036：先写 16 MB（angle 展开）→ kernel 再读 input 16 MB + 读展开后 angle 16 MB + 写 out 32 MB = **80 MB HBM**
- baseline 用 view（stride=0）读 angle，省掉展开的一次 HBM 写，故反而更快。

## 2. 方案 A：inner-broadcast 周期 pre-compute（kernel 内运行时分支，无 tilingKey）

**核心观察**：angle 周期长度 K（= angle.numel），每个 tile 取整数个 period →
**cos/sin tile pattern 对所有 tile 完全相同**。

实现（三处资格判定严格一致：op_api / op_host tiling / pybind）：

```text
资格 = inN==outN && anN<inN && inN%anN==0 && anN<=2048 && anN%8==0
```

- **op_api (`aclnn_polar.cpp`)**：资格命中时 angle **不 BroadcastTo**，保留原 shape [K] 传 kernel
- **op_host (`polar_tiling.cpp`)**：识别 `bcastMode=1`，按 **K-block 分核**（每核负载 K 整数倍 → tile 起点 period 对齐）
- **op_kernel (`polar.h`)**：`Process()` 内 `if(bcastMode==1)` 分支（纯 data 字段，无需多 kernel 注册）
  - `PrecomputeBcastCosSin()`：读 angle[0:K] → Cos/Sin 得 cosT/sinT[K] → 循环 m=T/K 次 broadcast 填满
    常驻 buffer（cosTile[0:T] / sinTile[T:2T]），**只做一次**
  - `ProcessTileBcast()`：main loop **只读 input** → `Mul(cosTile,input)`→real / `Mul(sinTile,input)`→imag
    → Gather 交织 → 写 out。无 angle 搬运、无每 tile Sin/Cos 重算。

**HBM/算力收益（Test_036, 4M）**：80 MB → **48 MB**（省 angle 展开 16 MB + angle 读 16 MB→256 B）；
Sin/Cos 从"每 tile 重算"降到"一次 precompute"。

> ⚠️ 实现陷阱：`DataCopy(srcStride=0)` **不是** "重复同段 broadcast"——它连续读 src，会把 cosT 之后的
> sinT 读进第二 period（症状 `npu.real==golden.imag`）。正解是把 broadcast 提到 Precompute 一次性循环填满。

## 3. 优化结果

### 关注的两个 case：从输基线反超

| Case | K | 优化前 | **优化后** | kernel 提速 | baseline | vs base 前→后 |
|---|---|---:|---:|---:|---:|---:|
| Test_036 | 64 | 168.8 | **42.1** | **4.0x** | 147.2 | 0.87x → **3.50x** |
| Test_044 | 512 | 331.4 | **161.9** | 2.0x | 276.6 | 0.83x → **1.71x** |

### 其它 angle 周期广播 case（顺带普遍 ~2x）

| Case | out 元素 | K | 优化前 | 优化后 | baseline | vs base |
|---|---:|---:|---:|---:|---:|---:|
| Test_035 | 262K | 64 | 17.6 | 15.7 | 54.7 | 3.47x |
| Test_037 | 16M | 64 | 654 | 321 | 765 | 2.38x |
| Test_038 | 67M | 64 | 2595 | 1261 | 3340 | 2.65x |
| Test_039 | 134M | 32 | 5181 | 2531 | 6771 | 2.67x |
| Test_040 | 268M | 16 | 10354 | 5144 | 13633 | 2.65x |
| Test_041 | 537M | 8 | 20701 | 10287 | 27361 | 2.66x |
| Test_047 | 134M | 8 | 5181 | 2581 | 6784 | 2.63x |
| Test_048 | 33M | 32 | 1301 | 641 | 1639 | 2.56x |

未启用优化（K=1/2/4 不满足 32B 对齐，走 same-shape 兜底）的 Test_042/045/046：因规模大（268-537M）
baseline 同为 HBM bound，same-shape 已 1.31-1.47x 胜基线，无需再优化。

### 全 50-case 结论：无一输基线

| 类别 | case | 加速比 |
|---|---|---|
| 小同 shape (1-12) | same-shape | 2.5-3.4x |
| 大同 shape (13-29) | same-shape | 2.5-3.1x |
| 小广播 (30-34, K<8) | same-shape 兜底 | 3.2-5.3x |
| 大广播 (35-50) | bcast 资格 ~2x kernel 提速；非资格规模大也胜 | 1.3-3.5x |

## 4. 官方 AscendOpTest 验证（ops-math 工程，customize_math vendor）

```bash
python run_test.py -i .../acceptance/op.json -c <Test_036,Test_044> \
  --op-type custom --op-path $ASCEND_OPP_PATH/vendors/customize_math/op_api --build
```

结果 `result.csv`：

| case | compare_result |
|---|---|
| Test_036 | **pass** |
| Test_044 | **pass** |

精度判据 err_threshold=[1e-3, 1e-3]，官方工具确认数值正确。

## 5. 改动文件（ops-math）

- `op_kernel/polar_tiling_data.h`：+ `inN/anN/bcastMode`
- `op_host/polar_tiling.cpp`：inner-bcast 检测 + K-block 分核
- `op_kernel/polar.h`：`Process()` 分支 + `PrecomputeBcastCosSin()` + `ProcessTileBcast()`
- `op_api/aclnn_polar.cpp`：资格命中时 angle 不 BroadcastTo
- `op_api/polar.cpp`：注释更正

同 shape 与非 inner-bcast 广播路径（中间轴 / input 侧广播，如 Test_043/049/050）行为不变，由 same-shape kernel + op_api BroadcastTo 兜底。
