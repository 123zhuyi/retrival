# Polar 测试框架（S8 同款）

`out = input · (cos(angle) + i·sin(angle))`，input(abs)/angle 为 fp32 可广播，out 为 complex64。
对齐基准：`cann/ops-math` 的 `math/complex/op_host/op_api/aclnn_polar.cpp`（l0 拼接版）。

## 文件构成（与 S8 case_910b/\<Op\>/ 完全一致）

| 文件 | 来源 | 说明 |
|---|---|---|
| `common/pytorch_npu_helper.hpp` | **逐字复用 S8** | `EXEC_NPU_CMD` 通用胶水（算子无关，动态 dlopen `aclnnPolar`） |
| `setup.py` / `get_time.py` | **逐字复用 S8** | pybind wheel 构建 / msprof `op_summary*.csv` 取 [20:40] 均值 |
| `extension/custom_op.cpp` | Polar 适配 | `EXEC_NPU_CMD(aclnnPolar, input, angle, result)`，result=complex64、shape=broadcast(input,angle)，50 轮供 msprof 取平均 |
| `test_op.py` | Polar 适配 | 8 个 case + 复数 verify（`view_as_real` 拆 real/imag 当 fp32，rtol/atol=1e-4） |
| `run.sh` | Polar 适配 | 重建 wheel → msprof 计时 → get_time → 基线校验；LD 指向 `custom_math` vendor |

## 前置（NPU 环境，改 kernel 后重做）

```bash
# 在 ops-math 仓编译并安装自定义 Polar 算子
bash build.sh --pkg --soc=ascend910b --ops=polar -j16     # A3: --soc=ascend910_93
./build_out/cann-ops-math-*linux*.run
```

装到 `${ASCEND_HOME_PATH}/opp/vendors/custom_math/`。

## 用法

```bash
cd case_910b/Polar
bash run.sh 2          # 跑 case2（广播主战场）
```

输出 `xxx verify result pass!` + `time_base = ... time_use = ...`。

## case 覆盖

| case | input shape | angle shape | 考点 |
|---|---|---|---|
| 1 | [2,6,10] | [2,6,10] | 同 shape 基础 |
| 2 | [4,1,8] | [4,5,8] | **广播：低维→高维（新增功能主战场）** |
| 3 | [1] | [3,4,5] | 广播：标量 input |
| 4 | [8,1] | [1,7] | 广播：双向 → [8,7] |
| 5 | [4096,4096] | [4096,4096] | 大 shape（性能） |
| 6 | [3,5,17,269] | 同 | 高维 + inner 非 32B 对齐（1076B/行） |
| 7 | [1] | [1] | 1 元素边界 |
| 8 | [64,1024] | [64,1024] | 大角度（Sin/Cos 范围归约压力） |

新增 case 直接往 `test_op.py` 的 `case_data` 加键即可（与 S8 习惯一致）。

## 与 S8 的差异（仅构建层）

- S8：每算子自带 `S8/<Op>/build.sh`，`auto_submit.sh` 直接 build。
- Polar：构建走 **ops-math 仓 `build.sh --pkg --ops=polar`**（仓级），与本测试框架解耦。
- `auto_submit.sh -o Polar` 仍可复用其 upload/run 阶段；build 阶段需替换为 ops-math 命令（待算子工程落地后再接线）。

## 注意

- 精度阈值现为 fp32 rtol/atol=1e-4；任务要求"AscendOpTest 默认阈值"，待 AscendOpTest 工具实际阈值确认后于 `test_op.py:verify_result` 回填。
- 性能基线 `run.sh:time_base` 现为哨兵值（仅正确性门禁）；硬件实测出 l0 参考实现耗时后回填真实基线（任务要求 ≥ 基线 95%）。
- 本框架只能在 NPU 环境运行（依赖 `torch_npu`），本地无法自检——与 S8 相同。
