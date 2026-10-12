# FusedMulAddN — PyTorch 接入 ST 测试（真实 NPU + ACLNN 两段式）

算子公式：`y_i = x1_i * x3[0] + x2_i`（`x3` 为单元素标量张量 ShapeSize=1，仅取 `x3[0]`，按标量广播）。

本目录通过 PyTorch 适配层接入自定义算子包的 ACLNN 两段式接口
（`aclnnFusedMulAddNGetWorkspaceSize` + `aclnnFusedMulAddN`），在真实 NPU（ascend910b）上
对拍 CPU golden，完成 L0+L1 全量功能/精度/边界/极端/确定性验证。

## 数据流

```text
PyTorch Tensor (NPU) -> torch_adapter.so -> aclnnFusedMulAddN(两段式) -> NPU 结果
CPU golden (golden.py)  <-- compare.py（MERE/MARE 社区标准 / 整型 bitwise）--> NPU 结果(取回 CPU)
```

## 文件

| 文件 | 说明 |
|------|------|
| `golden.py` | CPU golden `y=x1*x3[0]+x2`；浮点在 fp32 域算，整型两步回绕（mul→add） |
| `compare.py` | 精度比对：浮点 MERE/MARE（per_dtype 阈值），整型 bitwise；NaN/Inf 位置单独判定 |
| `test.py` | L0+L1 全量用例 + 调度；支持 `--golden-only`/`--level`/`--case` |
| `torch_adapter.cpp` | PyTorch 算子注册 + ACLNN 两段式封装（x1,x2,x3→y） |
| `CMakeLists.txt` | 构建 `libtorch_adapter.so`，链接自定义包 `libcust_opapi.so` |
| `run_torch.sh` | 一键编译 + 运行（自动定位 custom_math 包并设置运行时环境） |

## 前置：自定义算子包

torch_adapter 链接自定义包的 op_api lib，运行时需自定义 kernel。先构建并解包：

```bash
# 在 worktree 根
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n
bash build_out/cann-ops-math-custom_linux-*.run --noexec --quiet \
     --extract=$(pwd)/build_out/installed_custom_math
```

解包后得到 `build_out/installed_custom_math/packages/vendors/custom_math/`，含
`op_api/include/aclnn_fused_mul_add_n.h`、`op_api/lib/libcust_opapi.so` 及
`op_impl/.../kernel/ascend910b/fused_mul_add_n`（5 dtype kernel）。

## 运行

```bash
# 一键（推荐）：自动定位 custom_math、编译、设置 ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH、跑全量
bash run_torch.sh

# 仅 CPU golden 自测（无需 NPU / 自定义包）
bash run_torch.sh --golden-only

# 只跑 L0 / 指定用例
bash run_torch.sh --level L0
bash run_torch.sh --case L1_023
```

手动方式：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
         -DCUSTOM_OP_VENDOR_DIR=<.../vendors/custom_math>
make
export ASCEND_CUSTOM_OPP_PATH=<.../vendors/custom_math>
export LD_LIBRARY_PATH=<.../vendors/custom_math>/op_api/lib:$LD_LIBRARY_PATH
python3 ../test.py --lib ./libtorch_adapter.so
```

## 用例覆盖（对应 C++ ST 全量）

- **5 dtype**：float32 / float16 / bfloat16 / int32 / int16（全覆盖）
- **多 shape**：rank0 `[]` / `[1]` / `[1,1]`(x3形态) / 1D / 2D / 3D / 4D / 大 shape `[4096,1024]`（多核/UB 分块）
- **边界不变量**：x3=0 ⇒ y==x2（`zero_multiplier_yields_x2`）；x3=1 ⇒ y==x1+x2；空 tensor `returns_empty`
- **极端输入**：x1 含 NaN ⇒ NaN 传播；x1=+Inf ⇒ 与 oracle 一致；全零 ⇒ y 全 0；fp16 上界 60000；int32/int16 上下界回绕
- **确定性**：同输入连续执行 3 次 bitwise 一致（小 shape / 大 shape 多核）

## 精度标准（per_dtype，社区标准）

| dtype | 方法 | 阈值(MERE) |
|-------|------|-----------|
| float32 | MERE/MARE | 2^-13 ≈ 1.220703125e-04 |
| float16 | MERE/MARE | 2^-10 ≈ 9.765625e-04 |
| bfloat16| MERE/MARE | 2^-7 ≈ 7.8125e-03 |
| int32 / int16 | bitwise_equal | 0（绝对误差 0） |

> 报错路径（shape 不一致 / x3 非单元素 / dtype 不一致 / 非法 dtype / null）由 op_host UT
> 承担，本 PyTorch ST 聚焦上板功能/精度/确定性正向覆盖。
