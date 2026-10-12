# SortWithIndex —— PyTorch ST 测试工程（ascend910b）

> 算子：`sort_with_index`（OpType `SortWithIndex`）；目标芯片：ascend910b（DAV_2201）。
> 范式：Sort（索引跟随排序 / 重排）。

## 1. 目录结构

```text
tests/st/torch/
├── README.md           # 本文件
├── CMakeLists.txt      # 编译 torch_adapter.cpp -> libtorch_adapter.so（含 ACLNN 两段式封装）
├── torch_adapter.cpp   # PyTorch 注册 sort_with_index.forward + ACLNN aclnnSortWithIndex 两段式封装
├── golden.py           # CPU golden（910B NaN-开头语义）+ golden 自测
├── compare.py          # 精度比对（重排算子 bitwise + isnan-aware + 3 个结构性不变量）
└── test.py             # 全量用例定义 + 调度（4 组 dtype + 边界 + extreme + 负向）
```

## 2. 910B 实现语义（golden 必须遵循，不可用 torch.sort 默认约定）

| 语义点 | 910B 实现（本 golden 遵循） | torch.sort 默认（本 golden **不**用） |
|--------|----------------------------|--------------------------------------|
| 升序 NaN 落位 | **序列开头**（rank0 起） | 序列末尾 |
| 降序 NaN 落位 | 序列开头 | 序列开头 |
| NaN 比较 | 按 `isnan`（位型经 Muls(-1) 翻符号位，值仍 NaN） | 按位型 |
| ±Inf | +Inf 升序末尾、-Inf 升序开头 | 同 |
| ±0 | 视为相等 ties（stable 原始位置升序） | 同 |
| 重排精度 | value bitwise（有限值）+ indices 精确（rtol=atol=0） | — |

> golden.py 用 `np.lexsort`（稳定排序）+ 双路键（NaN-primary、ascending/descending-secondary）实现，
> 等价 C++ ST `ComputeGolden910b`（`std::stable_sort`），**独立于 `torch.sort` 的 NaN 约定**。

## 3. dtype 范围（4 组）

value{float16, float32, bfloat16, int32} × index{**int32**}（无 int64-index）。
int32 value 测试数据限定 `|x| ≤ 2^24`（910B 经 float 排序的值域）。

## 4. 用例覆盖

- **A**. 每 dtype 基础 shape × descending × stable；
- **B**. 多 rank（rank0–8）+ axis(-1, rank-1)；
- **C**. 单 tile 内多轴长（≤ 单 tile 上限：fp16~3008/fp32~2816/bf16~2816/int32~2560）；
- **D**. 边界：空 tensor `[0]`/`[3,0]`/`[0,8]`；
- **E**. extreme：NaN(升序落开头)/±Inf/全零/全相等/±0；
- **F**. 确定性：含 ties 重复执行；
- **负向**：x≠index shape、axis 非最后一维、axis 越界、轴长超上限。

## 5. 运行

### 5.1 CPU Golden 自测（无需 NPU）

```bash
python3 golden.py        # 仅 golden 自测（910B NaN-开头）
python3 test.py          # golden 自测 + 全量用例 mock 闭环（验证 golden + 比对逻辑）
```

### 5.2 编译 libtorch_adapter.so

```bash
mkdir -p build && cd build
# 指向已安装的 sort_with_index 自定义算子包（vendor=custom_math），提供 aclnn_sort_with_index.h + libcust_opapi.so
export ASCEND_CUSTOM_OPP_PATH=<vendor_root>/vendors/custom_math:
cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
make                     # 生成 libtorch_adapter.so
```

### 5.3 NPU 实跑（TorchNPU + NPU + 自定义算子包就绪时）

```bash
python3 ../test.py --lib ./libtorch_adapter.so
```

> TorchNPU不可用时，`test.py`自动回退为CPU golden mock闭环（不阻断）。
