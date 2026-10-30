# Real

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：提取复数张量的实部（real part）。对于实数类型输入，输出等于输入（恒等操作）。

- 计算公式：

$$
output_i=\text{real}(input_i)=\begin{cases}
\text{Re}(input_i), & \text{if } input_i \text{ is complex} \\
input_i, & \text{if } input_i \text{ is real}
\end{cases}
$$

- 与PyTorch对应：`torch.real()`
- 与NumPy对应：`np.real()`

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>待提取实部的输入张量。</td>
      <td>COMPLEX32, COMPLEX64, COMPLEX128, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>提取出的实部张量。</td>
      <td>FLOAT16, FLOAT, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>Tout</td>
      <td>属性</td>
      <td>可选属性，指定输出数据类型。默认值为DT_FLOAT(float32)。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 数据类型映射

| 输入类型 | 输出类型 | Tiling Key | Tout值 | 说明 |
|---------|---------|-----------|--------|------|
| COMPLEX128 | DOUBLE | - | - | 提取复数实部 (AICPU only) |
| COMPLEX32 | FLOAT16 | 1 | 1 | 提取复数实部 |
| COMPLEX64 | FLOAT | 2 | 0 | 提取复数实部 |
| FLOAT16 | FLOAT16 | 4 | 1 | 恒等操作 |
| FLOAT | FLOAT | 5 | 0 | 恒等操作 |

**注：**

- COMPLEX128 类型仅支持 AICPU 实现，ascend910b 的 AICore 不支持该类型。
- Tout值为数据类型枚举值（DT_FLOAT=0, DT_FLOAT16=1, DT_DOUBLE=11），可通过Tout属性可选指定输出类型。

## 约束说明

- 输入张量的shape必须与输出张量的shape相同
- 支持动态shape和动态rank
- 输入数据不能为空

## 实现说明

### 目录结构

```text
real/
├── op_host/                    # Host侧实现
│   ├── real_def.cpp           # 算子定义
│   ├── real_infershape.cpp    # Shape推导
│   └── real_tiling.cpp        # Tiling计算实现
├── op_kernel/                  # Kernel侧实现
│   ├── real.cpp               # Kernel入口
│   ├── real_kernel.h          # Kernel模板类实现
│   └── real_tiling.h          # Tiling数据结构定义
├── op_api/                     # API接口
│   ├── aclnn_real.cpp         # ACLNN接口实现
│   ├── aclnn_real.h           # ACLNN接口声明
│   ├── real.cpp               # 算子API实现
│   └── real.h                 # 算子API声明
├── docs/
│   └── aclnnReal.md           # API文档
├── examples/
│   └── test_aclnn_real.cpp    # ACLNN调用示例
└── tests/ut/                   # 单元测试
    ├── op_api/
    │   └── test_aclnn_real.cpp
    ├── op_host/
    │   └── test_real_tiling.cpp
    └── op_kernel/
        ├── test_real.cpp
        └── real_data/
            ├── gen_data.py
            └── compare_data.py
```

### Tiling参数说明

- `totalUsedCoreNum`: 实际使用的总核数
- `tailBlockNum`: 大核数量（余数block数）
- `ubPartDataNum`: 每次UB循环处理的元素数
- `smallCoreDataNum`: 小核数据量（元素数）
- `smallCoreLoopNum`: 小核UB循环次数
- `smallCoreTailDataNum`: 小核最后一次循环的元素数
- `bigCoreDataNum`: 大核数据量（元素数）
- `bigCoreLoopNum`: 大核UB循环次数
- `bigCoreTailDataNum`: 大核最后一次循环的元素数
- `tilingKey`: 算子类型标识（1=complex32, 2=complex64, 4=float16, 5=float）
- `useNonInplace`: 是否使用非inplace GatherMask路径（0=inplace, 1=非inplace）

### 多核处理策略（大小核分核）

1. **对齐粒度**：
   - Complex类型：128B对齐（满足GatherMask inplace的256B源数据约束）
   - Real类型：32B对齐

2. **分核策略**（参考Exp算子）：
   - 按输出数据类型字节数将总数据量对齐到对应block
   - `totalBlocks = Align(totalLength * dataTypeLength, alignSize) / alignSize`
   - 若 `ubPartDataNum >= totalLength`：使用1核
   - 否则：`coreNum = min(totalCoreNum, totalBlocks)`
   - `everyCoreBlockNum = totalBlocks / coreNum`，`tailBlockNum = totalBlocks % coreNum`

3. **大小核分配**：
   - 前 `tailBlockNum` 个核为大核，数据量 = `(everyCoreBlockNum + 1) * alignSize / dataTypeLength`
   - 其余核为小核，数据量 = `everyCoreBlockNum * alignSize / dataTypeLength`
   - 大小核负载差 ≤ 1 个 block

4. **偏移计算**：
   - 大核：`globalOffset = blockIdx * bigCoreDataNum`
   - 小核：`globalOffset = blockIdx * bigCoreDataNum - (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum)`
   - Complex类型输入：`inputOffset = globalOffset * 2`（每个元素占2个output元素空间）

### Complex双路径策略

GatherMask inplace要求 `count * 2 * sizeof(T) % 256 == 0`，tiling据此选择路径：

1. **Inplace路径**（`useNonInplace=0`）：
   - 适用：多核场景 / 单核且totalLength满足256B对齐
   - UB分配：`inQueue(2x) × 2缓冲 = 4倍系数`
   - GatherMask inplace：`GatherMask(src, src, mode=1, ...)`，pipeline prefetch优化

2. **非Inplace路径**（`useNonInplace=1`）：
   - 适用：单核且totalLength不满足256B对齐（如complex32 [4,4]=16元素，16×2×2=64 < 256）
   - UB分配：`inQueue(2x) + outQueue(1x) × 2缓冲 = 6倍系数`
   - GatherMask非inplace：`GatherMask(dst, src, mode=1, mask=count*2, repeatTimes=1)`

## 调用说明

### ACLNN API调用

参见 `examples/test_aclnn_real.cpp` 示例代码：

```cpp
#include "aclnnop/aclnn_real.h"

// 1. 获取workspace大小
aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor* self,
                                      aclTensor* out,
                                      uint64_t* workspaceSize,
                                      aclOpExecutor** executor);

// 2. 执行算子
aclnnStatus aclnnReal(void* workspace,
                      uint64_t workspaceSize,
                      aclOpExecutor* executor,
                      aclrtStream stream);
```

**注意：** ACLNN API使用`self`和`out`作为参数名，与图模式的`input`/`output`不同。

## 参考文献

- [PyTorch torch.real](https://pytorch.org/docs/stable/generated/torch.real.html)
- [NumPy np.real](https://numpy.org/doc/stable/reference/generated/numpy.real.html)
- [AscendC编程指南](https://www.hiascend.com/document)

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| alfengyuan | 个人开发者 | Real | 2026/3/30 | 新增Real算子 |
