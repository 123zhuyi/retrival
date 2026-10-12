# aclnnAcoshGrad

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |

## 功能说明

- 接口功能：计算 Acosh 的反向梯度。输入前向 Acosh 的输出 `y` 和上游梯度 `dy`，结果写入输出 `dx`。
- 计算公式：

  $$
  dx_i = \frac{dy_i}{\sinh(y_i)}
  $$

  实现中按 TBE 等价路径计算 `sinh(y)`：先计算 `s = y / 8`，使用 7 阶 Taylor 多项式近似 `sinh(s)`，再连续执行 3 次 `2 * v * sqrt(v * v + 1)` 倍角放大。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnAcoshGradGetWorkspaceSize` 获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用 `aclnnAcoshGrad` 执行计算。

```cpp
aclnnStatus aclnnAcoshGradGetWorkspaceSize(
    const aclTensor* y,
    const aclTensor* dy,
    aclTensor* dx,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnAcoshGrad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnAcoshGradGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 260px">
  <col style="width: 380px">
  <col style="width: 220px">
  <col style="width: 90px">
  <col style="width: 90px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>前向 Acosh 算子的输出 tensor。</td>
      <td>shape 和 dtype 必须与 dy 一致；元素数需大于 0。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>上游传播的梯度 tensor。</td>
      <td>shape 和 dtype 必须与 y 一致，不支持 broadcast。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>反向梯度输出 tensor。</td>
      <td>shape 和 dtype 与 y 保持一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  `aclnnStatus`：返回状态码。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>y、dy、dx、workspaceSize 或 executor 为空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>y、dy 或 dx 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>y、dy、dx 的数据类型不一致。</td>
    </tr>
    <tr>
      <td>y、dy、dx 的 shape 不一致，或输入元素数为 0。</td>
    </tr>
  </tbody>
  </table>

## aclnnAcoshGrad

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在 Device 侧申请的 workspace 内存地址。workspaceSize 为 0 时可传空指针。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>workspace 大小，由第一段接口获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op 执行器，包含算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的 Stream。</td></tr>
  </tbody>
  </table>

- **返回值**

  `aclnnStatus`：返回状态码。

## 约束说明

- 仅支持 FLOAT16、FLOAT、BFLOAT16。
- 仅支持 ND 格式。
- `y`、`dy`、`dx` 的 shape 必须完全一致，不支持 broadcast。
- 输入元素数需大于 0。
- FLOAT16、BFLOAT16 在 kernel 内部使用 FLOAT32 计算后再转换为原 dtype。
- `sinh(y)==0` 时结果按硬件除法行为产生 `inf` 或 `nan`，调用方需根据业务场景保证输入合法性。

## 调用示例

完整样例代码见 [test_aclnn_acosh_grad.cpp](../examples/test_aclnn_acosh_grad.cpp)。核心调用流程如下：

```cpp
#include "acl/acl.h"
#include "aclnn_acosh_grad.h"

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnAcoshGradGetWorkspaceSize(yTensor, dyTensor, dxTensor, &workspaceSize, &executor);

void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

aclnnAcoshGrad(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```
