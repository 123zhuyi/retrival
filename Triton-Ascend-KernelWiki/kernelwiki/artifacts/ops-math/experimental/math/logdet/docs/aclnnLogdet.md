# aclnnLogdet

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：计算输入方阵或方阵 batch `self` 的行列式自然对数，并将结果写入 `out`。
- 计算公式：

  设 $\det(\text{self})$ 表示对 `self` 最后两维对应的每个方阵分别计算行列式：

  $$
  \text{signValue} = \text{sign}(\det(\text{self}))
  $$

  $$
  \text{logAbsValue} = \log(|\det(\text{self})|)
  $$

  $$
  \text{out\_tmp} = \log(\text{signValue}) + \text{logAbsValue}
  $$

  $$
  \text{out} = \text{cast}(\text{out\_tmp})
  $$

  其中：
  - 当 $\det(\text{self}) > 0$ 时，$\text{out} = \log(\det(\text{self}))$。
  - 当 $\det(\text{self}) = 0$ 时，$\text{out} = -\infty$。
  - 当 $\det(\text{self}) < 0$ 时，$\text{out} = \text{NaN}$。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnLogdetGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnLogdet"接口执行计算。

```cpp
aclnnStatus aclnnLogdetGetWorkspaceSize(
  const aclTensor  *self,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnLogdet(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnLogdetGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>待计算行列式自然对数的输入方阵或方阵 batch，对应公式中 self。</td>
      <td><ul><li>支持空Tensor（空Tensor时直接返回成功，workspaceSize为0）。</li><li>最后两维必须构成方阵。</li><li>维度数不少于2。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>至少2维，shape为(*, n, n)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>self 的行列式自然对数结果，对应公式中 out。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需为FLOAT。</li><li>shape需严格等于self去掉最后两维后的batch shape，不支持broadcast。</li><li>当self为2维时，out为0维标量。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>self去掉最后两维后的batch shape</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>self或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>self或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的维度数小于2。</td>
    </tr>
    <tr>
      <td>self的最后两维不相等（不是方阵）。</td>
    </tr>
    <tr>
      <td>out的shape不等于self去掉最后两维后的batch shape。</td>
    </tr>
  </tbody></table>

## aclnnLogdet

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
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnLogdetGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnLogdet默认确定性实现。
- 仅支持float32数据类型。
- 非连续self会先转为连续Tensor后计算；非连续out通过ViewCopy回写。
- 支持大于8维的self，内部先reshape到3维计算后再reshape回原batch shape。
- 使用带部分主元选取的LU分解算法。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "aclnn_logdet.h"

int RunLogdetExample(aclrtStream stream, aclTensor* self, aclTensor* out)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段接口：做参数校验并返回 workspace 大小与执行器
    aclnnStatus ret = aclnnLogdetGetWorkspaceSize(self, out, &workspaceSize, &executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclError aclRet = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            return static_cast<int>(aclRet);
        }
    }

    // 第二段接口：在指定 stream 上执行
    ret = aclnnLogdet(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        return ret;
    }

    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    return ACLNN_SUCCESS;
}
```

### 示例说明

- `self` 为输入 Tensor，shape 需满足 `(*, n, n)`，dtype 必须为 `FLOAT`。
- `out` 为输出 Tensor，shape 必须等于 `self.shape[:-2]`，当 `self` 为 2 维时，`out` 为 0 维标量。
- 当 `self` 为空 Tensor 时，`aclnnLogdetGetWorkspaceSize` 会直接返回成功，且 `workspaceSize=0`。
- 非连续 `self/out` 可以直接传入，接口内部会分别处理连续化和回写。

### 一个 2x2 输入示例

若输入矩阵为：

```text
[[2.0, 0.0],
 [0.0, 3.0]]
```

则输出为标量：

```text
log(det(self)) = log(6.0)
```
