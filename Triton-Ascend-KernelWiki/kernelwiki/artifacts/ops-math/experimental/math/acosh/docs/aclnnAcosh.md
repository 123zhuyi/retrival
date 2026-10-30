# aclnnAcosh

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：对输入张量 self 逐元素计算反双曲余弦值，结果写入输出张量 out。
- 计算公式：

  $$
  out_i = \cosh^{-1}(self_i) = \ln\left(self_i + \sqrt{self_i^2 - 1}\right)
  $$

  其中 $self_i$ 表示输入张量第 i 个元素，$out_i$ 表示对应输出元素。当 $self_i < 1$ 时，数学上无实数解，计算结果为 NaN，与 PyTorch `torch.acosh` 语义一致。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnAcoshGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnAcosh"接口执行计算。

```cpp
aclnnStatus aclnnAcoshGetWorkspaceSize(
  const aclTensor  *self,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnAcosh(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnAcoshGetWorkspaceSize

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
      <td>输入张量，对应公式中self，表示待计算反双曲余弦值的数据。</td>
      <td><ul><li>支持空Tensor（元素数为0）。</li><li>数据类型需与out保持一致。</li><li>shape需与out完全一致。</li><li>输入值域建议为[1, +∞)；小于1的值会产生NaN，框架层不做拦截。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量，对应公式中out，用于存放反双曲余弦计算结果。</td>
      <td><ul><li>不支持空Tensor（self为空Tensor时除外）。</li><li>数据类型需与self一致。</li><li>shape需与self完全一致。</li></ul></td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
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
      <td>self或out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self或out的数据类型不在支持的范围之内（仅支持FLOAT16、FLOAT、BFLOAT16）。</td>
    </tr>
    <tr>
      <td>self与out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self或out的shape维度不在支持的范围之内（仅支持0-8维），或self与out的shape不一致。</td>
    </tr>
  </tbody></table>

## aclnnAcosh

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnAcoshGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnAcosh默认确定性实现。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持FLOAT16、FLOAT、BFLOAT16数据类型，不支持其他数据类型。
- 输入值域约束：acosh数学定义域为[1, +∞)，输入元素小于1时结果为NaN，接口层不做拦截，由调用方保证输入合法性。
- 逐元素算子，不涉及广播，输入输出shape须完全一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

完整示例代码见 `examples/example_aclnn_acosh.cpp`，以下为核心调用流程：

```cpp
#include "acl/acl.h"
#include "aclnn_acosh.h"

// 1. ACL 初始化
aclInit(nullptr);
aclrtSetDevice(0);
aclrtStream stream = nullptr;
aclrtCreateStream(&stream);

// 2. 准备输入数据（以 float32 为例，计算 acosh([1.0, 2.0, 5.0, 10.0])）
std::vector<float> inputData = {1.0f, 2.0f, 5.0f, 10.0f};
std::vector<int64_t> shape = {4};
size_t dataSize = inputData.size() * sizeof(float);

// 3. 分配 NPU 内存并拷贝输入数据
void* inputDev = nullptr;
void* outputDev = nullptr;
aclrtMalloc(&inputDev, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&outputDev, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(inputDev, dataSize, inputData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);

// 4. 创建 aclTensor（连续 ND 格式）
std::vector<int64_t> strides = {1};   // shape={4} 时 stride={1}
aclTensor* selfTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
    strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), inputDev);
aclTensor* outTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
    strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), outputDev);

// 5. 第一段接口：获取 workspace 大小和 executor
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnAcoshGetWorkspaceSize(selfTensor, outTensor, &workspaceSize, &executor);

// 6. 分配 workspace（大小为 0 时跳过）
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 7. 第二段接口：执行算子
aclnnAcosh(workspace, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);

// 8. 拷回结果
std::vector<float> output(inputData.size());
aclrtMemcpy(output.data(), dataSize, outputDev, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
// output = [0.0, 1.3169578, 2.2924316, 2.9932227]

// 9. 清理资源
if (workspace) aclrtFree(workspace);
aclDestroyTensor(selfTensor);
aclDestroyTensor(outTensor);
aclrtFree(inputDev);
aclrtFree(outputDev);
aclrtDestroyStream(stream);
aclrtResetDevice(0);
aclFinalize();
```
