# Sin

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：对输入矩阵的每个元素进行正弦操作后输出。

- 计算公式：

$$
out_{i}=sin(input_{i})
$$

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
      <td>self</td>
      <td>输入</td>
      <td>待进行sin计算的入参，公式中的input_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行sin计算的出参，公式中的out_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## aclnnSinGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*，计算输入): Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，shape维度不大于8，且shape需要与out一致，和out的数据类型满足数据类型推导规则。当输入的数据类型为INT8、INT16、INT32、INT64、UINT8、BOOL时，会转换为float数据类型做计算，计算结果转换成out的数据类型。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT8、INT16、INT32、INT64、UINT8、BOOL、FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - out(aclTensor *，计算输出): Device侧的aclTensor，支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，且shape需要与self一致，和self的数据类型满足数据类型推导规则。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1.传入的self或out是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. self和out的数据类型和数据格式不在支持的范围之内。
                         2. self和out的数据类型不满足数据类型推导规则。
                         3. self和out的维度大于8。
                         4. self和out的shape不一致。
```

## aclnnSin

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnSinGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## aclnnInplaceSinGetWorkspaceSize

- **参数说明：**

  - selfRef(aclTensor *，计算输出|计算输入): 支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND，shape维度不大于8。
    - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16。
  - workspaceSize(uint64_t *，出参)：返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor \**，出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现如下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1.传入的selfRef是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef数据类型和数据格式不在支持的范围之内。
                                 2. selfRef的维度大于8
```

## aclnnInplaceSin

- **参数说明：**

  - workspace(void *，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceSinGetWorkspaceSize获取。
  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、FLOAT16、BFLOAT16数据类型的输入数据范围为[-10^7, 10^7]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。
- <term>Atlas 推理系列产品</term>：FLOAT、FLOAT16数据类型的输入数据范围为[-65504, 65504]时满足精度要求，超过数值范围无法保证，请使用CPU进行计算。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sin](./examples/test_aclnn_sin.cpp) | 通过[aclnnSin](./docs/aclnnSin&aclnnInplaceSin.md)接口方式调用Sin算子。 |
