# aclnnIm2col

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id2 -->

## 功能说明

- 接口功能：将三维或四维NCHW输入中的滑动窗口展开为列矩阵，常用于卷积计算前的数据重排。
- 计算公式：

  $$
  outD = \left\lfloor\frac{inD + 2 \times paddingD - dilationD \times (kernelD - 1) - 1}{strideD}\right\rfloor + 1
  $$

  三维输入$[C,H,W]$的输出shape为$[C \times kernelH \times kernelW, outH \times outW]$；
  四维输入$[N,C,H,W]$的输出shape为
  $[N,C \times kernelH \times kernelW, outH \times outW]$。

## 函数原型

本接口采用两段式调用。必须先调用`aclnnIm2colGetWorkspaceSize`获取workspace大小和执行器，
再调用`aclnnIm2col`执行计算。

```cpp
aclnnStatus aclnnIm2colGetWorkspaceSize(
    const aclTensor   *self,
    const aclIntArray *kernelSize,
    const aclIntArray *dilation,
    const aclIntArray *padding,
    const aclIntArray *stride,
    const aclTensor   *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnIm2col(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnIm2colGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px"><col style="width: 100px"><col style="width: 270px"><col style="width: 360px">
  <col style="width: 180px"><col style="width: 100px"><col style="width: 150px"><col style="width: 100px">
  </colgroup><thead><tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th>
  <th>维度(shape)</th><th>非连续Tensor</th>
  </tr></thead><tbody>
  <tr><td>self（const aclTensor*）</td><td>输入</td><td>待展开的输入Tensor。</td>
  <td>三维时按CHW解释，四维时按NCHW解释。三维输入不支持空Tensor；四维输入仅支持N为0的空Tensor。</td>
  <td>FLOAT16、FLOAT、BFLOAT16、BOOL</td><td>ND</td><td>3或4</td><td>√</td></tr>
  <tr><td>kernelSize（const aclIntArray*）</td><td>输入</td><td>卷积核大小[kernelH, kernelW]。</td>
  <td>长度为2，所有元素必须大于0。</td><td>INT64</td><td>-</td><td>2</td><td>-</td></tr>
  <tr><td>dilation（const aclIntArray*）</td><td>输入</td><td>卷积核膨胀系数[dilationH, dilationW]。</td>
  <td>长度为2，所有元素必须大于0。</td><td>INT64</td><td>-</td><td>2</td><td>-</td></tr>
  <tr><td>padding（const aclIntArray*）</td><td>输入</td><td>H、W方向两侧的对称填充值[paddingH, paddingW]。</td>
  <td>长度为2，所有元素必须大于等于0。</td><td>INT64</td><td>-</td><td>2</td><td>-</td></tr>
  <tr><td>stride（const aclIntArray*）</td><td>输入</td><td>H、W方向的滑动步长[strideH, strideW]。</td>
  <td>长度为2，所有元素必须大于0。</td><td>INT64</td><td>-</td><td>2</td><td>-</td></tr>
  <tr><td>out（const aclTensor*）</td><td>输出</td><td>滑动窗口展开后的输出Tensor。</td>
  <td>数据类型必须与self相同，shape必须与公式推导结果相同。</td><td>与self一致</td><td>ND</td>
  <td>self为三维时是2维；self为四维时是3维</td><td>√</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回Device侧workspace大小。</td>
  <td>不可为空指针。</td><td>UINT64</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回包含算子计算流程的执行器。</td>
  <td>不可为空指针。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

```text
第一段接口完成入参校验，出现以下场景时报错：
161001（ACLNN_ERR_PARAM_NULLPTR）：self、kernelSize、dilation、padding、stride、out、workspaceSize或executor为空指针。
161002（ACLNN_ERR_PARAM_INVALID）：self或out的数据类型、数据格式、维度或shape不满足要求；属性长度或取值不满足
                                 要求；计算得到的输出shape无效或发生溢出。
```

## aclnnIm2col

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 |
  | --- | --- | --- | --- |
  | workspace（void*） | 输入 | Device侧workspace内存地址。 | 由第一段接口返回的workspaceSize申请；workspaceSize为0时可传空指针。 |
  | workspaceSize（uint64_t） | 输入 | Device侧workspace大小。 | 由`aclnnIm2colGetWorkspaceSize`获取。 |
  | executor（aclOpExecutor*） | 输入 | 包含算子计算流程的执行器。 | 由`aclnnIm2colGetWorkspaceSize`获取。 |
  | stream（aclrtStream） | 输入 | 指定执行任务的Stream。 | 不可为空指针。 |

- **返回值**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `self`的C、H、W必须大于0；四维输入的N必须大于等于0。
- 计算得到的`outH`、`outW`必须大于0，且所有输入、输出shape乘积均不得发生INT64溢出。
- 支持非连续输入与输出Tensor，接口内部完成连续化和结果回写。
- 确定性计算：`aclnnIm2col`默认确定性实现。

## 调用示例

可直接编译运行的示例见
[test_aclnn_im2col.cpp](../examples/test_aclnn_im2col.cpp)，编译和执行方式请参考
[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。示例覆盖三维输入、四维输入以及
FLOAT、FLOAT16、BFLOAT16、BOOL数据类型，并完整展示两段式接口调用和资源释放流程。
