# aclnnFusedMulAddN

> **接口形态说明**：本算子**同时提供 aclnn 两段式接口与图模式（GE IR）入口**。aclnn 两段式包装层
> （`op_host/op_api/`）便于走标准 PyTorch + ACLNN 精度验收路径。本文档为该 aclnn 两段式接口的接口文档。

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 本实验态实现仅适配 `ascend910b`（Atlas A2 训练/推理系列产品，DAV_2201）。内建算子
> `math/fused_mul_add_n` 另行覆盖 `ascend950`（arch35），不在本实验态范围内。

## 功能说明

- 算子功能：逐元素融合的标量乘加（融合 `mul` 与 `addn(n=2)`，要求 addn 的 n 为 2，mul 的其中一个
  乘数必须是 scalar 或仅含一个数的 tensor）。x3 为单元素标量张量（ShapeSize = 1），仅其首元素
  x3[0] 参与计算，按标量广播到 x1 的全部元素。

- 计算公式：

  $$
  y_i = x1_i \times x3[0] + x2_i
  $$

  其中 x1、x2、y 形状相同（逐元素，非矩阵乘）；x3 为单元素标量张量，仅取 x3[0] 作为标量乘数。

- 接口形态：标准 aclnn **两段式**设计（声明见 `op_host/op_api/aclnn_fused_mul_add_n.h`，由自定义算子包
  `custom_math` 导出至 `libcust_opapi.so`）：
  - 第一段 `aclnnFusedMulAddNGetWorkspaceSize`：完成参数校验、Contiguous、调度 L0
    `l0op::FusedMulAddN`（InferShape + 调度到已注册的 `FusedMulAddN` aicore kernel）、ViewCopy，返回所需
    workspace 大小与 op 执行器；
  - 第二段 `aclnnFusedMulAddN`：依据第一段返回的 workspace/executor 在指定 stream 上执行计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnFusedMulAddNGetWorkspaceSize` 接口获取入参并计算
本次 API 调用计算所需要的 workspace 大小，再调用 `aclnnFusedMulAddN` 接口执行计算。

```cpp
aclnnStatus aclnnFusedMulAddNGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* x3,
    aclTensor*       y,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);

aclnnStatus aclnnFusedMulAddN(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream);
```

## aclnnFusedMulAddNGetWorkspaceSize

- **参数说明**：

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 90px">
  <col style="width: 90px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>device 侧 aclTensor，主张量，被标量 x3[0] 乘。对应公式中 x1。dtype 需与 x2/x3/y 一致，shape 需与 x2/y 一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>ND</td>
      <td>0~8 维</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>device 侧 aclTensor，逐元素加到 x1 × x3[0] 上。对应公式中 x2。dtype、shape 需与 x1 一致。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
      <td>与 x1 一致</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>输入</td>
      <td>device 侧 aclTensor，单元素标量张量（ShapeSize = 1），仅取 x3[0] 作为标量乘数。对应公式中 x3[0]。dtype 需与 x1 一致。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
      <td>ShapeSize = 1（[1]/[1,1] 等价）</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>device 侧 aclTensor，计算结果，与 x1 同 dtype、同 shape。对应公式中 y。</td>
      <td>与 x1 保持一致</td>
      <td>ND</td>
      <td>与 x1 一致</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在 npu device 侧申请的 workspace 大小。</td>
      <td>uint64_t*</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回 op 执行器，包含算子计算流程。</td>
      <td>aclOpExecutor**</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：x1/x2/x3/y 数据类型支持 FLOAT、FLOAT16、
  BFLOAT16、INT32、INT16，且四者 dtype 必须完全一致。

- **返回值**：

  返回 `aclnnStatus` 状态码，具体参见 [aclnn 返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 110px">
  <col style="width: 540px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的 x1、x2、x3 或 y 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x1 的数据类型不在支持范围（FLOAT/FLOAT16/BFLOAT16/INT32/INT16）之内。</td>
    </tr>
    <tr>
      <td>x2、x3 或 y 的数据类型与 x1 不一致。</td>
    </tr>
    <tr>
      <td>x2 或 y 的 shape 与 x1 不一致；或 x1/x2/x3/y 维度数超过 8。</td>
    </tr>
    <tr>
      <td>x3 不是单元素标量张量（ShapeSize != 1）；或输入为不支持的私有格式（仅支持 ND）。</td>
    </tr>
  </tbody></table>

## aclnnFusedMulAddN

- **参数说明**：

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 480px">
  <col style="width: 270px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 npu device 侧申请的 workspace 内存起址。</td>
      <td>void*</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 npu device 侧申请的 workspace 大小，由第一段接口 aclnnFusedMulAddNGetWorkspaceSize 获取。</td>
      <td>uint64_t</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
      <td>aclOpExecutor*</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 acl stream 流。</td>
      <td>aclrtStream</td>
    </tr>
  </tbody></table>

- **返回值**：

  返回 `aclnnStatus` 状态码，具体参见 [aclnn 返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- x1、x2、x3、y 的数据类型必须完全一致，且属于 {FLOAT, FLOAT16, BFLOAT16, INT32, INT16}（host tiling
  与 aclnn 第一段接口双层强校验，不一致直接报错）。
- x1、x2、y 的形状必须一致（逐元素，非矩阵乘、无广播）。
- x3 必须为单元素标量张量（ShapeSize = 1），形态 `[1]` 与 `[1,1]` 等价（仅取 x3[0]）；否则报错。
- x1、x2、x3、y 维度数（rank）均不超过 8；仅支持 ND 格式，不支持私有格式。
- 本算子无 attribute（无属性参数）。
- 空 Tensor（x1/x2/y 任一为空）：第一段接口校验通过后直接置 `workspaceSize = 0` 返回，第二段无计算。
- FLOAT16 / BFLOAT16 在 kernel 内 cast 到 FLOAT 计算再 cast 回原 dtype，规避半精度中间累加精度损失；
  INT32 / INT16 直算，结果按整数语义（溢出按目标整型回绕，不做饱和）。
- 确定性说明：逐元素 FMA，不含 Reduce / 矩阵累加，默认确定性实现（相同输入恒产生相同输出）。

## 调用示例

完整的端到端 aclnn 两段式调用示例见 [examples/test_aclnn_fused_mul_add_n.cpp](../examples/test_aclnn_fused_mul_add_n.cpp)
（真实 NPU / ascend910b）。编译运行方式见 [examples/run.sh](../examples/run.sh) 与算子 README「编译运行」章节
（aclnn 示例依赖自定义算子包 `custom_math`，需先 `build.sh --pkg --experimental --soc=ascend910b --ops=fused_mul_add_n`
构建并解包，再 `bash run.sh aclnn`）。核心调用片段如下：

```cpp
#include "acl/acl.h"
#include "aclnn_fused_mul_add_n.h"

// ...（省略 device/stream 初始化与 aclTensor 构造，详见示例完整代码）
// 本例取 x1 = {1,2,3,4,5,6,7,8}、x2 全 1、x3[0] = 2（shape: x1/x2/y = [2,4], x3 = [1]）
// 期望 y_i = x1_i * x3[0] + x2_i = x1_i * 2 + 1 -> {3,5,7,9,11,13,15,17}

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;

// 第一段接口：参数校验 + 调度 L0 + 计算 workspace 大小
auto ret = aclnnFusedMulAddNGetWorkspaceSize(x1, x2, x3, y, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS,
          LOG_PRINT("aclnnFusedMulAddNGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

// 根据第一段接口计算出的 workspaceSize 申请 device 内存（workspaceSize 可能为 0，如空 tensor）
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
}

// 第二段接口：在 stream 上执行计算
ret = aclnnFusedMulAddN(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMulAddN failed. ERROR: %d\n", ret); return ret);

// 同步等待任务执行结束
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS,
          LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

// ...（拷回 y、校验结果、释放 aclTensor / device 资源，详见示例完整代码）
```

PyTorch（TorchNPU）侧的两段式封装见 [tests/st/torch/torch_adapter.cpp](../tests/st/torch/torch_adapter.cpp)：
在 `forward_npu` 中分配输出 tensor 与 workspace tensor，调用 `aclnnFusedMulAddNGetWorkspaceSize` 取
workspace 大小与 executor，再经 `OpCommand::RunOpApiV2` 异步入队执行 `aclnnFusedMulAddN`，workspace 内存
由 `torch::empty` 管理（不使用 `aclrtMalloc`）。

## 其它入口（图模式）

除 aclnn 两段式接口外，本算子还提供图模式（GE IR）入口，可通过算子原型
`REG_OP(FusedMulAddN)` 生成的 `op::FusedMulAddN` 构图调用，示例见
[examples/test_geir_fused_mul_add_n.cpp](../examples/test_geir_fused_mul_add_n.cpp)。
