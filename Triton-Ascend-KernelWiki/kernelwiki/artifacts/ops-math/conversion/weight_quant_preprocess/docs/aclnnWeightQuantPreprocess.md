# aclnnWeightQuantPreprocess

<!-- codespell:ignore outWeight -->

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/weight_quant_preprocess)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

完成伪量化Matmul（包括QuantBatchMatmulV5、GroupedMatmul-伪量化）的参数预处理：主要将weight从ND格式转换为FRACTAL_NZ格式，并在需要时对weightScale、weightOffsetOptional、biasOptional进行同步处理。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)：

1. 调用`aclnnWeightQuantPreprocessGetWorkspaceSize`获取workspace大小及执行器；
2. 调用`aclnnWeightQuantPreprocess`执行计算。

**注意**：用户需自行构造输出张量，参考约束说明中的shape计算公式。

```c++
aclnnStatus aclnnWeightQuantPreprocessGetWorkspaceSize(
    const aclTensor *weight,
    const aclTensor *weightScale,
    const aclTensor *weightOffsetOptional,
    const aclTensor *biasOptional,
    aclDataType      xDtype,
    aclDataType      xScaleDtype,
    int64_t          kGroupSize,
    aclTensor       *outWeight,
    aclTensor       *outWeightScale,
    aclTensor       *outWeightOffsetOptional,
    aclTensor       *outBiasOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)

aclnnStatus aclnnWeightQuantPreprocess(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnWeightQuantPreprocessGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1195px"><colgroup>
  <col style="width: 220px">
  <col style="width: 90px">
  <col style="width: 160px">
  <col style="width: 270px">
  <col style="width: 180px">
  <col style="width: 95px">
  <col style="width: 120px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度（shape）</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="tg-0pky">weight(aclTensor *)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Matmul的权重矩阵</td>
      <td class="tg-0pky">不支持空tensor</td>
      <td class="tg-0pky">float4_e2m1</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2-3</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightScale(aclTensor *)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">权重的反量化scale参数</td>
      <td class="tg-0pky">不支持空tensor</td>
      <td class="tg-0pky">float8_e8m0</td>
      <td class="tg-0pky">ND/NCL/NCHW</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOffsetOptional(aclTensor *)</td>
      <td class="tg-0pky">可选输入</td>
      <td class="tg-0pky">权重的反量化offset参数</td>
      <td class="tg-0pky">当前MM_MX_A8W4/GMM_MX_A8W4数据流不支持，必须为nullptr</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">biasOptional(aclTensor *)</td>
      <td class="tg-0pky">可选输入</td>
      <td class="tg-0pky">Matmul的偏置矩阵</td>
      <td class="tg-0pky">不支持空tensor，必须contiguous</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">不支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">xDtype(aclDataType)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">Matmul的激活矩阵的数据类型</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclDataType</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">xScaleDtype(aclDataType)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">激活的量化scale参数的数据类型</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclDataType</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">kGroupSize(int64_t)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">权重在per-group量化时K维度的group的大小</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">int64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeight(aclTensor *)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">预处理后的weight</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">int8/int4/fp8_e4m3/hif8/fp4_e2m1</td>
      <td class="tg-0pky">NZ</td>
      <td class="tg-0pky">2-5</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeightScale(aclTensor *)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">预处理后的weightScale</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">float16/bfloat16/fp8_e8m0</td>
      <td class="tg-0pky">ND/NCL/NCHW</td>
      <td class="tg-0pky">3-4</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">outWeightOffsetOptional(aclTensor *)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">预处理后的weightOffset</td>
      <td class="tg-0pky">当前MM_MX_A8W4/GMM_MX_A8W4数据流不支持，必须为nullptr</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">仅转置场景支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">outBiasOptional(aclTensor *)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">预处理后的bias</td>
      <td class="tg-0pky">必须contiguous</td>
      <td class="tg-0pky">float16/bfloat16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-2</td>
      <td class="tg-0pky">不支持</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize(uint64_t *)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">计算所需的workspace大小</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">uint64*</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor(aclOpExecutor **)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">包含算子计算流程的执行器</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">aclOpExecutor**</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">weight、weightScale、outWeight或outWeightScale是空指针；或biasOptional非空但outBiasOptional是空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="6">161002</td>
      <td class="tg-0pky">输入的数据类型组合不支持，无法匹配当前支持的MM_MX_A8W4/GMM_MX_A8W4数据流。</td>
    </tr>
    <tr>
      <td class="tg-0lax">weight、weightScale、outWeight或outWeightScale是空tensor；或biasOptional/outBiasOptional在提供时为空tensor。</td>
    </tr>
    <tr>
      <td class="tg-0lax">weight、weightScale、biasOptional、outWeight、outWeightScale或outBiasOptional的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td class="tg-0lax">weight、weightScale、biasOptional、outWeight、outWeightScale或outBiasOptional的shape或storage shape不满足校验条件。</td>
    </tr>
    <tr>
      <td class="tg-0lax">weight或weightScale的stride不满足转置要求，或biasOptional/outBiasOptional在提供时不连续。</td>
    </tr>
    <tr>
      <td class="tg-0lax">weightOffsetOptional或outWeightOffsetOptional非空，或kGroupSize不等于32。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_RUNTIME_ERROR</td>
      <td class="tg-0pky">361001</td>
      <td class="tg-0pky">产品型号不支持。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td class="tg-0pky">561101</td>
      <td class="tg-0pky">内部错误，执行器创建失败。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_INNER_NULLPTR</td>
      <td class="tg-0pky">561103</td>
      <td class="tg-0pky">workspaceSize或executor是空指针，或API内部构图接口返回空指针。</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantPreprocess

- **参数说明**

  | 参数名        | 输入/输出 | 描述                                                         |
  | ------------- | --------- | ------------------------------------------------------------ |
  | workspace     | 输入      | 在Device侧申请的workspace内存地址。                          |
  | workspaceSize | 输入      | 在Device侧申请的workspace大小，由第一段接口aclnnWeightQuantPreprocessGetWorkspaceSize获取。 |
  | executor      | 输入      | op执行器，包含了算子计算流程。                               |
  | stream        | 输入      | 指定执行任务的Stream。                                       |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnWeightQuantPreprocess默认确定性实现。

<details>
<summary><strong>MM_MX_A8W4数据流</strong>（MM表示Matmul；MX_A8W4表示x的数据类型为FLOAT8_E4M3FN，weight的数据类型为FLOAT4_E2M1，Mx量化模式）</summary>

- **weight**
  - 数据类型：FLOAT4_E2M1
  - 格式：ND
  - K % kGroupSize == 0
  - view shape：2-D `{K, N}`
  - storage shape：`{N, K}`（transposed）
  - stride：`[1, K]`（最后两维transposed）
  - 不支持空tensor

- **weightScale**
  - 数据类型：FLOAT8_E8M0
  - 格式：ND/NCL
  - view shape：3-D `{ceildiv(K, 64), N, 2}`
  - storage shape：`{N, ceildiv(K, 64), 2}`（transposed）
  - stride：`[2, 2*ceildiv(K,64), 1]`（维度0和1交换）
  - 不支持空tensor

- **weightOffsetOptional**
  - 当前不支持，必须为nullptr
  - outWeightOffsetOptional也必须为nullptr

- **biasOptional**
  - 数据类型：float16/bfloat16
  - 格式：ND
  - 必须为contiguous
  - 不支持空tensor（若提供）

- **kGroupSize**
  - 必须等于 32

- **xDtype**
  - FLOAT8_E4M3FN

- **xScaleDtype**
  - FLOAT8_E8M0

- **outWeight**
  - 数据类型：与weight相同
  - 格式：FRACTAL_NZ_C0_32
  - view shape：与weight view shape相同 `{K, N}`
  - storage shape：4-D `{ceildiv(K, 32), ceildiv(N, 16), 16, 32}`

- **outWeightScale**
  - 数据类型：与weightScale相同
  - 格式：ND
  - view shape：与weightScale view shape相同
  - storage shape：与weightScale storage shape相同

- **outBiasOptional**
  - 数据类型：与biasOptional相同
  - 格式：ND
  - 必须为contiguous
  - view shape：与biasOptional相同
  - storage shape：与biasOptional相同

</details>

<details>
<summary><strong>GMM_MX_A8W4数据流</strong>（GMM表示GroupedMatmul；MX_A8W4表示x的数据类型为FLOAT8_E4M3FN，weight的数据类型为FLOAT4_E2M1，Mx量化模式）</summary>

- **weight**
  - 数据类型：FLOAT4_E2M1
  - 格式：ND
  - K % kGroupSize == 0
  - view shape：3-D `{G, K, N}`
  - storage shape：`{G, N, K}`（transposed，最后两维交换）
  - stride：`[K*N, 1, K]`（维度1和2 transposed）
  - 不支持空tensor

- **weightScale**
  - 数据类型：FLOAT8_E8M0
  - 格式：ND/NCL/NCHW
  - view shape：4-D `{G, ceildiv(K, 64), N, 2}`
  - storage shape：`{G, N, ceildiv(K, 64), 2}`（transposed，维度2和3交换）
  - stride：`[2*ceildiv(K,64)*N, 2, 2*ceildiv(K,64), 1]`（维度2和3交换）
  - 不支持空tensor

- **weightOffsetOptional**
  - 当前不支持，必须为nullptr
  - outWeightOffsetOptional也必须为nullptr

- **biasOptional**
  - 数据类型：float16/bfloat16
  - 格式：ND
  - 必须为contiguous
  - 不支持空tensor（若提供）

- **kGroupSize**
  - 必须等于 32

- **xDtype**
  - FLOAT8_E4M3FN

- **xScaleDtype**
  - FLOAT8_E8M0

- **outWeight**
  - 数据类型：与weight相同
  - 格式：FRACTAL_NZ_C0_32
  - view shape：与weight view shape相同 `{G, K, N}`
  - storage shape：5-D `{G, ceildiv(K, 32), ceildiv(N, 16), 16, 32}`

- **outWeightScale**
  - 数据类型：与weightScale相同
  - 格式：ND
  - view shape：与weightScale view shape相同
  - storage shape：与weightScale storage shape相同

- **outBiasOptional**
  - 数据类型：与biasOptional相同
  - 格式：ND
  - 必须为contiguous
  - view shape：与biasOptional相同
  - storage shape：与biasOptional相同

</details>

- 其余数据类型与shape组合为预留接口，当前调用将返回ACLNN_ERR_PARAM_INVALID

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

**注意**：用户需自行计算并构造输出张量shape，参考约束说明中的公式：

- outWeight viewShape：与weight viewShape相同
- outWeight storageShape：`{CeilDiv(K, 32), CeilDiv(N, 16), 16, 32}`
- outWeight format：`ACL_FORMAT_FRACTAL_NZ_C0_32`

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_weight_quant_preprocess.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape)
        size *= d;
    return size;
}

class AclRuntimeGuard {
public:
    explicit AclRuntimeGuard(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclRuntimeGuard()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init(aclrtStream* stream)
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        stream_ = *stream;
        return ACL_SUCCESS;
    }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    AclRuntimeGuard aclGuard(deviceId);
    auto ret = aclGuard.Init(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Init failed" << std::endl; return ret);

    // weight: FLOAT4_E2M1, transposed (MM_MX_A8W4)
    int64_t k = 64;
    int64_t n = 128;
    int64_t C0 = 32; // FLOAT4_E2M1 对应 C0=32

    std::vector<int64_t> weightViewShape = {k, n};
    std::vector<int64_t> weightStorageShape = {n, k};
    std::vector<int64_t> weightStrides = {1, k};
    int64_t weightStorageSize = GetShapeSize(weightStorageShape);
    int64_t weightBytes = weightStorageSize / 2; // FP4: 4 bits = 0.5 bytes per element

    std::vector<int8_t> weightHostData(weightBytes, 0);
    void* weightDeviceAddr = nullptr;
    ret = aclrtMalloc(&weightDeviceAddr, weightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(weightDeviceAddr, weightBytes, weightHostData.data(), weightBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weight failed" << std::endl; return ret);
    aclTensor* weight = aclCreateTensor(weightViewShape.data(), weightViewShape.size(), ACL_FLOAT4_E2M1,
                                        weightStrides.data(), 0, ACL_FORMAT_ND, weightStorageShape.data(),
                                        weightStorageShape.size(), weightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPtr(weight, aclDestroyTensor);
    CHECK_RET(weight != nullptr, std::cout << "Create weight tensor failed" << std::endl; return ACL_ERROR_FAILURE);

    // weightScale: FLOAT8_E8M0, 3-D transposed (MM_MX_A8W4)
    // viewShape: {ceildiv(K,64), N, 2} = {1, 128, 2}
    // storageShape: {N, ceildiv(K,64), 2} = {128, 1, 2}
    // transposed stride: {2, 2, 1} (dim0 <-> dim1)
    std::vector<int64_t> scaleViewShape = {k / 64, n, 2};
    std::vector<int64_t> scaleStorageShape = {n, k / 64, 2};
    std::vector<int64_t> scaleStrides = {2, 2, 1};
    int64_t scaleStorageSize = GetShapeSize(scaleStorageShape);
    int64_t scaleBytes = scaleStorageSize; // FP8: 1 byte per element

    std::vector<int8_t> scaleHostData(scaleBytes, 0);
    void* scaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&scaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc weightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    ret = aclrtMemcpy(scaleDeviceAddr, scaleBytes, scaleHostData.data(), scaleBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Memcpy weightScale failed" << std::endl; return ret);
    aclTensor* weightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                             scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                             scaleStorageShape.size(), scaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightScalePtr(weightScale, aclDestroyTensor);
    CHECK_RET(weightScale != nullptr, std::cout << "Create weightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // 用户自行构造 outWeight (FRACTAL_NZ_C0_32)
    // viewShape 与 weight viewShape 相同，storageShape 按公式计算
    std::vector<int64_t> outWeightViewShape = {k, n};
    std::vector<int64_t> outWeightStorageShape = {CEIL_DIV(k, C0), CEIL_DIV(n, 16), 16, C0};
    int64_t outWeightStorageSize = GetShapeSize(outWeightStorageShape);
    int64_t outWeightBytes = outWeightStorageSize / 2; // FP4

    void* outWeightDeviceAddr = nullptr;
    ret = aclrtMalloc(&outWeightDeviceAddr, outWeightBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeight failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outWeightDeviceAddrPtr(outWeightDeviceAddr, aclrtFree);
    aclTensor* outWeight = aclCreateTensor(outWeightViewShape.data(), outWeightViewShape.size(), ACL_FLOAT4_E2M1,
                                           nullptr, 0, ACL_FORMAT_FRACTAL_NZ_C0_32, outWeightStorageShape.data(),
                                           outWeightStorageShape.size(), outWeightDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightPtr(outWeight, aclDestroyTensor);
    CHECK_RET(outWeight != nullptr, std::cout << "Create outWeight tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    // 构造 outWeightScale (viewShape 和 storageShape 都与 weightScale 相同)
    // 根据实现要求：outWeightScale 的 viewShape 和 storageShape 必须都与 weightScale 相同
    void* outScaleDeviceAddr = nullptr;
    ret = aclrtMalloc(&outScaleDeviceAddr, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc outWeightScale failed" << std::endl; return ret);
    std::unique_ptr<void, aclError (*)(void*)> outScaleDeviceAddrPtr(outScaleDeviceAddr, aclrtFree);
    aclTensor* outWeightScale = aclCreateTensor(scaleViewShape.data(), scaleViewShape.size(), ACL_FLOAT8_E8M0,
                                                scaleStrides.data(), 0, ACL_FORMAT_ND, scaleStorageShape.data(),
                                                scaleStorageShape.size(), outScaleDeviceAddr);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outWeightScalePtr(outWeightScale, aclDestroyTensor);
    CHECK_RET(outWeightScale != nullptr, std::cout << "Create outWeightScale tensor failed" << std::endl;
              return ACL_ERROR_FAILURE);

    aclDataType xDtype = ACL_FLOAT8_E4M3FN;
    aclDataType xScaleDtype = ACL_FLOAT8_E8M0;
    int64_t kGroupSize = 32;

    // 1. 获取 workspace 与执行器
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnWeightQuantPreprocessGetWorkspaceSize(
        weight, weightScale, nullptr, nullptr, // weightOffsetOptional, biasOptional
        xDtype, xScaleDtype, kGroupSize, outWeight, outWeightScale, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "GetWorkspaceSize failed" << std::endl; return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::cout << "Malloc workspace failed" << std::endl; return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 2. 执行计算
    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Preprocess failed" << std::endl; return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::cout << "Synchronize failed" << std::endl; return ret);

    // 3. 释放资源
    workspaceAddrPtr.reset();
    outWeightScalePtr.reset();
    outWeightPtr.reset();
    weightScalePtr.reset();
    weightPtr.reset();
    outScaleDeviceAddrPtr.reset();
    outWeightDeviceAddrPtr.reset();
    scaleDeviceAddrPtr.reset();
    weightDeviceAddrPtr.reset();
    return 0;
}

```
