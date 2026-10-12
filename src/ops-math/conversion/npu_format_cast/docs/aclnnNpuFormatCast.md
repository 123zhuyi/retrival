# aclnnNpuFormatCast

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/npu_format_cast)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- **接口功能**：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 完成ND[数据格式](../../../docs/zh/context/data_format.md)到指定C0大小的FRACTAL_NZ[数据格式](../../../docs/zh/context/data_format.md)的转换功能，C0是FRACTAL_NZ[数据格式](../../../docs/zh/context/data_format.md)最后一维的大小，C0由`additionalDtype`确定。
    - 完成指定C0大小的FRACTAL_NZ[数据格式](../../../docs/zh/context/data_format.md)到ND[数据格式](../../../docs/zh/context/data_format.md)的转换功能，其中支持的NZ格式包括：FRACTAL_NZ、FRACTAL_NZ_C0_2、FRACTAL_NZ_C0_4、FRACTAL_NZ_C0_16、FRACTAL_NZ_C0_32。
    - 当srcTensor数据类型为FLOAT8_E8M0、srcTensor格式为ND/NCL/NCHW、dstFormat为FRACTAL_NZ时，支持MX scale转换为昇腾亲和存储shape。
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 完成ND←→[NZ](../../../docs/zh/context/data_format.md)、NCL←→[NZ](../../../docs/zh/context/data_format.md)、NCHW←→[NZ](../../../docs/zh/context/data_format.md)、NCDHW←→[NZ](../../../docs/zh/context/data_format.md)的转换功能。C0是[NZ](../../../docs/zh/context/data_format.md)数据格式最后一维的大小。计算方法C0 = 32B / ge::GetSizeByDataType(static_cast additionalDtype)。
    - 完成NCDHW←→[NDC1HWC0](../../../docs/zh/context/data_format.md)、NCDHW←→[FRACTAL_Z_3D](../../../docs/zh/context/data_format.md)、NCHW←→[NC1HWC0](../../../docs/zh/context/data_format.md)、NHWC←→[NC1HWC0](../../../docs/zh/context/data_format.md)、NCHW←→[FRACTAL_Z](../../../docs/zh/context/data_format.md)、HWCN←→[FRACTAL_Z](../../../docs/zh/context/data_format.md)、NDHWC←→[NDC1HWC0](../../../docs/zh/context/data_format.md)、DHWCN←→[FRACTAL_Z_3D](../../../docs/zh/context/data_format.md)的转换功能。其中，C0与微架构强相关，该值等于cube单元的size，例如16。C1是将C维度按照C0切分：C1=C/C0，若结果不整除，最后一份数据需要padding到C0。计算方法C0 = 32B / ge::GetSizeByDataType(static_cast additionalDtype)（例如FP16的additionalDtype枚举值为1，对应的数据FP16为2byte）。
  <!-- end id8 -->

- **计算流程**：

  `aclnnNpuFormatCastCalculateSizeAndFormat`根据输入张量srcTensor、数据类型`additionalDtype`和目标张量的数据格式dstFormat计算出转换后目标张量dstTensor的shape和实际数据格式，用于构造dstTensor，然后调用`aclnnNpuFormatCast`把srcTensor转换为实际数据格式的目标张量dstTensor。

## 函数原型

必须先调用`aclnnNpuFormatCastCalculateSizeAndFormat`计算出dstTensor的shape和实际数据格式，再调用[两段式接口](../../../docs/zh/context/two_phase_api.md)。两段式接口先调用`aclnnNpuFormatCastGetWorkSpaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnNpuFormatCast`接口执行计算。

```c++
aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(
    const aclTensor* srcTensor,
    const int        dstFormat,
    const int        additionalDtype,
    int64_t**        dstShape,
    uint64_t*        dstShapeSize,
    int*             actualFormat)
```

```c++
aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(
    const aclTensor* srcTensor,
    aclTensor*       dstTensor,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```c++
aclnnStatus aclnnNpuFormatCast(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnNpuFormatCastCalculateSizeAndFormat

- **参数说明**

  <table style="table-layout: fixed; width: 1665px;">
    <colgroup>
        <col style="width: 180px">
        <col style="width: 120px">
        <col style="width: 200px">
        <col style="width: 310px">
        <col style="width: 290px">
        <col style="width: 300px">
        <col style="width: 130px">
        <col style="width: 145px">
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
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor（aclTensor*）</td>
            <td>输入</td>
            <td>转换的源Tensor。</td>
            <td>-</td>
            <td>INT8（2）、UINT8（4）、INT32（3）、UINT32（8）、FLOAT（0）、FLOAT16（1）、BFLOAT16（27）、FLOAT8_E4M3FN<sup>2</sup>（36）、FLOAT4_E2M1<sup>2</sup>（40）、FLOAT4_E1M2<sup>2</sup>（41）、HIFLOAT8<sup>2</sup>（34）</td>
            <td>ACL_FORMAT_ND（2）、ACL_FORMAT_NZ（29）、ACL_FORMAT_NCL（47）、ACL_FORMAT_NCHW<sup>1</sup>（0）、ACL_FORMAT_NHWC<sup>1</sup>（1）、ACL_FORMAT_HWCN<sup>1</sup>（16）、ACL_FORMAT_NCDHW<sup>1</sup>（30）、ACL_FORMAT_NDHWC<sup>1</sup>（27）、ACL_FORMAT_DHWCN<sup>1</sup>（31）、ACL_FORMAT_NDC1HWC0<sup>1</sup>（32）、ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup>（33）、ACL_FORMAT_NC1HWC0<sup>1</sup>（3）、ACL_FORMAT_FRACTAL_Z<sup>1</sup>（4）。</td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstFormat（int）</td>
            <td>输入</td>
            <td>输出张量的数据格式对应的枚举值，使用说明中括号内容为对应的枚举值，实际输入为对应枚举值。</td>
            <td>ACL_FORMAT_ND（2）、ACL_FORMAT_NZ（29）、ACL_FORMAT_NCL（47）、ACL_FORMAT_NCHW<sup>1</sup>（0）、ACL_FORMAT_NHWC<sup>1</sup>（1）、ACL_FORMAT_HWCN<sup>1</sup>（16）、ACL_FORMAT_NCDHW<sup>1</sup>（30）、ACL_FORMAT_NDHWC<sup>1</sup>（27）、ACL_FORMAT_DHWCN<sup>1</sup>（31）、ACL_FORMAT_NDC1HWC0<sup>1</sup>（32）、ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup>（33）、ACL_FORMAT_NC1HWC0<sup>1</sup>（3）、ACL_FORMAT_FRACTAL_Z<sup>1</sup>（4）。</td>
            <td>INT64</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>additionalDtype（int）</td>
            <td>输入</td>
            <td>转换为私有数据格式时，推断C0大小所使用的基本数据类型，具体数据类型对应的枚举值，使用说明中括号内容为对应的枚举值，实际输入为对应枚举值。</td>
            <td>如果传-1，additionalDtype默认使用srcTensor的数据类型。</td>
            <td>INT8（2）、UINT8（4）、INT32（3）、UINT32（8）、FLOAT（0）、FLOAT16（1）、BFLOAT16（27）、FLOAT8_E4M3FN<sup>2</sup>（36）、FLOAT4_E2M1<sup>2</sup>（40）、HIFLOAT8<sup>2</sup>（34）、-1</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShape（int64_t**）</td>
            <td>输出</td>
            <td>用于输出dstTensor的shape数组的指针。该指针指向的内存由本接口申请，调用者释放。</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstShapeSize（uint64_t*）</td>
            <td>输出</td>
            <td>用于输出dstTensor的shape数组大小的指针。</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>actualFormat（int*）</td>
            <td>输出</td>
            <td>用于输出dstTensor实际数据格式的指针。</td>
            <td>括号中对应其对应的枚举值。</td>
            <td>None</td>
            <td>ACL_FORMAT_ND（2）、ACL_FORMAT_NZ（29）、ACL_FORMAT_NCL（47）、ACL_FORMAT_NCHW<sup>1</sup>（0）、ACL_FORMAT_NHWC<sup>1</sup>（1）、ACL_FORMAT_HWCN<sup>1</sup>（16）、ACL_FORMAT_NCDHW<sup>1</sup>（30）、ACL_FORMAT_NDHWC<sup>1</sup>（27）、ACL_FORMAT_DHWCN<sup>1</sup>（31）、ACL_FORMAT_NDC1HWC0<sup>1</sup>（32）、ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup>（33）、ACL_FORMAT_NC1HWC0<sup>1</sup>（3）、ACL_FORMAT_FRACTAL_Z<sup>1</sup>（4）。</td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：上表数据类型列中的角标“1”代表该系列不支持的数据类型或数据格式。
  <!-- end id9 -->
  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：上表数据类型列中的角标“2”代表该系列不支持的数据类型或数据格式。
  <!-- end id10 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table>
    <thead>
      <tr>
        <th style="width: 291px">返回值</th>
        <th style="width: 135px">错误码</th>
        <th style="width: 724px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>传入的srcTensor是空指针。</td>
      </tr>
      <tr>
        <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="5"> 161002 </td>
        <td>srcTensor的数据格式不在具体范围内。</td>
      </tr>
      <tr>
        <td>dstFormat的数据格式不在具体范围内。</td>
      </tr>
      <tr>
        <td>additionalDtype的数据类型不在具体范围内。</td>
      </tr>
      <tr>
        <td>srcTensor的view shape维度不在[2, 6]的范围</td>
      </tr>
      <tr>
        <td>srcTensor传入空Tensor</td>
      </tr>
      <tr>
        <td rowspan="2"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="2"> 361001 </td>
        <td>产品型号不支持。</td>
      </tr>
      <tr>
        <td>转换格式不支持。</td>
      </tr>
    </tbody>
  </table>

## aclnnNpuFormatCastGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1665px;">
    <colgroup>
        <col style="width: 180px">
        <col style="width: 120px">
        <col style="width: 300px">
        <col style="width: 200px">
        <col style="width: 290px">
        <col style="width: 300px">
        <col style="width: 130px">
        <col style="width: 145px">
  </colgroup>
  <thead>
      <tr>
          <th>参数名</th>
          <th>输入/输出</th>
          <th>描述</th>
          <td>使用说明</td>
          <th>数据类型</th>
          <th>数据格式</th>
          <th>维度(shape)</th>
          <td>非连续Tensor</td>
      </tr>
  </thead>
  <tbody>
        <tr>
            <td>srcTensor（aclTensor*）</td>
            <td>输入</td>
            <td>输入张量，输入的数据只支持连续的Tensor。</td>
            <td>括号中对应其对应的枚举值。</td>
            <td>INT8（2）、UINT8（4）、INT32（3）、UINT32（8）、FLOAT（0）、FLOAT16（1）、BFLOAT16（27）、FLOAT8_E4M3FN<sup>2</sup>（36）、FLOAT4_E2M1<sup>2</sup>（40）、FLOAT4_E1M2<sup>2</sup>（41）、HIFLOAT8<sup>2</sup>（34）</td>
            <td>ND、NZ、NCDHW、NDC1HWC0、FRACTAL_Z_3D、NCL<sup>2</sup></td>
            <td>2-6</td>
            <td>-</td>
        </tr>
        <tr>
            <td>dstTensor（aclTensor*）</td>
            <td>输入</td>
            <td>转换后的目标张量，只支持连续的Tensor。</td>
            <td>括号中对应其对应的枚举值。</td>
            <td>INT8（2）、UINT8（4）、INT32（3）、UINT32（8）、FLOAT（0）、FLOAT16（1）、BFLOAT16（27）、FLOAT8_E4M3FN<sup>2</sup>（36）、FLOAT4_E2M1<sup>2</sup>（40）、HIFLOAT8<sup>2</sup>（34）</td>
            <td>ACL_FORMAT_ND（2）、ACL_FORMAT_NZ（29）、ACL_FORMAT_NCL（47）、ACL_FORMAT_NCHW<sup>1</sup>（0）、ACL_FORMAT_NHWC<sup>1</sup>（1）、ACL_FORMAT_HWCN<sup>1</sup>（16）、ACL_FORMAT_NCDHW<sup>1</sup>（30）、ACL_FORMAT_NDHWC<sup>1</sup>（27）、ACL_FORMAT_DHWCN<sup>1</sup>（31）、ACL_FORMAT_NDC1HWC0<sup>1</sup>（32）、ACL_FORMAT_FRACTAL_Z_3D<sup>1</sup>（33）、ACL_FORMAT_NC1HWC0<sup>1</sup>（3）、ACL_FORMAT_FRACTAL_Z<sup>1</sup>（4）。</td>
            <td>4-8</td>
            <td>-</td>
        </tr>
        <tr>
            <td>workspaceSize（uint64_t*）</td>
            <td>输入</td>
            <td>需要在Device侧申请的workspace的大小。</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor（aclOpExecutor**）</td>
            <td>输入</td>
            <td>包含算子计算流程的op执行器。</td>
            <td>-</td>
            <td>None</td>
            <td>None</td>
            <td>None</td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：上表数据类型列中的角标“1”代表该系列不支持的数据类型或数据格式。
  <!-- end id11 -->
  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：上表数据类型列中的角标“2”代表该系列不支持的数据类型或数据格式。
  <!-- end id12 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table>
    <thead>
      <tr>
        <th style="width: 291px">返回值</th>
        <th style="width: 135px">错误码</th>
        <th style="width: 724px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="1"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="1"> 161001 </td>
        <td>传入的srcTensor、dstTensor是空指针。</td>
      </tr>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="4"> 161002 </td>
        <td>srcTensor的数据类型，数据格式不在支持范围内。</td>
      </tr>
      <tr>
        <td>dstTensor的数据类型，数据格式不在支持范围内。</td>
      </tr>
      <tr>
        <td>srcTensor、dstTensor传入非连续的Tensor。</td>
      </tr>
      <tr>
        <td>srcTensor的view shape维度不在[2, 6]的范围，dstTensor的storage shape维度不在[4, 8]的范围。<sup>2</sup></td>
      </tr>
      <tr>
        <td rowspan="1"> ACLNN_ERR_RUNTIME_ERROR </td>
        <td rowspan="1"> 361001 </td>
        <td>产品型号不支持。</td>
      </tr>
    </tbody>
  </table>

  <!-- npu="950" id13 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：上表数据类型列中的角标“1”代表该系列不支持的拦截类型。
  <!-- end id13 -->
  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：上表数据类型列中的角标“2”代表该系列不支持的拦截类型。
  <!-- end id14 -->

## aclnnNpuFormatCast

- **参数说明**

  <table style="table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNpuFormatCastGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnNpuFormatCast默认确定性实现。

- 输入和输出支持以下数据类型组合：

  <!-- npu="950" id15 -->
  <details>
  <summary><term>Ascend 950PR/Ascend 950DT</term></summary>

  - aclnnNpuFormatCastCalculateSizeAndFormat接口：

      | srcTensor | srcTensor[数据格式](../../../docs/zh/context/data_format.md) | dstFormat | additionalDtype              | actualFormat                    |
      | --------- | -------------------------------------------------------- | --------- | ---------------------------- | ------------------------------- |
      | INT8      | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_INT8(2)                  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1)、ACL_BF16(27) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1)、ACL_BF16(27) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36) | ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16      | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT16(1) | ACL_FORMAT_FRACTAL_NZ(29) |
      | BFLOAT16     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_BF16(27)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT8_E4M3FN     | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT8_E8M0     | ACL_FORMAT_ND(2)/ACL_FORMAT_NCL(47)/ACL_FORMAT_NCHW(0) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E8M0   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT8_E4M3FN(36)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT4_E2M1(40)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E1M2 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FLOAT4_E1M2(41)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | HIFLOAT8 | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29) | HIFLOAT8(34)   | ACL_FORMAT_FRACTAL_NZ(29) |
      | FLOAT4_E2M1 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT4_E2M1(40) |  ACL_FORMAT_ND(2) |
      | FLOAT4_E2M1 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_FLOAT4_E2M1(40) |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | INT8 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_INT8(2)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | UINT8 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_UINT8(4)  |  ACL_FORMAT_ND(2) |
      | FLOAT8_E4M3FN | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT8_E4M3FN(36)  |  ACL_FORMAT_ND(2) |
      | FLOAT16 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT16(1) |  ACL_FORMAT_ND(2) |
      | BFLOAT16 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_BF16(27) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_2(52) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_4(53) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | INT32 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_INT32(3) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_2(52) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_4(53) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |
      | FLOAT | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2) | ACL_FLOAT(0) |  ACL_FORMAT_ND(2) |

  - aclnnNpuFormatCastGetWorkspaceSize接口：

      | srcTensor | dstTensor数据类型 | srcTensor[数据格式](../../../docs/zh/context/data_format.md) | dstTensor[数据格式](../../../docs/zh/context/data_format.md)            |
      | --------- | ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------ |
      | INT8      | INT8              | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT32     | INT32             | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ_C0_16(50) |
      | FLOAT     | FLOAT             | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ_C0_16(50)/ACL_FORMAT_FRACTAL_NZ_C0_32(51) |
      | FLOAT16   | FLOAT16           | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | BFLOAT16  | BFLOAT16          | ACL_FORMAT_ND(2) | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT8_E4M3FN  | FLOAT8_E4M3FN   | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT8_E8M0  | FLOAT8_E8M0   | ACL_FORMAT_ND(2)/ACL_FORMAT_NCL(47)/ACL_FORMAT_NCHW(0)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT4_E2M1  | FLOAT4_E2M1       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ_C0_32(51)       |
      | FLOAT4_E2M1  | FLOAT4_E2M1       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | FLOAT4_E1M2  | FLOAT4_E1M2       | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | HIFLOAT8  | HIFLOAT8             | ACL_FORMAT_ND(2)  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ(29) | ACL_FORMAT_ND(2)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2)       |
      | INT8      | INT8                 | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ(29)       | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ_C0_16(50) | ACL_FORMAT_ND(2)       |
      | UINT8     | UINT8                | ACL_FORMAT_FRACTAL_NZ_C0_32(51) | ACL_FORMAT_ND(2)       |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_2(52)     | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_4(53)     | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_16(50)    | ACL_FORMAT_ND(2)  |
      | INT32     | INT32             | ACL_FORMAT_FRACTAL_NZ_C0_32(51)    | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_2(52)     | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_4(53)     | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_16(50)    | ACL_FORMAT_ND(2)  |
      | FLOAT     | FLOAT             | ACL_FORMAT_FRACTAL_NZ_C0_32(51)    | ACL_FORMAT_ND(2)  |
      | FLOAT16   | FLOAT16           | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)       |
      | BFLOAT16  | BFLOAT16          | ACL_FORMAT_FRACTAL_NZ(29)          | ACL_FORMAT_ND(2)        |
      | FLOAT8_E4M3FN  | FLOAT8_E4M3FN    |  ACL_FORMAT_FRACTAL_NZ(29)     | ACL_FORMAT_ND(2)        |
      | FLOAT4_E2M1  | FLOAT4_E2M1        |  ACL_FORMAT_FRACTAL_NZ(29)     | ACL_FORMAT_ND(2)        |
      | FLOAT4_E2M1  | FLOAT4_E2M1        |  ACL_FORMAT_FRACTAL_NZ_C0_32(51)  | ACL_FORMAT_ND(2)        |

  - C0计算方法：$C0=\frac{32B}{size\ of\ additionalDtype}$

      | additionalDtype | C0 |
      | --------------- | -- |
      | ACL_INT8(2)     | 32 |
      | ACL_FLOAT16(1)  | 16 |
      | ACL_BF16(27)    | 16 |
      | ACL_FLOAT8_E4M3FN(36)    | 32 |
      | ACL_HIFLOAT8(34)    | 32 |

  - FLOAT8_E8M0特殊场景:
    - srcTensor数据类型为FLOAT8_E8M0，srcTensor格式为ND/NCL/NCHW，dstFormat为FRACTAL_NZ时，仅支持3维或4维scale输入。转换后的storage shape分别为`(ceil(N/16), ceil(K/64), 16, 2)`或`(g, ceil(N/16), ceil(K/64), 16, 2)`。
    - FLOAT8_E8M0 scale转换为昇腾亲和存储shape的能力不支持图模式，仅支持单算子模式和aclnn直调场景。

  - 当前不支持的特殊场景:
    - srcTensor的数据类型和additionalDtype相同，srcTensor格式为ND且类型为FLOAT16或BFLOAT16时，若维度表示为[k, n]，则k为1场景暂不支持。
    - 不支持调用当前接口转昇腾亲和[数据格式](../../../docs/zh/context/data_format.md)FRACTAL_NZ后，进行任何能修改张量的操作，如contiguous、pad、slice等;
    - 当srcTensor的shape后两维任意一维度shape等于1场景，也不允许转昇腾亲和[数据格式](../../../docs/zh/context/data_format.md)FRACTAL_NZ后再进行任何修改张量的操作，包括transpose。
  </details>
  <!-- end id15 -->

  <!-- npu="A3,910b" id16 -->
  <details>
  <summary><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></summary>

  - aclnnNpuFormatCastCalculateSizeAndFormat接口参数：

      | srcTensor | dstFormat                 | additionalDtype              | actualFormat                    |
      | --------- | ------------------------- | ---------------------------- | ------------------------------- |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_ND(2)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | ACL_FORMAT_FRACTAL_NZ(29)       |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCL(47)    | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32   | ACL_FORMAT_FRACTAL_NZ(29) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCHW(0)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_NZ(29)    |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCDHW(30)     | ACL_FORMAT_FRACTAL_NZ(29) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_NZ(29)    |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_ND(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_ND(2)       |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCL(47) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCL(47)       |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCHW(0)       |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_NZ(29)    | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32         | ACL_FORMAT_NCDHW(30)     |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCDHW(30)    | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NDC1HWC0(32)  |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NDC1HWC0(32)    | ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32           | ACL_FORMAT_NCDHW(30)     |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCDHW(30)    |ACL_FORMAT_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z_3D(33)  |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_Z_3D(33)    |ACL_FORMAT_NCDHW(30) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCDHW(30)  |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_DHWCN(30)    | ACL_FORMAT_FRACTAL_Z_3D(33) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z_3D(33) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCHW(0)   | ACL_FORMAT_NC1HWC0(3) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NC1HWC0(3) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NC1HWC0(3)   | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCHW(0) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NHWC(2)   | ACL_FORMAT_NC1HWC0(3) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NC1HWC0(3) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NC1HWC0(3)   | ACL_FORMAT_NHWC(2) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NHWC(2) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NCHW(0)   | ACL_FORMAT_FRACTAL_Z(4) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z(4) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_Z(4)   | ACL_FORMAT_NCHW(0) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NCHW(0) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_HWCN(16)   | ACL_FORMAT_FRACTAL_Z(4) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_FRACTAL_Z(4) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_FRACTAL_Z(4)   | ACL_FORMAT_HWCN(16) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_HWCN(16) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NDHWC(27)   | ACL_FORMAT_NDC1HWC0(32) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NDC1HWC0(32) |
      | 数据类型：INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32，数据格式：ACL_FORMAT_NDC1HWC0(32)   | ACL_FORMAT_NDHWC(27) | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32    | ACL_FORMAT_NDHWC(27) |

  - aclnnNpuFormatCastGetWorkspaceSize接口：

      | srcTensor | dstTensor数据类型 | dstTensor数据格式               |
      | --------- | ----------------- | ------------------------------- |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_FRACTAL_NZ(29)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_ND(2)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCDHW(30)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NDC1HWC0(32)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FRACTAL_Z_3D(33)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCL(47)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NCHW(0)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NC1HWC0(3)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NHWC(2)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_FRACTAL_Z(4)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_HWCN(16)       |
      | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32  | INT8, UINT8, FLOAT, FLOAT16, BF16, INT32, UINT32          | ACL_FORMAT_NDHWC(27)       |

  - C0计算方法：$C0=\frac{32B}{ge::GetSizeByDataType(static_cast additionalDtype)}$

      | srcTensor的基础类型 | C0 |
      | --------------- | -- |
      | ACL_FLOAT(0)、ACL_INT32(3)、ACL_UINT32(8)     | 8 |
      | ACL_FLOAT16(1)、ACL_BF16(27)  | 16 |
      | ACL_INT8(2)、ACL_UINT8(4)、ACL_HIFLOAT8(34)    | 32 |

  - 当前不支持的特殊场景:
    - 不支持调用当前接口转昇腾亲和[数据格式](../../../docs/zh/context/data_format.md)FRACTAL_NZ后,进行任何能修改张量的操作，如contiguous、pad、slice等;
    - 不允许转昇腾亲和[数据格式](../../../docs/zh/context/data_format.md)FRACTAL_NZ后再进行任何修改张量的操作，包括transpose。
  </details>
  <!-- end id16 -->

## 调用示例

<!-- npu="950" id17 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

  #define CHECK_RET(cond, return_expr) \
  do {                               \
      if (!(cond)) {                   \
      return_expr;                   \
      }                                \
  } while (0)

  #define LOG_PRINT(message, ...)     \
  do {                              \
      printf(message, ##__VA_ARGS__); \
  } while (0)

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2.构造输入与输出，需要根据API的接口自定义构造
      int64_t k = 64;
      int64_t n = 128;
      int64_t srcDim0 = k;
      int64_t srcDim1 = n;
      int dstFormat = 29;
      aclDataType srcDtype = aclDataType::ACL_INT32;
      aclDataType additionalDtype = aclDataType::ACL_FLOAT16;

      std::vector<int64_t> srcShape = {srcDim0, srcDim1};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(k * n, 1);
      for (size_t i = 0; i < k; i++) {
          for (size_t j = 0; j < n; j++) {
              srcHostData[i * n + j] = (j + 1) % 128;
          }
      }

      std::vector<int32_t> dstTensorHostData(k * n, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // 创建src  aclTensor
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3.调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 计算目标tensor的shape和format
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5.获取输出的值，将device侧内存上的结果拷贝至host侧
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6.释放dstShape、aclTensor和aclScalar
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7.释放device资源
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
<!-- end id17 -->

<!-- npu="A3,910b" id18 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_npu_format_cast.h"

  #define CHECK_RET(cond, return_expr) \
  do {                               \
      if (!(cond)) {                   \
      return_expr;                   \
      }                                \
  } while (0)

  #define LOG_PRINT(message, ...)     \
  do {                              \
      printf(message, ##__VA_ARGS__); \
  } while (0)

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat, const int additionalDtype,  int64_t** dstShape, uint64_t* dstShapeSize, int* actualFormat);
  extern "C" aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,uint64_t* workspaceSize, aclOpExecutor** executor);
  extern "C" aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      // 此处修改src的format
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                                  shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                aclDataType dataType, aclTensor** tensor, aclFormat format) {
      auto size = hostData.size() * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                  format, *storageShape, *storageShapeSize, *deviceAddr);
      return 0;
  }

  int main() {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2.构造输入与输出，需要根据API的接口自定义构造

      int dstFormat = 32;
      //此处修改目标format
      aclDataType srcDtype = aclDataType::ACL_INT32;
      int additionalDtype = -1;

      // std::vector<int64_t> srcShape = {srcDim0 , srcDim1};
      int64_t N = 1;
      int64_t C = 17;
      int64_t D = 1;
      int64_t H = 2;
      int64_t W = 2;

      std::vector<int64_t> srcShape = {N, C, D, H, W};
      void* srcDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;
      aclTensor* srcTensor = nullptr;
      aclTensor* dstTensor= nullptr;
      std::vector<int32_t> srcHostData(N * C * D * H * W, 1);

      int num = 0;
      for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
              for (int d = 0; d < D; ++d) {
                  for (int h = 0; h < H; ++h) {
                      for (int w = 0; w < W; ++w) {
                          // 按 行主序排布，计算线性索引
                          int index = (((n * C + c) * D + d) * H + h) * W + w;
                          srcHostData[index] = num;
                          num++;
                      }
                  }
              }
          }
      }

      std::vector<int32_t> dstTensorHostData(N * C * D * H * W, 1);

      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;

      // 创建src  aclTensor
      ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, srcDtype, &srcTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3.调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;
      std::cout << "init actualFormat = " << actualFormat << std::endl;
      // 计算目标tensor的shape和format
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, dstFormat, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      std::cout << "actualFormat = " << actualFormat << std::endl;
      std::cout << "&dstShape = " << &dstShape << std::endl;
      std::cout << "dstShape = [ ";
      for (int64_t i = 0; i < dstShapeSize; ++i) {
          std::cout << dstShape[i] << " ";
      }
      std::cout << "]" << std::endl;

      ret = CreateAclTensorWithFormat(dstTensorHostData, srcShape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &dstTensor, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
      ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. (固定写法)同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5.获取输出的值，将device侧内存上的结果拷贝至host侧
      auto size = 1;
      for (size_t i = 0; i < dstShapeSize; i++) {
          size *= dstShape[i];
      }

      std::vector<int32_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dstDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
      }

      // 6.释放dstShape、aclTensor和aclScalar
      delete[] dstShape;
      aclDestroyTensor(srcTensor);
      aclDestroyTensor(dstTensor);

      // 7.释放device资源
      aclrtFree(srcDeviceAddr);
      aclrtFree(dstDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
<!-- end id18 -->
