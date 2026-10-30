# TransData

## 产品支持情况

| 产品                                              | 是否支持 |
|:------------------------------------------------| :------: |
| <term>Ascend 950PR/Ascend 950DT</term>          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>             |    √     |
| <term>Atlas 推理系列产品</term>                       |    √     |
| <term>Atlas 训练系列产品</term>                       |    √     |

## 功能说明

- 算子功能：对各种数据格式进行格式转换。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>src</td>
      <td>输入</td>
      <td>待转换tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT8、BOOL、INT16、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>src_format</td>
      <td>属性</td>
      <td>原数据格式字符串。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_format</td>
      <td>属性</td>
      <td>目标数据格式字符串。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>src_subformat</td>
      <td>属性</td>
      <td>原子数据格式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_subformat</td>
      <td>属性</td>
      <td>目标子数据格式。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>属性</td>
      <td>约束说明中的groups列名。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32、INT32、INT8、BOOL、INT16、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

* 对于带有padding的分支：不支持INT16、INT64、UINT8、UINT16、UINT32、UINT64。

## 约束说明

* 必须满足以下值范围：
'<====>'表示格式是双向支持的，无论是输入还是输出。
'====>'表示不支持该格式转换，输入和输出数据类型必须相互对应。
[数据格式简介](../../docs/zh/context/data_format.md)

  | src_format <====> dst_format | dtype                              | C0    | groups |
  | :-------------------------: | :--------------------------------: |:-----:| :----: |
  | NCHW <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |
  | NCHW <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |
  | NCHW <====> NC1HWC0         | int8, uint8                        | 32    | 1      |
  | NHWC <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |
  | NHWC <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |
  | NHWC <====> NC1HWC0         | int8,  uint8                       | 32    | 1      |
  | ND <====> FRACTAL_NZ        | float32, int32,uint32              | 16    | 1      |
  | ND <====> FRACTAL_NZ        | bfloat16, float16                  | 16    | 1      |
  | ND <====> FRACTAL_NZ        | int8, uint8                        | 32    | 1      |
  | NCHW <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |
  | NCHW <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |
  | NCHW <====> FRACTAL_Z       | int8,  uint8                       | 32    | 1      |
  | HWCN <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |
  | HWCN <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |
  | HWCN <====> FRACTAL_Z       | int8, uint8                        | 32    | 1      |
  | NCDHW <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |
  | NCDHW <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |
  | NCDHW <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |
  | NDHWC <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |
  | NDHWC <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |
  | NDHWC <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | float32, int32,uint32              | 8,16  | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |
  | NCDHW <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |
  | DHWCN <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |
  | NCHW <====> FRACTAL_Z       | float32, uint32, int32             | 8     | >1     |
  | NCHW <====> FRACTAL_Z       | float16, bfloat16, uint16, int16   | 16    | >1     |
  | HWCN ====> FRACTAL_Z        | float16, bfloat16, uint16, int16   | 16    | >1     |
  | NCDHW <====> FRACTAL_Z_3D   | float32, uint32, int32             | 8     | >1     |
  | NCDHW <====> FRACTAL_Z_3D   | float16, bfloat16, uint16, int16   | 16    | >1     |
  | FRACTAL_Z_3D ====> DHWCN    | float32, uint32, int32             | 8     | >1     |
  | FRACTAL_Z_3D ====> DHWCN    | float16, bfloat16, uint16, int16   | 16    | >1     |
  | NCHW ====> FRACTAL_Z_C04    | float16, bfloat16                  | 16    | 1      |
  | FRACTAL_Z_C04 ====> NCHW    | float32                            | 16    | 1      |
  | ND ====> FRACTAL_NZ_C0_16   | float32, uint32, int32             | 16    | 1      |

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_trans_matmul_weight](examples/test_aclnn_trans_matmul_weight.cpp) | 通过[aclnnTransMatmulWeight](docs/aclnnTransMatmulWeight.md)接口方式调用Pack算子。 |
