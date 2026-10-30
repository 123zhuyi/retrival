/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file transdata_proto.h
 * \brief
 */
#ifndef OP_PROTO_TRANSDATA_PROTO_H_
#define OP_PROTO_TRANSDATA_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
* @brief Do format transfer for various data format.
* In general, the framework will insert it atomatically. \n

* @par Inputs:
* src: A Tensor. Support 2D ~ 6D. For all branches can be types: bfloat16, float16, float32, int32, int8, bool.
*      For branches without padding also can be types: int16, int64, uint8, uint16, uint32, uint64. \n

* @par Attributes:
* @li src_format: A string source data format, can be "NHWC", "NCHW" etc.
* @li dst_format: A string target data format, can be "NCHW" etc.
* @li src_subformat: A optional int32 for source sub-format, default value is 0.
* @li dst_subformat: A optional int32 for target sub-format, default value is 0.
* @li groups: A optional int32, the N axis must be divisible by "groups". Defaults to 1. \n

* @par Outputs:
* dst: A Tensor. Support 2D ~ 6D. Has the same type as "src".
*\n
*\n
* The following value ranges must be met.
* '<===>' indicates that format is bidirectionly supported, either input or output.
* '===>' indicates that format is unbidirectionly supported, and the input and
* output data types must be correspond to each other. \n
*\n
*\n
| src_format <===> dst_format | dtype                              | C0    | groups |\n
| :-------------------------: | :--------------------------------: |:-----:| :----: |\n
| NCHW <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> NC1HWC0         | int8, uint8                        | 32    | 1      |\n
| NHWC <====> NC1HWC0         | float32, int32,uint32              | 8,16  | 1      |\n
| NHWC <====> NC1HWC0         | bfloat16, float16                  | 16    | 1      |\n
| NHWC <====> NC1HWC0         | int8,  uint8                       | 32    | 1      |\n
| ND <====> FRACTAL_NZ        | float32, int32,uint32              | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | bfloat16, float16                  | 16    | 1      |\n
| ND <====> FRACTAL_NZ        | int8, uint8                        | 32    | 1      |\n
| ND <====> FRACTAL_NZ        | float4_e2m1                        | 64    | 1      |\n
| ND ====> FRACTAL_NZ         | float4_e1m2                        | 64    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| NCHW <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| NCHW <====> FRACTAL_Z       | int8,  uint8                       | 32    | 1      |\n
| HWCN <====> FRACTAL_Z       | float32, int32,uint32              | 8,16  | 1      |\n
| HWCN <====> FRACTAL_Z       | bfloat16, float16                  | 16    | 1      |\n
| HWCN <====> FRACTAL_Z       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |\n
| NCDHW <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NDHWC <====> NDC1HWC0       | float32, int32,uint32              | 8,16  | 1      |\n
| NDHWC <====> NDC1HWC0       | bfloat16, float16                  | 16    | 1      |\n
| NDHWC <====> NDC1HWC0       | int8, uint8                        | 32    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, int32,uint32              | 8,16  | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| NCDHW <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | float32, int32,uint32              | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | bfloat16, float16                  | 16    | 1      |\n
| DHWCN <====> FRACTAL_Z_3D   | int8, uint8                        | 32    | 1      |\n
| NCHW <====> FRACTAL_Z       | float32, uint32, int32             | 8     | >1     |\n
| NCHW <====> FRACTAL_Z       | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| HWCN ====> FRACTAL_Z        | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float32, uint32, int32             | 8     | >1     |\n
| NCDHW <====> FRACTAL_Z_3D   | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float32, uint32, int32             | 8     | >1     |\n
| FRACTAL_Z_3D ====> DHWCN    | float16, bfloat16, uint16, int16   | 16    | >1     |\n
| NCHW ====> FRACTAL_Z_C04    | float16, bfloat16                  | 16    | 1      |\n
| FRACTAL_Z_C04 ====> NCHW    | float32                            | 16    | 1      |\n
| ND <====> FRACTAL_NZ_C0_16  | float32, uint32, int32             | 16    | 1      |\n
| ND <====> FRACTAL_NZ_C0_32  | float4_e2m1                        | 32    | 1      |\n
| FRACTAL_NZ_C0_2 ====> ND    | float32, int32                     | 2     | 1      |\n
| FRACTAL_NZ_C0_4 ====> ND    | float32, int32                     | 4     | 1      |\n
*\n
*
*/
REG_OP(TransData)
    .INPUT(src, TensorType({TensorType::BasicType(), DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(dst, TensorType({TensorType::BasicType(), DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .REQUIRED_ATTR(src_format, String)
    .REQUIRED_ATTR(dst_format, String)
    .ATTR(src_subformat, Int, 0)
    .ATTR(dst_subformat, Int, 0)
    .ATTR(groups, Int, 1)
    .OP_END_FACTORY_REG(TransData)

} // namespace ge
#endif // OP_PROTO_TRANSDATA_PROTO_H_