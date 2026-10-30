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
 * \file angle_v2_proto.h
 * \brief
 */
#ifndef OP_PROTO_ANGLE_V2_H_
#define OP_PROTO_ANGLE_V2_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief  Computes the element_wise angle(in radians) of the given input tensor.

* @par Inputs:
* x: A ND tensor of type float16, float, complex64, bool, uint8, int8, int16, int32, int64. \n
*
* @par Outputs:
* y: A ND tensor of type float16, float32. \n
*
* @par Third-party framework compatibility
* Compatible with the Pytorch operator Angle. \n
*/
REG_OP(AngleV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_COMPLEX64, DT_BOOL, DT_UINT8,
                          DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AngleV2)

}  // namespace ge


#endif  // OP_PROTO_ANGLE_V2_H_