/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_CROSS_H_
#define OPS_OP_PROTO_INC_CROSS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Calculate the cross product of two tensors. \n

*@par Inputs:
*One inputs, including:
* @li x1: A tensor. Must be one of the following types:
*     float16, float32, int32, int8, uint8, int16, double,
*     int64, uint16, uint32, uint64, complex64, complex128. \n
* @li x2: A tensor. Must be one of the following types:
*     float16, float32, int32, int8, uint8, int16, double,
*     int64, uint16, uint32, uint64, complex64, complex128. \n

*@par Attributes:
*@li dim: the dimination of compute.Defaults to -65530. \n

*@par Outputs:
*y: A Tensor with the same type and shape of x1's. \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator cross. \n
*/
REG_OP(Cross)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT16, DT_DOUBLE, DT_INT64, DT_UINT16, DT_UINT32,
                           DT_UINT64, DT_COMPLEX64, DT_COMPLEX128}))
    .ATTR(dim, Int, -65530)
    .OP_END_FACTORY_REG(Cross)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CROSS_H_

