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
 * \file tensor_move_proto.h
 * \brief
 */
#ifndef OP_PROTO_TENSOR_MOVE_H_
#define OP_PROTO_TENSOR_MOVE_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Copy data from x to y.

*@par Inputs:
* One input, including:
* x: A ND Tensor, Support 1D ~ 8D, Must be one of the following types:
     double, float16, float, int8, int32, uint32, uint8, int64, uint64, int16,
     uint16, bool, bfloat16, hifloat8, float8_e5m2, float8_e4m3fn, complex32, complex64. \n

*@par Outputs:
* y: A ND Tensor. Has the same dtype as "x". \n

*@par Third-party framework compatibility

*@par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                          DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                           DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(TensorMove)

} // namespace ge
#endif // OP_PROTO_TENSOR_MOVE_H_