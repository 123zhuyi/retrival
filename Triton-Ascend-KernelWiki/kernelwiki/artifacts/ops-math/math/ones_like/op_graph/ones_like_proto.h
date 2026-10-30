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
 * \file ones_like_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ONES_LIKE_H_
#define OPS_OP_PROTO_INC_ONES_LIKE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Returns a tensor of the same shape and type with all elements set to one.
 *@par Inputs:
 * One input:
 * x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types: float16,
 * float32, int8, uint8, int16, uint16, int32, int64, complex128, bool, double, bfloat16.
 *
 *@par Outputs:
 * y: A ND Tensor of the same dtype as "x".
 *
 *@par Third-party framework compatibility
 * Compatible with TensorFlow operator OnesLike.
 */
REG_OP(OnesLike)
    .INPUT(
        x, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DI_UINT16, DT_INT32, DT_INT64,
                DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OUTPUT(
        y, TensorType(
               {DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DI_UINT16, DT_INT32, DT_INT64,
                DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(OnesLike)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ONES_LIKE_H_
