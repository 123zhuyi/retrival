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
 * \file axpy_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_AXPY_V2_H_
#define OPS_OP_PROTO_INC_AXPY_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the result of x2 * alpha + x1.

* @par Inputs:
* @li x1: An ND tensor of type float16, bfloat16, float32, int32, int64, uint8, int8, bool.
* @li x2: An ND tensor of type float16, bfloat16, float32, int32, int64, uint8, int8, bool.
* The shapes of "x1", "x2" must comply with the broadcast rule.
* @li alpha: A scalar tensor of type float16, bfloat16, float32, int32, int64, uint8, int8, bool. Shape must be [1]. \n

* @par Outputs:
* y: An ND tensor with type is after 'x1', 'x2' and 'alpha' type promotion,
* whose shape is generated after 'x1', 'x2' broadcast opratioan. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator Axpy.
*/
REG_OP(AxpyV2)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .INPUT(alpha, "T3")
    .OUTPUT(y, "T4")
    .DATATYPE(T1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_BF16, DT_INT64, DT_UINT8, DT_INT8, DT_BOOL}))
    .DATATYPE(T4, Promote({"T1", "T2", "T3"}))
    .OP_END_FACTORY_REG(AxpyV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_AXPY_V2_H_
