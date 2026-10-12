/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IS_FINITE_PROTO_H
#define IS_FINITE_PROTO_H

namespace ge {
/**
 * @brief Compute element-wise finiteness, return a boolean tensor.

 * @par Inputs:
 * x: An ND tensor. Support 1D ~ 8D. Must be one of the following types:
 * bfloat16, float16, float32, double, bool, uint8, int8, uint16, int16, int32, uint32, uint64, int64.

 * @par Outputs:
 * y: An ND tensor. Support 1D ~ 8D. Must have the same shape and format as input "x", and dytpe is bool.

 * @par Third-party framework compatibility.
 * Compatible with tensorflow IsFinite operator.
 */
REG_OP(IsFinite)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BOOL, DT_UINT8, DT_INT8, DT_UINT16,
                          DT_INT16, DT_INT32, DT_UINT32, DT_UINT64, DT_INT64}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsFinite)
}

#endif