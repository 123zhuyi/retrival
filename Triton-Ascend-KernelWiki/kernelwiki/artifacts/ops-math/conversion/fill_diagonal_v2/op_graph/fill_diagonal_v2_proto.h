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
 * \file fill_diagonal_v2_proto.h
 * \brief
 */
#ifndef OPS_CONVERSION_FILL_DIAGONAL_V2_GRAPH_PLUGIN_FILL_DIAGONAL_V2_PROTO_H_
#define OPS_CONVERSION_FILL_DIAGONAL_V2_GRAPH_PLUGIN_FILL_DIAGONAL_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Fill diagonal of at least 2 dimension tensors with value inplace. \n

* @par Inputs:
* @li x: A tensor. Must be one of the following types:float16, float32, float64, int8,
                                                    int16, int32, int64, uint8, bool, bfloat16. \n
* @li fill_value: A tensor. Scalar value to be filled, Must have the same type as "x". \n

* @par Outputs:
* x: A tensor. Refers to the same tensor as input "x". \n

* @par Attributes:
* wrap: An optional bool. Defaults to "False". If "True", use recursive fill. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator FillDiagonal.
*/
REG_OP(FillDiagonalV2)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .INPUT(fill_value, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .OUTPUT(x, TensorType({BasicType(), DT_BOOL, DT_BF16}))
    .ATTR(wrap, Bool, false)
    .OP_END_FACTORY_REG(FillDiagonalV2)

} // namespace ge

#endif