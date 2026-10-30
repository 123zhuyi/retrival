/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_MATRIX_DIAG_PART_V3_H_
#define OP_PROTO_MATRIX_DIAG_PART_V3_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns a batched diagonal tensor with given batched diagonal values .

* @par Inputs:
* @li x: A Tensor. Rank r tensor where r >= 2.
* @li k: A Tensor of type int32. Diagonal offset(s). Positive value means superdiagonal,
         0 refers to the main diagonal, and negative value means subdiagonals. k can be a
         single integer (for a single diagonal) or a pair of integers specifying the low and
         high ends of a matrix band. k[0] must not be larger than k[1].
* @li padding_value:A Tensor. Must have the same type as input. The value to fill the area
                    outside the specified diagonal band with. \n

* @par Outputs:
* @li y: A Tensor. Has the same type as "input". \n

* @par Attributes:
* @li align:An optional string from: "LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT". Defaults to "RIGHT_LEFT".

* @par Third-party framework compatibility
* Compatible with the Tensorflow  operator FillDiagonal.
*/
REG_OP(MatrixDiagPartV3)
    .INPUT(x, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(align, String, "RIGHT_LEFT")
    .OP_END_FACTORY_REG(MatrixDiagPartV3)
} // namespace ge

#endif // OP_PROTO_MATRIX_DIAG_PART_V3_H_