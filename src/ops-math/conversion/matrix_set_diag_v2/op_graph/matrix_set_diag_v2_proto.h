/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_SET_DIAG_PROTO_H_
#define MATRIX_SET_DIAG_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Returns a batched matrix tensor with new batched diagonal values.

* @par Inputs:
* Three inputs, including:
* @li input: Rank `r+1`, where `r >= 1`. \n

* @li diagonal: Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`. \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
* diagonal, and negative value means subdiagonals. `k` can be a single integer
* (for a single diagonal) or a pair of integers specifying the low and high ends
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @par Outputs:
* output: Rank `r+1`, with `output.shape = input.shape`. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/

#ifndef OPS_PROTO_DEF_MATRIXSETDIAGV2
#define OPS_PROTO_DEF_MATRIXSETDIAGV2
REG_OP(MatrixSetDiagV2)
    .INPUT(input, TensorType::BasicType())
    .INPUT(diagonal, TensorType::BasicType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixSetDiagV2);

#endif
} // namespace ge

#endif
