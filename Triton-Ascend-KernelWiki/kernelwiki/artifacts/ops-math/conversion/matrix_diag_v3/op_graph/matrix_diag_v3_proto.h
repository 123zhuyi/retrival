/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PROTO_MATRIX_DIAG_V3_H_
#define OP_PROTO_MATRIX_DIAG_V3_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns a batched diagonal tensor with given batched diagonal values .

* @par Inputs:
* Five inputs, including:
* @li x: Rank `r`, where `r >= 1`. \n

* @li k:
* Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
* diagonal, and negative value means subdiagonals. `k` can be a single integer
* (for a single diagonal) or a pair of integers specifying the low and high ends
* of a matrix band. `k[0]` must not be larger than `k[1]`. \n

* @li num_rows:
* The number of rows of the output matrix. If it is not provided, the op assumes
* the output matrix is a square matrix and infers the matrix size from k and the
* innermost dimension of `diagonal`. \n

* @li num_cols: An NCHW, NHWC, or ND Tensor.
* The number of columns of the output matrix. If it is not provided, the op
* assumes the output matrix is a square matrix and infers the matrix size from
* k and the innermost dimension of `diagonal`. \n

* @li padding_value: The number to fill the area outside the specified diagonal band with. \n

* @par Attributes:
* @li align: An optional string from: "LEFT_RIGHT", "RIGHT_LEFT", "LEFT_LEFT", "RIGHT_RIGHT".
* Defaults to "RIGHT_LEFT". \n

* @par Outputs:
* @li y: Has rank `r+1` when `k` is an integer or `k[0] == k[1]`, rank `r` otherwise. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ScatterUpdate.
*/
REG_OP(MatrixDiagV3)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .INPUT(k, TensorType({DT_INT32}))
    .INPUT(num_rows, TensorType({DT_INT32}))
    .INPUT(num_cols, TensorType({DT_INT32}))
    .INPUT(padding_value, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL}))
    .ATTR(align, String, "RIGHT_LEFT")
    .OP_END_FACTORY_REG(MatrixDiagV3)
} // namespace ge

#endif // OP_PROTO_MATRIX_DIAG_V3_H_