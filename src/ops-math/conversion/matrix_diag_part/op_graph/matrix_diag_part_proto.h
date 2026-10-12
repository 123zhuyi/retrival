/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_MATH_CONVERSION_MATRIX_DIAG_PART_OP_GRAPH_MATRIX_DIAG_PART_PROTO_H_
#define OPS_MATH_CONVERSION_MATRIX_DIAG_PART_OP_GRAPH_MATRIX_DIAG_PART_PROTO_H_

namespace ge {

/**
 * @brief Returns the main diagonal (k=0) of the innermost matrix of x. \n
 *
 * @par Inputs:
 * x: A Tensor. Must be one of BasicType. \n
 *
 * @par Outputs:
 * y: A Tensor. Has the same type as "x". \n
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator MatrixDiagPart.
 */
#ifndef OPS_PROTO_DEF_MATRIXDIAGPART
#define OPS_PROTO_DEF_MATRIXDIAGPART
REG_OP(MatrixDiagPart)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(MatrixDiagPart)
#endif
} // namespace ge

#endif // OPS_MATH_CONVERSION_MATRIX_DIAG_PART_OP_GRAPH_MATRIX_DIAG_PART_PROTO_H_
