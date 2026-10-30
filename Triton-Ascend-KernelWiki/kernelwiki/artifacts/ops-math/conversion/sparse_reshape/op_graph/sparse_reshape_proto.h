/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_reshape_proto.h
 * \brief Operator registration for SparseReshape
 */
#ifndef SPARSE_RESHAPE_PROTO_H_
#define SPARSE_RESHAPE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Reshapes a sparse tensor from input shape to output shape.
 *@par Inputs:
 *Three inputs, including:
 * @li indices: A 2D Tensor. Must be one of the following types: int32, int64. Shape [nnz, input_rank].
 * @li shape: A 1D Tensor. Must be one of the following types: int32, int64. Shape [input_rank].
 * @li new_shape: A 1D Tensor. Must be one of the following types: int32, int64. Shape [output_rank].
 *
 *@par Outputs:
 *Two outputs, including:
 * @li y_indices: A 2D Tensor. Must be one of the following types: int32, int64. Shape [nnz, output_rank].
 * @li y_shape: A 1D Tensor. Must be one of the following types: int32, int64. Shape [output_rank].
 *
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator SparseReshape.
 */
REG_OP(SparseReshape)
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .INPUT(new_shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y_indices, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y_shape, TensorType({DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(SparseReshape)
} // namespace ge
#endif // SPARSE_RESHAPE_PROTO_H_
