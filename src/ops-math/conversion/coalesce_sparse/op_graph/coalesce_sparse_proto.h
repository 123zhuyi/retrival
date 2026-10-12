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
 * \file coalesce_sparse_proto.h
 * \brief
 */

#ifndef COALESCE_SPARSE_PROTO_H
#define COALESCE_SPARSE_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Tutel combine function in moe.
 *
 * @par Inputs:
 * @li unique_len: A one-dimensional tensor of the type DT_INT32, DT_INT64.
 * @li unique_indices:A one-dimensional tensor of the type DT_INT32, DT_INT64.
 * @li indices: A two-dimensional tensor of the type DT_INT32, DT_INT64.
 * @li values: A mutable Tensor of the type DT_INT32, DT_FLOAT16, DT_FLOAT32.
 *
 * @par Outputs:
 * @li new_inidces: A two-dimensional tensor of the type DT_INT32, DT_INT64.
 * @li new_values: A mutable Tensor of the type DT_INT32, DT_FLOAT16, DT_FLOAT32.\n
 */
REG_OP(CoalesceSparse)
    .INPUT(unique_len, TensorType({DT_INT32, DT_INT64}))
    .INPUT(unique_indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(values, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(new_inidces, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(new_values, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT32}))
    .OP_END_FACTORY_REG(CoalesceSparse)
} // namespace ge
#endif