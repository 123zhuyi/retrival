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
 * \file addcmul_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADDCMUL_H_
#define OPS_OP_PROTO_INC_ADDCMUL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs the element-wise multiplication of tensor x1 by tensor x2,
* multiply the result by the scalar value and add it to tensor input_data

* @par Inputs:
* Four inputs, including:
* @li input_data: A mutable input Tensor. Must be one of the following types:
*     float16, bfloat16, float32, double, int64, int8, int32, uint8.
* @li x1: A mutable input Tensor of the same dtype as input_data.
* @li x2: A mutable input Tensor of the same dtype as input_data.
* @li value: A ND mutable input tensor which includes only one element.
*            Must be one of the following types:
*            float16, bfloat16, float32, double, int64, int8, int32, uint8. \n

* @par Outputs:
* y: A mutable output Tensor. Has the same dtype as input_data. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator Addcmul.
*/
REG_OP(Addcmul)
    .INPUT(input_data, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_DOUBLE, DT_INT64}))
    .OP_END_FACTORY_REG(Addcmul)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADDCMUL_H_
