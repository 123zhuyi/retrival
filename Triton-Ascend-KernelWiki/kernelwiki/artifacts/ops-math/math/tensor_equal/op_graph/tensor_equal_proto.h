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
 * \file tensor_equal_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TENSOR_EQUAL_H_
#define OPS_OP_PROTO_INC_TENSOR_EQUAL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Compare two tensors are totally equal or not, only output a bool value"

* @par Inputs:
* Two inputs, including:
* @li input_x: A ND tensor. the first tensor. Must be one of the following types: float16, float32, double,
* int64, int32, int8, uint8, bool, bfloat16, int16, uint16, uint32, uint64. \n
* @li input_y: A ND tensor of the same dtype as "input_x". \n

* @par Outputs:
* output_z: A ND tensor. Bool type, compare result of the two inputs. True if element in input_x is equal to input_y,
False otherwise. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch equal operator. \n
*/
REG_OP(TensorEqual)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL,
                                DT_BF16, DT_INT16, DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(input_y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_INT8, DT_UINT8, DT_BOOL,
                                DT_BF16, DT_INT16, DT_UINT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(output_z, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(TensorEqual)

} // namespace ge

#endif // OPS_OP_PROTO_INC_TENSOR_EQUAL_H_
