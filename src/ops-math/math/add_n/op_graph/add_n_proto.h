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
 * \file add_n_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADD_N_H_
#define OPS_OP_PROTO_INC_ADD_N_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Adds all input tensors element-wise.

* @par Inputs:
* Dynamic inputs, including:
* x: A list of tensor objects, each with same shape and type. The supported types are:
* bfloat16, float16, float32, int32, int64. It's a dynamic input. \n

* @par Attributes:
* N: A required attribute of type int32, means nums of inputs. \n

*@par Outputs:
*y: An ND tensor. Has the same shape and type as the elements of "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AddN.
*/
REG_OP(AddN)
    .DYNAMIC_INPUT(x, TensorType({NumberType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({NumberType(), DT_VARIANT}))
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(AddN)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADD_N_H_
