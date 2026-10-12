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
 * \file log1p_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LOG1P_H_
#define OPS_OP_PROTO_INC_LOG1P_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the logarithm of (x + 1) element-wise, y = ln(x + 1).

* @par Inputs:
* One input:\n
* x: A ND Tensor. Must be one of the following types: bfloat16, float16, float32, double, complex64, complex128. \n

* @par Outputs:
* y: A ND Tensor of the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Log1p.
*/
REG_OP(Log1p)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Log1p)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LOG1P_H_

