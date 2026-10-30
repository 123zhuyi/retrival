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
 * \file expm1_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_EXPM1_H_
#define OPS_OP_PROTO_INC_EXPM1_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the exp(x) - 1 element-wise, y = e^x - 1.

* @par Inputs:
* One input:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, double, complex64, complex128.

* @par Outputs:
* y: A ND Tensor of the same dtype as "x".

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Expm1.
*/
REG_OP(Expm1)
    .INPUT(x, TensorType::UnaryDataType())
    .OUTPUT(y, TensorType::UnaryDataType())
    .OP_END_FACTORY_REG(Expm1)
} // namespace ge

#endif // OPS_OP_PROTO_INC_EXPM1_H_
