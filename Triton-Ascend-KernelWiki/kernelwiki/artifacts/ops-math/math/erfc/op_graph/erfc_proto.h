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
 * \file erfc_proto.h
 * \brief
 */
#ifndef OPS_MATH_ERFC_PROTO_H_
#define OPS_MATH_ERFC_PROTO_H_

#include "graph/types.h"
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the Gauss complementary error function of "x" element-wise.

* @par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16 ,float32, double.

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Erfc.
*/

REG_OP(Erfc)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Erfc)
} // namespace ge
#endif // OPS_MATH_ERFC_PROTO_H_
