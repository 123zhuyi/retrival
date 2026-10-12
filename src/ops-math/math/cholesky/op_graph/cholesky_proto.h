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
 * \file cholesky_proto.h
 * \brief Cholesky 算子 GE IR 原型注册（图模式）
 *
 * 输入 x，输出 y，无属性。
 */
#ifndef OPS_OP_PROTO_INC_CHOLESKY_H_
#define OPS_OP_PROTO_INC_CHOLESKY_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Computes the Cholesky decomposition of one or more square matrices . \n

 *@par Inputs:
 *The input x has to be symmetric and positive definite.Inputs include:
 *x:A Tensor. Must be one of the following types: double, float32, float16,
 complex64, complex128. Shape is [..., M, M] . \n

 *@par Outputs:
 *y:A Tensor. Has the same type as x . \n

 *@attention Constraints:
 *The input x is a tensor of shape [..., M, M] whose inner-most 2 dimensions
 form square matrices.

 *@par Third-party framework compatibility
 *Compatible with tensorflow Cholesky operator.
 */
REG_OP(Cholesky)
    .INPUT(x, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_FLOAT16, DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Cholesky)

} // namespace ge

#endif // OPS_OP_PROTO_INC_CHOLESKY_H_
