/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ZETA_PROTO_H_
#define ZETA_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief Compute the Hurwitz zeta function.

 *@par Inputs:
 *The input q must be the same type as x. Inputs include:
 *@li x:A Tensor. Must be one of the following types: float32, double.
 *@li q:A Tensor. Must have the same type as x. \n

 *@par Outputs:
 *z:A Tensor. Has the same type as x. \n

 *@attention Constraints:
 *The implementation for Zeta on Ascend uses ai cpu, with bad performance.

 *@par Third-party framework compatibility.
 *Compatible with tensorflow Zeta operator.
 */
REG_OP(Zeta)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT}))
    .INPUT(q, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OUTPUT(z, TensorType({DT_DOUBLE, DT_FLOAT}))
    .OP_END_FACTORY_REG(Zeta)
} // namespace ge

#endif // ZETA_PROTO_H_
