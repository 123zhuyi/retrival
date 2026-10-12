/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACOSH_GRAD_PROTO_H_
#define ACOSH_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 *@brief AcoshGrad: dx = dy / sinh(y), reverse gradient of acosh.
 *@par Inputs:
 *Two inputs:
 * @li y: A Tensor. Must be float16, bfloat16, or float32.
 * @li dy: A Tensor. Must be float16, bfloat16, or float32. Same shape and dtype as y.
 *@par Outputs:
 *dx: A Tensor. Must be float16, bfloat16, or float32. Same shape and dtype as y.
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow AcoshGrad operator.
 */
REG_OP(AcoshGrad)
    .INPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OP_END_FACTORY_REG(AcoshGrad)
} // namespace ge
#endif // ACOSH_GRAD_PROTO_H_
