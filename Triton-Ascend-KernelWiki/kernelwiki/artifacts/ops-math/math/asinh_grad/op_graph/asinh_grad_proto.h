/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file asinh_grad_proto.h
 * \brief
 */
#ifndef ASINH_GRAD_PROTO_H_
#define ASINH_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge
{
/**
*@brief Computes gradients for Asinh operation. Computes: z = dy / cosh(y) = 2 * dy * exp(y) / (exp(2y) + 1).

*
*@par Inputs:
* @li y: A tensor. Must be one of the following types: bfloat16, float16, float32. The asinh forward output.
* @li dy: A tensor of the same dtype and shape as "y". The upstream gradient. Broadcasting is not supported.
*
*@par Outputs:
* z: A tensor. Has the same dtype and shape as "y". The gradient with respect to asinh's input x.
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator AsinhGrad.
*
*/
REG_OP(AsinhGrad)
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AsinhGrad)
} // namespace ge
#endif // ASINH_GRAD_PROTO_H_
