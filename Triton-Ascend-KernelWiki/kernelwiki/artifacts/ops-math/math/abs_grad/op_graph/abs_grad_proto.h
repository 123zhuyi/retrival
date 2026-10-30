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
 * \file abs_grad_proto.h
 * \brief
 */
#ifndef OP_PROTO_ABS_GRAD_PROTO_H_
#define OP_PROTO_ABS_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Computes gradients for absolute operation.

 *
 * @par Inputs:
 * @li y: A tensor of type float16 or float32 or bfloat16. Support broadcasting operations.
 * @li dy: A tensor of the same dtype as "y".
 *
 * @attention Constraints:
 * "dy" has the same dtype as "y".
 *
 * @par Outputs:
 * z: A tensor. Has the same dtype as "y".
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator AbsGrad.
 *
 */
REG_OP(AbsGrad)
    .INPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(z, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(AbsGrad)

} // namespace ge
#endif // OP_PROTO_ABS_GRAD_PROTO_H_
