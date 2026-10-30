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

#ifndef OPS_OP_PROTO_INC_ATANH_H_
#define OPS_OP_PROTO_INC_ATANH_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes inverse hyperbolic tangent of x element-wise.
 * @par Formula: y = 0.5 * ln((1 + x) / (1 - x))
 *
 * @par Inputs:
 *  @li x: A tensor. Must be one of: float32, float16, bfloat16.
 *
 * @par Outputs:
 *  @li y: A tensor with the same type and shape as x.
 */
REG_OP(Atanh)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(Atanh)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ATANH_H_
