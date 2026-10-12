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
 * \file atan2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ATAN2_H_
#define OPS_OP_PROTO_INC_ATAN2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes arctangent of x1/x2 element-wise, respecting signs of the arguments. Support broadcasting operations.

*
* @par Inputs:
* @li x1: A ND tensor. Must be one of the following types: bfloat16, float16, float32, float64
* @li x2: A ND tensor of the same dtype as "x1".
*
* @par Outputs:
* y: A ND tensor. Has the same dtype as "x1".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Atan2.
*
*/
REG_OP(Atan2)
    .INPUT(x1, TensorType({FloatingDataType, DT_BF16}))
    .INPUT(x2, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Atan2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ATAN2_H_

