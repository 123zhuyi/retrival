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
 * \file nan_to_num_proto.h
 * \brief nan_to_num proto
 */
#ifndef OPS_MATH_NAN_TO_NUM_PROTO_H_
#define OPS_MATH_NAN_TO_NUM_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Replaces NaN, positive infinity, and negative infinity values in x with specified values.
 * @par Inputs
 * x: An ND tensor of type float16, bfloat16, float32.
 * @par Outputs
 * y: An ND tensor of the same type as x.
 * @par Attributes
 * nan: An attr of type float. The value to replace NaN values.
 * posinf: An attr of type float. The value to replace positive infinity values.
 * neginf: An attr of type float. The value to replace negative infinity values.
 * @par Third-party framework compatibility
 * Compatible with PyTorch operator nan_to_num.
 */
REG_OP(NanToNum)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(nan, Float)
    .REQUIRED_ATTR(posinf, Float)
    .REQUIRED_ATTR(neginf, Float)
    .OP_END_FACTORY_REG(NanToNum)
} // namespace ge
#endif // OPS_MATH_NAN_TO_NUM_PROTO_H_
