/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clip_by_value_v2_proto.h
 * \brief
 */

#ifndef CLIP_BY_VALUE_V2_PROTO_H_
#define CLIP_BY_VALUE_V2_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Clips tensor values to a specified min and max.
* When the input is bfloat16, float16, float32, int32 or int64, broadcasting operations are supported.  \n

* @par Inputs:
* Three inputs, including:
* @li x: A ND tensor with TensorType::NumberType().
* @li clip_value_min: A ND tensor of the same dtype as "x".
* @li clip_value_max: A ND tensor of the same dtype as "x". \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as "x". \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator clip.
*/
REG_OP(ClipByValueV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(clip_value_min, TensorType::NumberType())
    .INPUT(clip_value_max, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(ClipByValueV2)
} // namespace ge

#endif