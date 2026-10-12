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
 * \file masked_scale_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_MASKED_SCALE_H_
#define OPS_OP_PROTO_INC_MASKED_SCALE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Calculates x * maske * value.
 *
 * @par Inputs:
 * @li x: An tensor of type float16 or float32, specifying the input to the data layer.
 * @li mask: An tensor of type int8 or float16 or float32, be same shape with x. \n
 *
 * @par Attributes:
 * value: An optional float, default value is 1.0. \n
 *
 * @par Outputs:
 * y: The output tensor of type float16 or float32. Same dtype and shape as x.
 *
 */
REG_OP(MaskedScale)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(mask, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(value, Float)
    .OP_END_FACTORY_REG(MaskedScale)

} // namespace ge

#endif // OPS_OP_PROTO_INC_MASKED_SCALE_H_

