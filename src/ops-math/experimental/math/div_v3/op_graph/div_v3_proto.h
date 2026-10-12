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
 * \file div_v3_proto.h
 * \brief DivV3 operator prototype registration
 */

#ifndef OPS_OP_PROTO_INC_DIV_V3_H_
#define OPS_OP_PROTO_INC_DIV_V3_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Division with rounding mode (DivMod).
 *
 * @par Inputs:
 *  @li x1: dividend tensor. float32, float16, int32, int16, bfloat16.
 *  @li x2: divisor tensor.  float32, float16, int32, int16, bfloat16.
 *
 * @par Attributes:
 *  @li mode: rounding mode. 0=RealDiv, 1=TruncDiv, 2=FloorDiv.
 *
 * @par Outputs:
 *  y: result tensor. float32, float16, int32, int16, bfloat16.
 */
REG_OP(DivV3)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT16, DT_INT32, DT_BF16}))
    .ATTR(mode, Int, 0)
    .OP_END_FACTORY_REG(DivV3)

} // namespace ge

#endif // OPS_OP_PROTO_INC_DIV_V3_H_
