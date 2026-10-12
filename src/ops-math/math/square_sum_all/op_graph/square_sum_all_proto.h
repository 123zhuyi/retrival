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
 * \file square_sum_all_proto.h
 * \brief SquareSumAll graph IR definition.
 */

#ifndef OPS_OP_PROTO_INC_SQUARE_SUM_ALL_H_
#define OPS_OP_PROTO_INC_SQUARE_SUM_ALL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Computes the sum of squares for two independent tensors.
 *
 * @par Inputs:
 * @li x1: A non-empty tensor of type float32. On Ascend 950, ND supports 0D to 8D, while NCHW and NHWC
 *          require 4D.
 * @li x2: A non-empty tensor with the same shape and format as x1 and type float32.
 *
 * @par Outputs:
 * @li y1: A scalar tensor of type float32. On Ascend 950, its format is ND. y1 = sum(x1 * x1).
 * @li y2: A scalar tensor of type float32. On Ascend 950, its format is ND. y2 = sum(x2 * x2).
 */
#ifndef OPS_PROTO_DEF_SQUARESUMALL
#define OPS_PROTO_DEF_SQUARESUMALL
REG_OP(SquareSumAll)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(SquareSumAll)
#endif
} // namespace ge

#endif // OPS_OP_PROTO_INC_SQUARE_SUM_ALL_H_
