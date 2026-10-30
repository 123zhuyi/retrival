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
 * \file is_pos_inf_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_POS_INF_H_
#define OPS_OP_PROTO_INC_IS_POS_INF_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Compute element-wise infiniteness, return a boolean tensor.

 * @par Inputs:
 * x: A tensor of type float16, float32, bfloat16, format is ND.

 * @par Outputs:
 * y: A tensor. Has the same shape as x. Returns which elements of x are isposinf,
 * format is ND, dtype is bool.

 * @par Third-party framework compatibility.
 * Compatible with the Pytorch operator IsPosInf.

 *@attention Constraints:
 * Warning: The dtype of x does not support double now.
 */
REG_OP(IsPosInf)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BFLOAT16}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsPosInf)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_POS_INF_H_
