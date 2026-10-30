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
 * \file is_nan_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_NAN_H_
#define OPS_OP_PROTO_INC_IS_NAN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns which elements of x are NaN.

 * @par Inputs:
 * x: A Tensor, Format is ND, Support 1D ~ 8D.
 * Type must be one of the following types: float16, bfloat16, float32, double.

 * @par Outputs:
 * y: A Tensor of type bool, shape is same as x. Returns which elements of x are isnan

 * @par Third-party framework compatibility.
 * Compatible with tensorflow IsNan operator.
 */
REG_OP(IsNan)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(IsNan)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_NAN_H_
