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
 * \file floor_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_FLOOR_H_
#define OPS_OP_PROTO_INC_FLOOR_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns element-wise largest integer not greater than "x".

*@par Inputs:
* x: An ND or 5HD tensor. Support 1D~8D. Must be one of the following types:
* bfloat16, float16, float32, double.

*@par Outputs:
*y: A ND Tensor of the same dtype as "x".

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Floor.
*/
REG_OP(Floor)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Floor)

} // namespace ge

#endif // OPS_OP_PROTO_INC_FLOOR_H_
