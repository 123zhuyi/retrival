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
 * \file ceil_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_CEIL_H_
#define OPS_OP_PROTO_CEIL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns element-wise smallest integer not less than "x".

*@par Inputs:
* x: A ND Tensor of type bfloat16 or float16 or float32 or float64. \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Ceil.
*/
REG_OP(Ceil)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Ceil)

} // namespace ge

#endif // OPS_OP_PROTO_CEIL_H_
