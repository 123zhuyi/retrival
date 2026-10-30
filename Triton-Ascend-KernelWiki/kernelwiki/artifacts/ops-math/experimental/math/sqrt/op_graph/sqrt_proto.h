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
 * \file sqrt_proto.h
 * \brief
 */

#ifndef OP_PROTO_SQRT_PROTO_H_
#define OP_PROTO_SQRT_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Computes the sqrt of a tensor.

*@par Inputs:
* @li x: A tensor of type float16, float32, bf16.
*@par Outputs:
* @li y: A tensor of type float16, float32, bf16.

*@par Third-party framework compatibility
* Compatible with the Pytorch operator Sqrt.
*/

REG_OP(Sqrt)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Sqrt);

} // namespace ge

#endif // OP_PROTO_SQRT_PROTO_H_
