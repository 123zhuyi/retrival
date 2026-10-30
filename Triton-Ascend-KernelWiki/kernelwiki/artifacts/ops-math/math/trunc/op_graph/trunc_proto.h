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
 * \file trunc_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TRUNC_H_
#define OPS_OP_PROTO_INC_TRUNC_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns a new tensor with the truncated integer values of the elements of input.

*@par Inputs:
*One inputs, including:
* input_x: A tensor. Must be one of the following types: float16, bfloat16, float32, int8, uint8, int32. \n

*@par Outputs:
* output_y: A tensor with the same type and shape of input_x \n

*@par Third-party framework compatibility
*Compatible with the Pytorch operator Trunc. \n
*/
REG_OP(Trunc)
    .INPUT(input_x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .OUTPUT(output_y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8}))
    .OP_END_FACTORY_REG(Trunc)

} // namespace ge

#endif // OPS_OP_PROTO_INC_TRUNC_H_

