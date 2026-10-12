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
 * \file lerp_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LERP_H_
#define OPS_OP_PROTO_INC_LERP_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Calculate the lerp function. \n

 * @par Inputs:
 * Three inputs, including:
 * @li start: A ND tensor. Must be one of the following types:
 *     float16, float32, bfloat16. \n
 * The shape of start, end and weight should satisfy the broadcast relationship. \n
 * @li end: A ND tensor. The same dtype as start. \n
 * @li weight: A ND tensor. The same dtype as start. \n

 * @par Outputs:
 * y: A ND tensor with the same dtype of start's. \n
 * The shape of 'y' is same as the shape of the tensor \n
 * whose shape is generated after 'start', 'end', and 'weight' broadcast operation.

 * @par Third-party framework compatibility
 * Compatible with the PyTorch operator Lerp. \n
 */
REG_OP(Lerp)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(end, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(Lerp)

} // namespace ge

#endif // OPS_OP_PROTO_INC_LERP_H_
