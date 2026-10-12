/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file global_average_pool_proto.h
 * \brief
 */
#ifndef GLOBAL_AVERAGE_POOL_PROTO_H_
#define GLOBAL_AVERAGE_POOL_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same channel.
* This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor.

* @par Inputs:
* @li x: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W),
* where N is the batch size, C is the number of channels, and H and W are the height and the width of the data.
* For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.

* @par Outputs:
* y: Output data tensor from pooling across the input tensor. The output tensor has the same rank as the input.
* The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are all 1

* @par Restrictions:
* Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it directly.
*/
REG_OP(GlobalAveragePool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(GlobalAveragePool);

} // namespace ge

#endif // GLOBAL_AVERAGE_POOL_PROTO_H_

