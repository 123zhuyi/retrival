/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CDIST_PROTO_H
#define CDIST_PROTO_H

namespace ge {
/**
 * @brief Computes batched the p-norm distance between each pair of
 *the two collections of row vectors. \n

 *@par Inputs:
 *Two inputs, including:
 * @li x1: A tensor with shpae: B x P x M. Must be one of the following types:
 *     bfloat16, float16, float32. \n
 * @li x2: A tensor with shpae: B x R x M. Must be one of the following types:
 *     bfloat16, float16, float32. \n

 *@par Attributes:
 * @li p: An optional float >= 0 or inf. Defaults to 2.0. \n

 *@par Outputs:
 * y: A Tensor with the same type of x1's and with shape B x P x R. \n

 *@par Third-party framework compatibility
 *Compatible with the Pytorch operator Cdist. \n
 */
REG_OP(Cdist)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Float, 2.0)
    .OP_END_FACTORY_REG(Cdist)
}

#endif