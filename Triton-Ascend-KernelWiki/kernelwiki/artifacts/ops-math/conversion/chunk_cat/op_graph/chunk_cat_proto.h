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
 * \file chunk_cat_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_CHUNK_CAT_PROTO_H_
#define OPS_OP_PROTO_INC_CHUNK_CAT_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief split input tensors and concat along one dimension .

* @par Inputs:
* One input:
* x: Dynamic input. A ND Tensor.
*    Must be one of the following types: float32, float16, bfloat16. \n

* @par Attributes:
* @li dim: A required int32, or int64.
  Specifies the dimension along which to chunk. No default value.
* @li num_chunks:  A required int32, or int64.
  Specifies the number of chunks to split "x" into. No default value. \n

* @par Outputs:
* y: A ND Tensor. concat by x . \n

* @attention Constraints:
* @li "x" is a list of at least 2 "tensor" objects of the same type.
* @li "dim" is in the range [-len(x.shape), len(x.shape)].
* @li "num_chunks" is larger than 0. \n

* @par Third-party framework compatibility
* Compatible with the PyTorch operator ChunkCat. \n
*/
REG_OP(ChunkCat)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dim, Int)
    .REQUIRED_ATTR(num_chunks, Int)
    .OP_END_FACTORY_REG(ChunkCat)
} // namespace ge

#endif // OPS_OP_PROTO_INC_CHUNK_CAT_PROTO_H_
