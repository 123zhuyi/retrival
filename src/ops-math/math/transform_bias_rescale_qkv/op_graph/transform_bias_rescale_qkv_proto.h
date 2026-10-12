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
 * \file transform_bias_rescale_qkv_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_TRANSFORM_BIAS_RESCALE_QKV_OPS_H_
#define OPS_OP_PROTO_INC_TRANSFORM_BIAS_RESCALE_QKV_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief In Multi-Head Attention(MHA) computation, bias and rescale qkv tensor.
* @par Inputs:
* @li qkv:  A 3D tensor. Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, token, 3 * num_heads * dim_per_head).
* @li qkv_bias: A 1D tensor. Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (3 * num_heads * dim_per_head).
* @par Outputs:
* @li q: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @li k: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @li v: A 4D tensor.Type is:BFloat16, Float16 or Float32. Format support
      ND. shape is (batch, num_heads,token, dim_per_head).
* @par Attributes:
      num_heads: head nums. Type is Int64.
*/
REG_OP(TransformBiasRescaleQkv)
    .INPUT(qkv, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(qkv_bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(q, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(k, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_heads, Int)
    .OP_END_FACTORY_REG(TransformBiasRescaleQkv)

} // namespace ge

#endif