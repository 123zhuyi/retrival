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
 * \file stateless_normal_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STATELESS_NORMAL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STATELESS_NORMAL_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Outputs deterministic pseudorandom values from a normal distribution,
* with GPU-parity (same seed+offset produces same sequence as CUDA). \n

* @par Inputs:
* @li shape: 1-D. The shape of the output tensor. Must be one of the following types: int64.
* @li seed: 0-D. Seed for the Philox4x32-10 RNG algorithm. Must be one of the following types: int64.
* @li offset: 0-D. Offset for the Philox4x32-10 RNG algorithm. Must be one of the following types: int64.
* @li mean: Scalar or tensor. Mean of the normal distribution. Must be one of the following types: float, float16, bfloat16. Only the 0th element is used for calculation if a tensor is input.
* @li std: Scalar or tensor. Standard deviation of the normal distribution. Must be one of the following types: float, float16, bfloat16. Only the 0th element is used for calculation if a tensor is input. \n

* @par Attributes:
* dtype: Output data type. Must be one of the following types: float16, bfloat16, float32.
* Defaults to float32. \n

* @par Outputs:
* y: Returns random values with specified shape.
* Must be one of the following types: float16, bfloat16, float32. \n

* @par Third-party framework compatibility
* Compatible with PyTorch torch.normal (stateless, GPU-parity mode).
*/
REG_OP(StatelessNormal)
    .INPUT(shape, TensorType({DT_INT64}))
    .INPUT(seed, TensorType({DT_INT64}))
    .INPUT(offset, TensorType({DT_INT64}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(std, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(dtype, Type, DT_FLOAT)
    .OP_END_FACTORY_REG(StatelessNormal)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_STATELESS_NORMAL_H_
