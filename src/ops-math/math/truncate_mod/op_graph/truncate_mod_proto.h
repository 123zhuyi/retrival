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
 * \file truncate_mod_proto.h
 * \brief
 */
#ifndef TRUNCATE_MOD_PROTO_H_
#define TRUNCATE_MOD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns element-wise remainder of division. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: bfloat16, float16, float32,
* double, int32, int64, int8, uint8.
* @li x2: A ND Tensor of the same dtype as "x1". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype as "x1". \n

* @attention Constraints:
* @li x2: The input data does not support 0
* @li When value of tensor exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in Atlas 200/300/500 Inference Product.
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent
* @li If shape is expressed as (D1,D2... ,Dn), then D1*D2... *DN<=1000000,n<=8

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TruncateMod.
*/
REG_OP(TruncateMod)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT8, DT_UINT8, DT_INT32}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT8, DT_UINT8, DT_INT32}))
    .OP_END_FACTORY_REG(TruncateMod)
} // namespace ge
#endif // TRUNCATE_MOD_PROTO_H_
