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
 * \file mod_proto.h
 * \brief
 */

#ifndef MOD_PROTO_H_
#define MOD_PROTO_H_
#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Returns element-wise remainder of division. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND tensor. Must be one of the following types: bfloat16, float16, float32,
* int32, int64, int8, uint8, double.
* @li x2: A ND tensor of the same dtype as "x1". \n

* @par Outputs:
* y: A ND tensor. Has the same dtype as "x1". \n

* @attention Constraints:
* @li x2: The input data does not support 0.
* @li When NUM exceeds 2048 , the accuracy of operator cannot guarantee the
* requirement of double thousandths in the mini form.
* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent.
* @li If shape is expressed as (D1,D2... ,Dn),
* then D1*D2... *DN<=1000000,n<=8. \n

* @par Third-party framework compatibility:
* Compatible with the TensorFlow operator Mod.
*/
REG_OP(Mod)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT8, DT_UINT8,
                           DT_INT64, DT_DOUBLE, DT_BF16}))
    .OP_END_FACTORY_REG(Mod)
} // namespace ge

#endif