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
 * \file pad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PAD_OPS_H_
#define OPS_OP_PROTO_INC_PAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Pad a tensor.

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types: bfloat16, float16,
*     float32, double, int32, uint8, int16, int8, complex64, int64, qint8,
*     quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, bool,
*     hifloat8, float8_e5m2, float8_e4m3fn, float8_e8m0, float4_e2m1, float4_e1m2. Supported format list ["ND"].
* @li paddings: A Tensor of type int32 or int64. Supported format list ["ND"]. \n

* @par Outputs:
* y: A Tensor of the same type as "x". Supported format list ["ND"]. \n

* @li Due to different architectures, the calculation results of this operator
* on NPU and CPU may be inconsistent. \n

* @attention Constraints:
* If the type of x is float4_e2m1 or float4_e1m2, paddings values should be even number.
* If the type of x is hifloat8, float8_e5m2, float8_e4m3fn or float8_e8m0,
* paddings values should be non-negative integers. \n

* @par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*/
REG_OP(Pad)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OP_END_FACTORY_REG(Pad)

} // namespace ge

#endif
