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
 * \file pad_v3_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PAD_V3_OPS_H_
#define OPS_OP_PROTO_INC_PAD_V3_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Pads a tensor.

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, bfloat16,
* float32, double, int32, uint8, int16, int8, complex64, int64,
* qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, bool,
* hifloat8, float8_e5m2, float8_e4m3fn, float8_e8m0, float4_e2m1, float4_e1m2.
* @li paddings: A Tensor of type int32 or int64, specify the padding sizes.
* The size of paddings should be twice of the x shape size.
* If the type of x is float4_e2m1 or float4_e1m2, paddings values should be even number.
* If the type of x is hifloat8, float8_e5m2, float8_e4m3fn or float8_e8m0,
* paddings values should be non-negative integers.
* @li constant_values: An optional Tensor, dtype same as "x".
* Is used only in "constant" mode.

* @par Attributes:
* @li mode: An optional string, Defaults to "constant", indicates paddings mode,
* support "constant", "reflect", "edge", "symmetric", "circular".
* In constant mode the padded value is constant_values, default 0 while constant_values is null.
* In edge mode the padded value is the border value of input x.
* In reflect mode the padded value do not include the borders,
* while in symmetric mode the padded value do include the borders.
* In circular mode, pads x using circular of the input boundary.
* @li paddings_contiguous: An optional bool value, Defaults to true.
* If true, paddings is arranged as [[leftpad_0, rightpad_0], [leftpad_1, rightpad_1], ...]
* If false, paddings is arranged as [[leftpad_0, leftpad_1, ...], [rightpad_0, rightpad_1, ...]]

* @par Outputs:
* y: A Tensor of the same type as "x".
* y.shape[i] = x.shape[i] + leftpad_i + rightpad_i, where y.shape[i] >= 0.

* @attention Constraints:
* "symmetric" and "circular" mode is supported since arch35. \n
* "symmetric" mode: the leftpad_i and rightpad_i should be in [-x.shape[i], x.shape[i]]. \n
* "reflect" mode: the leftpad_i and rightpad_i should be in [-x.shape[i], x.shape[i]). \n
* "constant" mode: the leftpad_i and rightpad_i should be greater than or equal to -x.shape[i]. \n
* "edge" mode: the leftpad_i and rightpad_i should be greater than or equal to -x.shape[i]. \n
* "circular" mode: the leftpad_i and rightpad_i should be in [-x.shape[i], x.shape[i]]. \n

* @par Third-party framework compatibility:
* Compatible with ONNX operator Pad.
*/
REG_OP(PadV3)
    .INPUT(x, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .INPUT(paddings, TensorType::IndexNumberType())
    .OPTIONAL_INPUT(constant_values, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                                                 DT_FLOAT8_E4M3FN, DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .OUTPUT(y, TensorType({TensorType::BasicType(), DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT8_E8M0, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2}))
    .ATTR(mode, String, "constant")
    .ATTR(paddings_contiguous, Bool, true)
    .OP_END_FACTORY_REG(PadV3)

} // namespace ge

#endif
