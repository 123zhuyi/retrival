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
 * \file broadcast_to_proto.h
 * \brief
 */
#ifndef OP_PROTO_BROADCAST_TO_H_
#define OP_PROTO_BROADCAST_TO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Broadcasts an array for a compatible shape.
 *  Broadcasting is the process of making arrays to have compatible shapes
 *  for arithmetic operations. Two shapes are compatible if for each
 *  dimension pair they are either equal or one of them is one. When trying
 *  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
 *  and works its way forward.
 *
 * @par Inputs:
 * @li x: A tensor, support all dtype include(BasicType, bool, string, hifloat8, float8_e5m2, float8_e4m3fn).
 * @li shape: A tensor.
 *     A 1D tensor of type int32 or int64, for the shape of the desired output.
 *
 * @par Outputs:
 * y: A tensor. Has the same tensor info of "x".
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator BroadcastTo.
 *
 */
REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)

} // namespace ge
#endif // OP_PROTO_BROADCAST_TO_H_
