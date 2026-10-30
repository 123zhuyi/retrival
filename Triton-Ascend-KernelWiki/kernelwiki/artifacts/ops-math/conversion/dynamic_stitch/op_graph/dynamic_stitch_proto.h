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
 * \file dynamic_stitch_proto.h
 * \brief
 */
#ifndef OP_PROTO_DYNAMIC_STITCH_H_
#define OP_PROTO_DYNAMIC_STITCH_H_

#include <algorithm>
#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge
{
/**
 * @brief Interleaves the values from the "x" tensors into a single tensor.
 * @par Inputs:
 * @li indices: A tensor list containing "N" tensors. Format supports ND, and data type must be DT_INT32.
 * @li x: A tensor list containing "N" tensors. 
 * Each x[i].shape must start with the corresponding indices[i].shape, and the rest of x[i].shape must be constant.
 * That is, we must have x[i].shape = indices[i].shape + constant.
 * Format supports ND, and data type must be one of the follow types:
 * DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, 
 * DT_FLOAT, DT_DOUBLE, DT_QINT32, DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, DT_COMPLEX128.
 * @par Outputs:
 * y: A Tensor with the same data type and format as "x". 
 * Its shape is [max(indices) + 1] + constant, where constant is x[i].shape - indices[i].shape. 
 * @par Attributes:
 * N: An optional attribute of type int, specifying the length of "indices" and "x", which must meet "N" >= 1. Defaults to "1". 
 * @attention Constraints:
 * On Atlas A2 Training Series Product/ Atlas 800I A2 Inference Product and Atlas A3 Training Series Product, or "N" > 64, 
 * DynamicStitch will run on the Ascend AI CPU, which delivers poor performance.
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator DynamicStitch.
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DynamicStitch)
    .DYNAMIC_INPUT(indices, TensorType({DT_INT32}))
    .DYNAMIC_INPUT(x, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, \
        DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, \
        DT_QINT32, DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, \
        DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32,\
        DT_INT64, DT_UINT64, DT_BOOL, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_DOUBLE, \
        DT_QINT32, DT_QUINT8, DT_QINT8, DT_STRING, DT_COMPLEX64, \
        DT_COMPLEX128}))
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(DynamicStitch)

} // namespace ge
#endif // OP_PROTO_DYNAMIC_STITCH_H_
