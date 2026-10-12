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
 * \file bias_add_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_BIAS_ADD_H_
#define OPS_OP_PROTO_INC_BIAS_ADD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Adds 'bias' to 'x'. Support broadcasting operations.

*@par Inputs:
*Two inputs, including:
* @li x: A ND tensor of type NumberType, format list [ND, NCHW, NHWC, NCDHW, NDHWC].
*Must be one of the following types: float32, float64, int32, uint8, int16,
*int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
* @li bias: A 1D tensor with size the C dimension of x:
*when x format is NCHW or NCDHW, C dimension is x.shape[1]. \n
*When x format is NHWC or NDHWC, C dimension is x.shape[-1]. \n
*when x format is ND and data_format is in [NCHW, NCDHW], C dimension is x.shape[1]. \n
*when x format is ND and data_format is in [NHWC, NDHWC], C dimension is x.shape[-1]. \n

*@par Attributes:
*data_format: An optional string. Defaults to "NHWC". \n

*@par Outputs:
*y: A ND tensor with same type and shape and format as "x". \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator BiasAdd.
*/
REG_OP(BiasAdd)
    .INPUT(x, TensorType::NumberType())
    .INPUT(bias, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(BiasAdd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_BIAS_ADD_H_
