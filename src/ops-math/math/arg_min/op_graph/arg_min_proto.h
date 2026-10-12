/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_min_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ARG_MIN_H_
#define OPS_OP_PROTO_INC_ARG_MIN_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the index with the smallest value across dimensions of a tensor.

*@par Inputs:
*Two inputs, including:
*@li x: A ND Tensor. Must be one of the following types: float32, float64, int32, uint8, int16, int8, complex64, int64, qint8, quint8, qint32, bfloat16, uint16, complex128, float16, uint32, uint64.
*format is ND.
*@li dimension: A 1D ND tensor or scalar, data type is int32 or int64 and value range is [-rank(input x), rank(input x)].
* Describes which dimension of the input tensor to reduce across.
*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64".

*@par Outputs:
*y: A ND Tensor of type "dtype".

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMin.
*/
REG_OP(ArgMin)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMin)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ARG_MIN_H_

