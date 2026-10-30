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
 * \file arg_max_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ARG_MAX_V2_H_
#define OPS_OP_PROTO_INC_ARG_MAX_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Returns the index with the largest value across axes of a tensor.

*@par Inputs:
* Two inputs, including:
*@li x: A ND and multi-dimensional tensor of type float16, float32, int32, or int16.
*@li dimension: A Scalar of type int32, specifying the index with the largest value.

*@par Attributes:
*dtype: The output type, either "int32" or "int64". Defaults to "int64".

*@par Outputs:
*y: A ND tensor of type int32 or int64, specifying the index with the largest value. The dimension is one less than that of "x".

*@attention Constraints:
*@li x: If there are multiple maximum values, the index of the first maximum value is used.
*@li The value range of "dimension" is [-dims, dims - 1]. "dims" is the dimension length of "x".

*@par Third-party framework compatibility
* Compatible with TensorFlow operator ArgMax.
*/
REG_OP(ArgMaxV2)
    .INPUT(x, TensorType::NumberType())
    .INPUT(dimension, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_INT32, DT_INT64}))
    .ATTR(dtype, Type, DT_INT64)
    .OP_END_FACTORY_REG(ArgMaxV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ARG_MAX_V2_H_

