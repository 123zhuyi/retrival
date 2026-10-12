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
 * \file get_dynamic_dims_proto.h
 * \brief Definition of the GetDynamicDims operator.
 */
#ifndef GET_DYNAMIC_DIMS_PROTO_H_
#define GET_DYNAMIC_DIMS_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Get dynamic dims after GetNext. \n

*@par Inputs:
*input: A nested structure of Tensor objects, from GetNext's output. Must be one of the following types: int32, int64.
\n

*@par Attributes:
*@li shape_info: GE shape_info for each inputs, -1 means unknown dim.
*@li N: A int that indicates the inputs number. \n

*@par Outputs:
*dims: GE unknown dims, a vector of int32 or int64. \n
*/
#ifndef OPS_PROTO_DEF_GETDYNAMICDIMS
#define OPS_PROTO_DEF_GETDYNAMICDIMS
REG_OP(GetDynamicDims)
    .DYNAMIC_INPUT(input, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(dims, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(shape_info, ListInt)
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(GetDynamicDims)
#endif
} // namespace ge
#endif // GET_DYNAMIC_DIMS_PROTO_H_
