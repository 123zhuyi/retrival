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
 * \file tensor_redirect_proto.h
 * \brief
 */
#ifndef OP_PROTO_TENSOR_REDIRECT_H_
#define OP_PROTO_TENSOR_REDIRECT_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Copy data from x to output_x.

* @par Inputs:
* One input, including:
* x: A ND Tensor. Must be one of the following types:
     bfloat16, float16, float32, int8, uint8, int16, uint16, int32, uint32, int64, uint64. \n
     Format is ND, Support 1D ~ 8D. \n

* @par Outputs:
* output_x: A ND Tensor. Has the same dtype and format as "x". \n

*/
#ifndef OPS_PROTO_DEF_TENSORREDIRECT
#define OPS_PROTO_DEF_TENSORREDIRECT
REG_OP(TensorRedirect)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16, DT_UINT64,
                          DT_UINT32, DT_BF16}))
    .OUTPUT(output_x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT8, DT_INT64, DT_INT16, DT_UINT16,
                                  DT_UINT64, DT_UINT32, DT_BF16}))
    .OP_END_FACTORY_REG(TensorRedirect)
#endif

} // namespace ge
#endif // OP_PROTO_TENSOR_REDIRECT_H_
