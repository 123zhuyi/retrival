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
 * \file mem_set_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MEM_SET_V2_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MEM_SET_V2_H_

#include "graph/operator_reg.h"
namespace ge {

/**
* @brief Set initial values for memory of input tensor list . \n
* @par Inputs:
* x: A list of input tensors. It's a dynamic input. \n

* @par Outputs:
* x: A list of output tensor objects, with the same address as the input tensor list.It's a dynamic output. \n

* @par Attributes:
* @li values_int: integer values to be set, the default value is 0. \n
* @li values_float: float values to be set, the default value is 0.0. \n
*/
REG_OP(MemSetV2)
    .DYNAMIC_INPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64, DT_BOOL}))
    .DYNAMIC_OUTPUT(
        x, TensorType(
               {DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32,
                DT_UINT64, DT_BOOL}))
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSetV2)

} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
