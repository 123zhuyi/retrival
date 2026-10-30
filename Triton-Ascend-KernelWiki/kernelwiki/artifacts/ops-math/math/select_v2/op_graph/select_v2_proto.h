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
 * \file select_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SELECT_V2_H_
#define OPS_OP_PROTO_INC_SELECT_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Select elements from "then" or "else", depending on "condition" .

* @par Inputs:
* Three inputs, including:
* @li condition: A tensor of type bool. If condittion is true, outputs will be set as then. If condittion is false, outputs will be set as else.
* @li then: A tensor. Must be one of the following types: float16, float32, double, int8, int16, int32, int64, 
 *uint8, uint16, uint32, uint64, complex64, complex128, bool, bfloat16
* @li else: A tensor of the same type as "then" . \n

* @par Outputs:
* result: A tensor. Has the same type as "then" . \n

* @attention Constraints:
* @li The input tensors of condition, then and else must meet the broadcast relationship.
* @li The shape of result is formed by broadcasting condition, then and else.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SelectV2.
*/
REG_OP(SelectV2)
    .INPUT(condition, TensorType({DT_BOOL}))
    .INPUT(then,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .INPUT(else,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OUTPUT(result,TensorType({DT_COMPLEX128,DT_COMPLEX64,DT_DOUBLE,DT_FLOAT,DT_FLOAT16,DT_INT16,DT_INT32,DT_INT64,DT_INT8,DT_UINT16,DT_UINT32,DT_UINT64,DT_UINT8,DT_BOOL,DT_BF16}))
    .OP_END_FACTORY_REG(SelectV2)


} // namespace ge

#endif // OPS_OP_PROTO_INC_SELECT_V2_H_

