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
 * \file gcd_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_GCD_H_
#define OPS_OP_PROTO_INC_GCD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Returns x1 and x2 greatest common divisor element-wise.
* Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor, support 1D ~ 8D.
* Must be one of the following types: float, float16, bfloat16, uint8, int8, int16, int32, int64.
* @li x2: A ND Tensor. Inputs may have different dtypes and are promoted by the ACLNN wrapper before the
* owned AiCore Gcd kernel is launched. Support 1D ~ 8D. \n

* @par Outputs:
* y: A ND Tensor. The promoted Gcd result may be cast to the requested output dtype. \n

* @par Third-party framework compatibility
* The Atlas A2/A3 Ascend C path supports float, float16, bfloat16, uint8, int8, int16, int32 and int64.
* Floating finite inputs are truncated toward zero before integer Gcd semantics; mixed dtype inputs retain the
* ACLNN Gcd promote/cast behavior. The existing Ascend 950 path is retained for uint8, int8, int16, int32 and int64.
*/
REG_OP(Gcd)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .DATATYPE(T3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_UINT8, DT_INT8, DT_INT16, DT_INT32, DT_INT64}))
    .OP_END_FACTORY_REG(Gcd)

} // namespace ge

#endif // OPS_OP_PROTO_INC_GCD_H_
