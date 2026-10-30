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
 * \file data_compare_proto.h
 * \brief DataCompare GE IR 注册
 */
#ifndef OPS_DATA_COMPARE_PROTO_H_
#define OPS_DATA_COMPARE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Returns the num value of abs(x1-x2) > atol+rtol*abs(x2) element-wise. \n

* @par Inputs:
* @li x1: A tensor. Must be one of the following types: float32, int32, uint8, int8, float16, bfloat16.
* Supported format list ["ND"].
* @li x2: A tensor of the same dtype as "x1". Supported format list ["ND"]. \n

* @par Attributes:
* @li atol: Defaults to "1e-05".
* @li rtol: Defaults to "1e-03". \n

* @par Outputs:
* @li num: A tensor of type float32. Supported format list ["ND"]. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*
*/

REG_OP(DataCompare)
    .INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(num, TensorType({DT_FLOAT}))
    .ATTR(atol, Float, 1e-5f)
    .ATTR(rtol, Float, 1e-3f)
    .OP_END_FACTORY_REG(DataCompare)

} // namespace ge

#endif // OPS_DATA_COMPARE_PROTO_H_
