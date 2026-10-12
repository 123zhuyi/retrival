/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lin_space_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_
#define OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Generates values in an interval .

*@par Inputs:
* Three ND inputs, including:
*@li start: A 1D Tensor, for the first entry in the range. 
* Type must be one of the following types:
* float32, float16, double, bfloat16, int32, int16, int8, uint8.
*@li stop: A 1D Tensor, for the last entry in the range.
* Type must be one of the following types:
* float32, float16, double, bfloat16, int32, int16, int8, uint8.
*@li num: A 1D Tensor of type int32 or int64, for the common difference of the entries . \n

*@par Outputs:
*output: A 1D Tensor, Type is same with "start". \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator lin_space.
*/
REG_OP(LinSpace)
    .INPUT(start, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(stop, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .INPUT(num, TensorType::IndexNumberType())
    .OUTPUT(output, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT32, DT_INT16, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(LinSpace)

} // namespace ge

#endif  // OPS_OP_PROTO_INC_LIN_SPACE_OPS_H_