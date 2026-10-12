/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_RANGE_H_
#define OPS_OP_PROTO_INC_RANGE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Creates a sequence of numbers . \n

* @par Inputs:
* Three inputs, including:
* @li start: A 0D tensor (scalar). Acts as first entry in the range if "limit"
*   is not "None"; otherwise, acts as range limit and first entry defaults to "0".
*   The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND.
* @li limit: A 0D tensor (scalar). Upper limit of sequence, exclusive. If "None",
*   defaults to the value of "start" while the first entry of the range
*   defaults to "0". The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND.
* @li delta: A 0D tensor (scalar). Number that increments "start".
*   Defaults to "1". The supported types are:float16, float32, int32, double, int64, bfloat16, format supports ND. \n
*
* @par Outputs:
* y: A 1D tensor which is the sequence of numbers, format supports ND.
*    The supported types are:float16, float32, int32, double, int64, bfloat16. \n
*    The types of start/limit/delta and output should be the same when the dtypes are:float16, bfloat16, int64. \n
*    The auto inferred type of output is the same as input when the types of start/limit/delta are the same,
*    otherwise the auto inferred type of output is float32. \n
*    The double type of y is not supported in Ascend950 AI Processor. \n
*    If the input parameters start, limit, delta are all of type double,
*    the output parameters y is only supported as float32 in Ascend950 AI Processor. \n
*
* @par Attributes:
* is_closed: An optional attribute of type bool, inducating upper limit is closed or not.
* If true, upper limit is closed. If false, upper limit is opened. The default value is false.
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Range or PyTorch operator Range/Arange.
*/
REG_OP(Range)
    .INPUT(start, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(limit, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .INPUT(delta, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_DOUBLE, DT_INT64, DT_BF16}))
    .ATTR(is_closed, Bool, false)
    .OP_END_FACTORY_REG(Range)

} // namespace ge

#endif // OPS_OP_PROTO_INC_RANGE_H_
