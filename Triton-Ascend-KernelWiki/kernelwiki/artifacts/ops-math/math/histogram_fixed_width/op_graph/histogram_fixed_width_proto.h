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
 * \file histogram_fixed_width_proto.h
 * \brief
 */
#ifndef OP_PROTO_HISTOGRAM_FIXED_WIDTH_PROTO_H_
#define OP_PROTO_HISTOGRAM_FIXED_WIDTH_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Return histogram of x into nbins equally spaced bins.

*@par Inputs:
* @li x: A Tensor of type float16, float32, int32, int64.
* @li range: A 1-D Tensor of shape [2] with the same dtype as x, containing [min, max].
* @li nbins: A scalar Tensor of type int32, the number of histogram bins. \n

*@par Outputs:
* y: A 1-D Tensor of type int32 with shape [nbins], the histogram result. \n

*@par Constraints:
* If min == max, the operator will adjust min to min + 1 to trigger HistogramV2 error. \n

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator HistogramFixedWidth.
*/
REG_OP(HistogramFixedWidth)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(range, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64}))
    .INPUT(nbins, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .ATTR(dtype, Int, 3)
    .OP_END_FACTORY_REG(HistogramFixedWidth);

} // namespace ge

#endif // OP_PROTO_HISTOGRAM_FIXED_WIDTH_PROTO_H_
