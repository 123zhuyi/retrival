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
 * \file hans_encode_proto.h
 * \brief
 */
#ifndef OP_PROTO_HISTOGRAM_PROTO_H_
#define OP_PROTO_HISTOGRAM_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Computes the histogram of a tensor.

*@par Inputs:
* x: A Tensor of type float16,float32,int64,int32,int16,int8,uint8. \n

*@par Attributes:

* @li bins: Optional. Must be one of the following types: int32. Defaults to 100.
* @li min: Optional. Must be one of the following types: float32. Defaults to 0.0.
* @li max: Optional. Must be one of the following types: float32. Defaults to 0.0. \n

*@par Outputs:
* y: A Tensor. A Tensor of type float32,int64,int32,int16,int8,uint8 . \n

* @attention Constraints:
* The operator will use the interface set_atomic_add(), therefore weights and output should be float32 only. \n

*@par Third-party framework compatibility
* Compatible with the Pytorch operator Histc.
*/
REG_OP(Histogram)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .ATTR(bins, Int, 100)
    .ATTR(min, Float, 0.0)
    .ATTR(max, Float, 0.0)
    .OP_END_FACTORY_REG(Histogram);

}  // namespace ge


#endif  // OP_PROTO_HISTOGRAM_PROTO_H_