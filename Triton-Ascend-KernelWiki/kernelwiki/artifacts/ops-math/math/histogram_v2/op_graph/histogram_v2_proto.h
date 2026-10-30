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
#ifndef OP_PROTO_HISTOGRAM_V2_PROTO_H_
#define OP_PROTO_HISTOGRAM_V2_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
namespace ge {
/**
* @brief Computes the histogram of a tensor.

*@par Inputs:
* @li x: A tensor of type float16, float32, int64, int32, int16, int8, uint8.
* @li min: A tensor of type float16, float32, int64, int32, int16, int8, uint8, with only one element.
* @li max: A tensor of type float16, float32, int64, int32, int16, int8, uint8, with only one element.
*@par Attributes:

 * bins: Optional. Type Must be int64. Value must be greater than 0, Defaults to 100.
 *@li y_dtype: A Type, The output type, either "int32" or "float". Defaults to "int32". \n

 *@par Outputs:
 * y: A tensor of type int32 or float32 .

*@par Third-party framework compatibility
* Compatible with the Pytorch operator Histc.
*/

REG_OP(HistogramV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .INPUT(min, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .INPUT(max, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8}))
    .OUTPUT(y, TensorType({DT_INT32, DT_FLOAT}))
    .ATTR(bins, Int, 100)
    .ATTR(y_dtype, Type, DT_INT32)
    .OP_END_FACTORY_REG(HistogramV2);

}  // namespace ge


#endif  //OP_PROTO_HISTOGRAM_V2_PROTO_H_