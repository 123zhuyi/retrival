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
 * \file hans_encode_proto.h
 * \brief
 */
#ifndef OP_PROTO_HANS_ENCODE_PROTO_H_
#define OP_PROTO_HANS_ENCODE_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Losslessly compress the exponent bits of the input tensor.
*
* @par Inputs:
* Two inputs, including:
* @li input_tensor: A continuous tensor to be compressed. Number of elements must be a multiple of 64.
* Must be one of the following types: bfloat16, float16, float32.
* @li pdf: A tensor of type int32, with shape(1, 256). Exponential bit frequency distribution of input tensor.
*
* @par Attributes:
* @li statistic: An optional bool, specifying whether to compute the statistic PDF. 
* True: use the online statistical pdf;
* False: use the input pdf. Defaults to false.
* @li reshuff: An optional bool, specifying whether reshuff compression results on continuous memory. Default to false.
* True: the result of multi-core compression is not reshuffled; 
* False: the result of multi-core compression is reshuffled in memory. Default to false.
*
* @attention Constraints:
* The sum of fixed tensor and var tensor size must be greater than
* (size(input_tensor) + size(input_tensor) / 64 + 8448 * processCoreDim + 512).
*
* @par Outputs:
* @li pdf: An int32 tensor specifying the exponential bit frequency distribution of the input tensor, 
* valid when statistic is true.
* @li mantissa: Mantissa output. The type must be the same as input_tensor.
* @li fixed: Compress output part 1. The type must be the same as input_tensor.
* @li var: Compress output part 2. The type must be the same as input_tensor.
* 
* @par Third-party framework compatibility
* Compatible with the PyTorch operator HansEncode.
*/
REG_OP(HansEncode)
    .INPUT(input_tensor, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(mantissa, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(fixed, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(statistic, Bool, false)
    .ATTR(reshuff, Bool, false)
    .OP_END_FACTORY_REG(HansEncode)

}  // namespace ge


#endif  // OP_PROTO_HANS_ENCODE_PROTO_H_