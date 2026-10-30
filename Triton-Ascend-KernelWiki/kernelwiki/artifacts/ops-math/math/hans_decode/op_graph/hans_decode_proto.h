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
 * \file hans_decode_proto.h
 * \brief
 */
#ifndef OP_PROTO_HANS_DECODE_PROTO_H_
#define OP_PROTO_HANS_DECODE_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Losslessly decompress.
*
* @par Inputs:
* Four inputs, including:
* @li mantissa: Encode mantissa output. Must be one of the following types: bfloat16, float16, float32.
* @li fixed: Compress output part 1. The type must be the same as mantissa.
* @li var: Compress output part 1. The type must be the same as mantissa.
* @li pdf: A tensor of type int32, with shape(1, 256). Exponential bit frequency distribution of input tensor.
*
* @par Attributes:
* @li reshuff: An optional bool, specifying whether reshuff results on continuous memory. 
* True: the result of multi-core compression is not reshuffled; 
* False: the result of multi-core compression is reshuffled in memory. Default to false.
*
* @par Outputs:
* output: A continuous tensor to be decompressed. The type must be the same as mantissa.
*
* @par Third-party framework compatibility
* Compatible with the PyTorch operator HansDecode.
*/
REG_OP(HansDecode)
    .INPUT(mantissa, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(fixed, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(var, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(pdf, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(reshuff, Bool, false)
    .OP_END_FACTORY_REG(HansDecode)
}  // namespace ge

#endif // OP_PROTO_HANS_DECODE_H_