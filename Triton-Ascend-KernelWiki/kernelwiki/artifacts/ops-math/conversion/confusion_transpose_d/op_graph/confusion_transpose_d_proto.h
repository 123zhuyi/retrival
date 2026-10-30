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
 * \file confusion_transpose_d_proto.h
 * \brief
 */

 #ifndef OPS_OP_PROTO_CONFUSION_TRANSPOSE_D_H_
 #define OPS_OP_PROTO_CONFUSION_TRANSPOSE_D_H_

 #include "graph_plugin/operator_reg.h"

namespace ge{

static const std::vector<ge::DataType> confusionTransposeDDataType = {
    ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64,ge::DT_UINT8, 
    ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,ge::DT_FLOAT16, ge::DT_FLOAT, 
    ge::DT_BF16};    

/**
*@brief Fusion of Reshape and Transpose Operations.

*@par Inputs:
*One input, including:
* x: A tensor. Must be one of the following types:
*   int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float, bfloat16. \n

*@par Attributes:
* @li perm: A required listInt. The index of the aixs in the original tensor that corresponds to each axis in the transposed tensor. \n
* @li shape: An required listInt. The shape of the tensor after reshaping.
* @li transpose_first: An required bool. The attribution determines whether to perform the transpose operation first.

*@par Outputs:
* y: A Tensor with the same type and shape of x. \n
*/
REG_OP(ConfusionTransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(perm, ListInt)
    .REQUIRED_ATTR(shape, ListInt)
    .REQUIRED_ATTR(transpose_first, Bool)
    .OP_END_FACTORY_REG(ConfusionTransposeD)
}; // namespace ge

#endif // OPS_OP_PROTO_CONFUSION_TRANSPOSE_D_H_
