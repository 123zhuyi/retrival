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
 * \file addr_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ADDR_H_
#define OPS_OP_PROTO_INC_ADDR_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Add x1 and the outer product of x2 and x3.

* @par Inputs:
* @li x1: A 2D tensor with data type float32, float16, bfloat16, int8, uint8, bool, which can be broadcasted.
          Shape after broadcasting must be same with the outer product of x2 and x3.
          Supported format list ["ND"].
* @li x2: A 1D tensor with data type float32, float16, bfloat16, int8, uint8, bool. Supported format list ["ND"].
* @li x3: A 1D tensor with data type float32, float16, bfloat16, int8, uint8, bool. Supported format list ["ND"].
* @li beta: A tensor of data type float32, float16, bfloat16, int8, uint8, bool with only one data.
            Supported format list ["ND"].
* @li alpha: A tensor of data type float32, float16, bfloat16, int8, uint8, bool with only one data.
             Supported format list ["ND"].

* @par Outputs:
* y:  A 2D tensor.

* @par Third-party framework compatibility
* Compatible with the ONNX operator ConstantOfShape.
*/
REG_OP(Addr)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(beta, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8, DT_UINT8, DT_BOOL}))
    .OP_END_FACTORY_REG(Addr)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ADDR_H_
