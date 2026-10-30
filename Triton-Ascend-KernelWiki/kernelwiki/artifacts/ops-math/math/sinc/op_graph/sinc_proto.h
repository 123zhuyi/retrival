/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sinc_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_SINC_H_
#define OPS_OP_PROTO_INC_SINC_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes sinc of "x" element-wise.

* @par Inputs:
* One input: \n
* x: An ND Tensor that supports the data type UnaryDataType. \n

* @par Outputs:
* y: An ND Tensor with the same dtype and shape of input "x". \n

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Sinc.
*/
REG_OP(Sinc)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(Sinc)
} // namespace ge

#endif // OPS_OP_PROTO_INC_SINC_H_
