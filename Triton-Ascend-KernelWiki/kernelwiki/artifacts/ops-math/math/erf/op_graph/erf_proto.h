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
 * \file erf_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_ERF_H_
#define OPS_OP_PROTO_INC_ERF_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Computes the Gauss error function of 'x' element-wise. \n

* @par Inputs:
* x: A Tensor of type bfloat16, float16, float32 or double. the format can be
*    [NCHW,NHWC,ND]

* @par Outputs:
* y: A Tensor. Has the same type, format and shape as 'x'. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Erf.
*/
REG_OP(Erf)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Erf)

} // namespace ge

#endif // OPS_OP_PROTO_INC_ERF_H_
