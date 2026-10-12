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
 * \file reciprocal_grad_proto.h
 * \brief ReciprocalGrad 算子原型定义（图模式）
 */
#ifndef OPS_OP_PROTO_INC_RECIPROCAL_GRAD_H_
#define OPS_OP_PROTO_INC_RECIPROCAL_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief ReciprocalGrad 算子：计算 reciprocal 函数的梯度
 *
 * 计算公式：z = -y * y * dy
 * 其中：
 * - y: 前向输出（倒数函数的输出值 y = 1/x）
 * - dy: 上游梯度（损失函数对 reciprocal 输出的梯度）
 * - z: 最终梯度（损失函数对 reciprocal 输入的梯度）
 *
 * @par Inputs:
 * - y: 前向输出张量，支持 FLOAT16、FLOAT、BFLOAT16、DOUBLE、COMPLEX32、COMPLEX64、COMPLEX128
 * - dy: 上游梯度张量，shape 与 y 相同，支持 FLOAT16、FLOAT、BFLOAT16、DOUBLE、COMPLEX32、COMPLEX64、COMPLEX128
 *
 * @par Outputs:
 * - z: 梯度计算结果，shape 与 y 相同，dtype 与输入一致
 *
 * @par Third-party framework compatibility:
 * Compatible with TensorFlow operator reciprocal_grad.
 */
#ifndef OPS_PROTO_DEF_RECIPROCALGRAD
#define OPS_PROTO_DEF_RECIPROCALGRAD
REG_OP(ReciprocalGrad)
    .INPUT(y, TensorType(UnaryDataType))
    .INPUT(dy, TensorType(UnaryDataType))
    .OUTPUT(z, TensorType(UnaryDataType))
    .OP_END_FACTORY_REG(ReciprocalGrad)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_RECIPROCAL_GRAD_H_
