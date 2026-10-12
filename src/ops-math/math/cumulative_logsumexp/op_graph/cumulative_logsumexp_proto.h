/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumulative_logsumexp_proto.h
 * \brief
 */

#ifndef CUMULATIVE_LOGSUMEXP_PROTO_H_
#define CUMULATIVE_LOGSUMEXP_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
 * @brief Computes the cumulative log-sum-exp of the tensor "x" along "axis".
 *
 * @par Inputs:
 * Two inputs, including:
 * @li x: A Tensor. Must be one of the following types: float32, float16.
 * @li axis: A Tensor of type int32, int64 or int16. Specifies the dimension for accumulation.
 *
 * @par Attributes:
 * @li exclusive: A bool. Defaults to "false". If "true", performs exclusive cumulative log-sum-exp.
 * @li reverse: A bool. Defaults to "false". If "true", accumulates from the end of the tensor.
 *
 * @par Outputs:
 * y: A Tensor. Has the same type and shape as "x".
 */
#ifndef OPS_PROTO_DEF_CUMULATIVELOGSUMEXP
#define OPS_PROTO_DEF_CUMULATIVELOGSUMEXP
REG_OP(CumulativeLogsumexp)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(axis, TensorType({DT_INT32, DT_INT64, DT_INT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .ATTR(exclusive, Bool, false)
    .ATTR(reverse, Bool, false)
    .OP_END_FACTORY_REG(CumulativeLogsumexp)
#endif // OPS_PROTO_DEF_CUMULATIVELOGSUMEXP
} // namespace ge

#endif
