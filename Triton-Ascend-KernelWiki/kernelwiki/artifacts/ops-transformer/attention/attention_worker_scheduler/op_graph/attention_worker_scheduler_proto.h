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
 * \file attention_worker_scheduler_proto.h
 * \brief
 */

#ifndef ATTENTION_WORKER_SCHEDULER_PROTO_H
#define ATTENTION_WORKER_SCHEDULER_PROTO_H

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Data Scanning Operator for the Attention Side in Decoupled Attention-FFN Deployment Scenarios. This operator
polls data from FFNToAttention to confirm its readiness state.
*@par Inputs:
*@li  schedule_context: A tensor. The type support int8. Its shape must be 1D (1024). Format: ND

*@par Outputs:
*schedule_context: A ND Tensor that holds the new value of schedule_context after the value has been assigned.
*Has the same shape and dtype and format as the input "schedule_context". \n
*/
REG_OP(AttentionWorkerScheduler)
    .INPUT(schedule_context, TensorType({DT_INT8}))
    .OUTPUT(schedule_context, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(AttentionWorkerScheduler)
} // namespace ge

#endif // ATTENTION_WORKER_SCHEDULER_PROTO_H
