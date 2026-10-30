/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "polar.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Polar);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT};

static inline bool IsAiCoreSupport(const aclTensor *abs, const aclTensor *angle)
{
    if (abs->GetDataType() != angle->GetDataType()) {
        return false;
    }
    return CheckType(abs->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static inline const aclTensor *PolarAiCore(const aclTensor *abs, const aclTensor *angle,
    aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(PolarAiCore, abs, angle, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Polar, OP_INPUT(abs, angle), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "Polar ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

const aclTensor *Polar(const aclTensor *abs, const aclTensor *angle, aclOpExecutor *executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(abs->GetViewShape(), angle->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.",
                op::ToString(abs->GetViewShape()).GetString(),
                op::ToString(angle->GetViewShape()).GetString());
        return nullptr;
    }

    auto out = executor->AllocTensor(broadcastShape, op::DataType::DT_COMPLEX64);
    CHECK_RET(out != nullptr, nullptr);
    if (IsAiCoreSupport(abs, angle)) {
        return PolarAiCore(abs, angle, out, executor);
    } else {
        return nullptr;
    }
}
}  // namespace l0op
