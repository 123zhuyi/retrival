/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/pad.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Pad);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16, op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_UINT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_BF16,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_UINT16, op::DataType::DT_INT64, op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,        op::DataType::DT_UINT8,       op::DataType::DT_INT16,
    op::DataType::DT_UINT16,      op::DataType::DT_INT32,       op::DataType::DT_UINT32,
    op::DataType::DT_INT64,       op::DataType::DT_UINT64,      op::DataType::DT_BF16,
    op::DataType::DT_FLOAT16,     op::DataType::DT_FLOAT,       op::DataType::DT_BOOL,
    op::DataType::DT_HIFLOAT8,    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_FLOAT8_E8M0, op::DataType::DT_FLOAT4_E2M1, op::DataType::DT_FLOAT4_E1M2};

inline static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    } else if (IsRegBase()) {
        return CheckType(self->GetDataType(), REGBASE_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

inline const aclTensor* PadAiCore(const aclTensor* self, const aclTensor* paddings, aclTensor* out,
                                  aclOpExecutor* executor)
{
    L0_DFX(PadAiCore, self, paddings, out);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(Pad, OP_INPUT(self, paddings), OP_OUTPUT(out));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(retAicore != ACLNN_SUCCESS, return nullptr,
                                         "Pad add to aicore launch list failed.");
    return out;
}

inline const aclTensor* PadAiCpu(const aclTensor* self, const aclTensor* paddings, aclTensor* out,
                                 aclOpExecutor* executor)
{
    L0_DFX(PadAiCpu, self, paddings, out);
    static internal::AicpuTaskSpace space("Pad");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Pad, OP_ATTR_NAMES(), OP_INPUT(self, paddings), OP_OUTPUT(out));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

const aclTensor* Pad(const aclTensor* self, const aclTensor* paddings, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(self->GetDataType(), self->GetViewFormat(), self->GetViewFormat());
    auto ret = INFER_SHAPE(Pad, OP_INPUT(self, paddings), OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pad InferShape failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(self)) {
        return PadAiCore(self, paddings, out, executor);
    } else {
        return PadAiCpu(self, paddings, out, executor);
    }
}
} // namespace l0op
