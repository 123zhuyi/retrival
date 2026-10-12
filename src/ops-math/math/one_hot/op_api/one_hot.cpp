/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file one_hot.cpp
 * \brief
 */

#include "one_hot.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(OneHot);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_UINT8, op::DataType::DT_INT32, op::DataType::DT_INT64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    static bool isSimtVersion = IsRegBase();
    if (isSimtVersion) {
        return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_950);
    }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

const aclTensor* OneHot(
    const aclTensor* self, const aclTensor* depth, const aclTensor* onValue, const aclTensor* offValue, int64_t axis,
    aclOpExecutor* executor)
{
    L0_DFX(OneHot, self, depth, onValue, offValue, axis);

    auto out = executor->AllocTensor(onValue->GetDataType(), onValue->GetStorageFormat(), onValue->GetOriginalFormat());
    auto ret = INFER_SHAPE(OneHot, OP_INPUT(self, depth, onValue, offValue), OP_ATTR(axis), OP_OUTPUT(out));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "OneHot InferShape failed.");
        return nullptr;
    }

    if (!IsAiCoreSupport(self)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self dtype %s should be in dtype support list [%s].",
            op::ToString(self->GetDataType()).GetString(), op::ToString(AICORE_DTYPE_SUPPORT_LIST).GetString());
        return nullptr;
    }

    auto retAicore =
        ADD_TO_LAUNCHER_LIST_AICORE(OneHot, OP_INPUT(self, depth, onValue, offValue), OP_ATTR(axis), OP_OUTPUT(out));
    OP_CHECK(
        retAicore == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "OneHotAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}
} // namespace l0op
