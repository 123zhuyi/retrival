/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "slogdet_api.h"

#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Slogdet);

// 仅 fp32 走 AiCore（CP1 锁定）
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

// AiCore kernel：单输入 self，双输出 signOut/logOut
static const std::tuple<const aclTensor*, const aclTensor*> SlogdetAiCore(
    const aclTensor* self, const aclTensor* signOut, const aclTensor* logOut, aclOpExecutor* executor)
{
    L0_DFX(SlogdetAiCore, self, signOut, logOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Slogdet, OP_INPUT(self), OP_OUTPUT(signOut, logOut));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SlogdetAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return std::tuple(nullptr, nullptr));
    return std::tuple(signOut, logOut);
}

const std::tuple<const aclTensor*, const aclTensor*> Slogdet(const aclTensor* self, aclOpExecutor* executor)
{
    L0_DFX(Slogdet, self);

    // 输出 batch 形状 = self.shape[:-2]
    auto shapeVec = ToShapeVector(self->GetViewShape());
    shapeVec.pop_back();
    shapeVec.pop_back();
    op::Shape targetShape;
    ToShape(shapeVec, targetShape);

    auto signOut = executor->AllocTensor(targetShape, self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(signOut != nullptr, std::tuple(nullptr, nullptr));
    auto logOut = executor->AllocTensor(targetShape, self->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(logOut != nullptr, std::tuple(nullptr, nullptr));

    INFER_SHAPE(Slogdet, OP_INPUT(self), OP_OUTPUT(signOut, logOut));

    if (IsAiCoreSupport(self)) {
        return SlogdetAiCore(self, signOut, logOut, executor);
    }
    return std::tuple(nullptr, nullptr);
}
} // namespace l0op
