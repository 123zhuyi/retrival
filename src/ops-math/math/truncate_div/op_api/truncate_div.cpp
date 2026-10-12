/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "truncate_div.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/aicpu/aicpu_task.h"
#include "op_api/aclnn_check.h"

#include <unordered_map>
#include <unordered_set>

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TruncateDiv);

static const std::unordered_map<op::DataType, std::unordered_set<op::DataType>> DTYPE_MAPPING = {
    {op::DataType::DT_FLOAT, {op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16}},
    {op::DataType::DT_FLOAT16, {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}},
    {op::DataType::DT_BF16, {op::DataType::DT_BF16}},
    {op::DataType::DT_INT8, {op::DataType::DT_INT8}},
    {op::DataType::DT_INT16, {op::DataType::DT_INT16}},
    {op::DataType::DT_INT32, {op::DataType::DT_INT32, op::DataType::DT_FLOAT}},
    {op::DataType::DT_INT64, {op::DataType::DT_INT64}},
    {op::DataType::DT_UINT8, {op::DataType::DT_UINT8}}};

static inline bool IsAiCoreSupport(const aclTensor* self, const aclTensor* other)
{
    auto selfDtype = self->GetDataType();
    auto otherDtype = other->GetDataType();

    auto it = DTYPE_MAPPING.find(selfDtype);
    if (it == DTYPE_MAPPING.end()) {
        return false;
    }

    return it->second.find(otherDtype) != it->second.end();
}

static const aclTensor* TruncateDivAiCore(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                          aclOpExecutor* executor)
{
    L0_DFX(TruncateDivAiCore, self, other, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(TruncateDiv, OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncateDivAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

static const aclTensor* TruncateDivAiCpu(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                         aclOpExecutor* executor)
{
    L0_DFX(TruncateDivAiCpu, self, other, out);
    static internal::AicpuTaskSpace space("TruncateDiv", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(TruncateDiv, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "TruncateDivAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return out;
}

const aclTensor* TruncateDiv(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(TruncateDiv, self, other);
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
                op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    op::DataType outDataType;

    if (self->GetDataType() == other->GetDataType()) {
        outDataType = self->GetDataType();
    } else {
        outDataType = op::DataType::DT_FLOAT;
    }

    auto divOut = executor->AllocTensor(broadcastShape, outDataType);

    if (IsAiCoreSupport(self, other)) {
        return TruncateDivAiCore(self, other, divOut, executor);
    } else {
        return TruncateDivAiCpu(self, other, divOut, executor);
    }
}

} // namespace l0op
