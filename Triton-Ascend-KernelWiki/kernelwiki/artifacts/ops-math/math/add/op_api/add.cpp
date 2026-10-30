/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Add);

static const std::initializer_list<DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_BOOL,
    DataType::DT_INT8,  DataType::DT_UINT8,   DataType::DT_INT64};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT8,     DataType::DT_UINT8,
    DataType::DT_INT64, DataType::DT_BF16,    DataType::DT_BOOL,  DataType::DT_COMPLEX64};

static const std::initializer_list<DataType> ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT8, DataType::DT_UINT8};

static inline const std::initializer_list<DataType>& GetAiCoreDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_LOGI("AddL0", "curArch is %u", static_cast<uint32_t>(curArch));
    switch (curArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_3102: {
            return ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
        }
    }
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static inline bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

bool IsAddSupportNonContiguous(const aclTensor* self, const aclTensor *other) {
  bool isSupportNonContiguous = IsRegBase();
  return isSupportNonContiguous && IsAiCoreSupport(self) && IsAiCoreSupport(other);
}

// AICORE算子kernel
static const aclTensor* AddAiCore(
    const aclTensor* self, const aclTensor* other, aclTensor* addOut, aclOpExecutor* executor)
{
    L0_DFX(AddAiCore, self, other, addOut);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Add算子加入任务队列
    // Add是算子的OpType，self、other是算子的输入，addOut是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Add, OP_INPUT(self, other), OP_OUTPUT(addOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return addOut;
}

// AICPU算子kernel
static const aclTensor* AddAiCpu(
    const aclTensor* self, const aclTensor* other, aclTensor* addOut, aclOpExecutor* executor)
{
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Add算子加入任务队列
    // Add是算子的OpType，self、other是算子的输入，addOut是算子的输出
    L0_DFX(AddAiCpu, self, other);

    static internal::AicpuTaskSpace space("Add");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Add, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(addOut));
    OP_CHECK(
        ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return addOut;
}

const aclTensor* Add(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    bool isMixDataType = (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_FLOAT) ||
                         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_FLOAT16) ||
                         (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT) ||
                         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_BF16);

    auto addOut = isMixDataType ? executor->AllocTensor(broadcastShape, DataType::DT_FLOAT) :
                                  executor->AllocTensor(broadcastShape, self->GetDataType());
    if (isMixDataType || (IsAiCoreSupport(self) && IsAiCoreSupport(other))) {
        return AddAiCore(self, other, addOut, executor);
    }

    return AddAiCpu(self, other, addOut, executor);
}

const aclTensor* AddInplace(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
            op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    // 校验输出tensor的shape和other tensor一致
    if (broadcastShape != other->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and other broadcastShape [%s] not equal to other shape [%s], do no support inplace from the 'other' tensor!", 
            op::ToString(broadcastShape).GetString(), op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }

    // 校验输出tensor的dtype和other tensor一致
    bool isMixDataType = (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_FLOAT) ||
                         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_FLOAT16) ||
                         (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT) ||
                         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_BF16);
    if (isMixDataType && (other->GetDataType() == DataType::DT_FLOAT16 || other->GetDataType() == DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtype DataType::DT_FLOAT not equal to other dtype [%s], do no support inplace from the 'other' tensor!", 
            op::ToString(other->GetDataType()).GetString());
        return nullptr;
    }

    auto addOut = const_cast<aclTensor *>(other);

    if (isMixDataType || (IsAiCoreSupport(self) && IsAiCoreSupport(other))) {
        return AddAiCore(self, other, addOut, executor);
    }

    return AddAiCpu(self, other, addOut, executor);
}

} // namespace l0op
