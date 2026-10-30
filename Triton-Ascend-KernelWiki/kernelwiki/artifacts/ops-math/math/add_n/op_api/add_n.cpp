/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_n.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AddN);

static const std::initializer_list<DataType> AICORE910B_AND_AICORE91093_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_INT64, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT
};

// 根据dtype判断是否支持AICORE
static inline bool IsAiCoreSupport(const aclTensorList *tensors)
{
    op::DataType data_type = (*tensors)[0]->GetDataType();
    if (!CheckType(data_type, AICORE910B_AND_AICORE91093_DTYPE_SUPPORT_LIST)) {
        return false;
    }
    return true;
}

// AICORE算子kernel
static inline const aclTensor *AddNAiCore(const aclTensorList *tensors, aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(AddNAiCore, tensors, out);
    int32_t n = tensors->Size();
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AddN, OP_INPUT(tensors), OP_OUTPUT(out), OP_ATTR(n));
    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddNAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

// AICPU算子kernel
static inline const aclTensor *AddNAiCpu(const aclTensorList *tensors, aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(AddNAiCpu, tensors, out);
    int32_t n = tensors->Size();
    static internal::AicpuTaskSpace space("AddN");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(AddN, OP_ATTR_NAMES({"N"}), OP_INPUT(tensors), OP_OUTPUT(out), OP_ATTR(n));

    OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AddNAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
    return out;
}

const aclTensor *AddN(const aclTensorList *tensors, aclOpExecutor *executor)
{
    auto out = executor->AllocTensor((*tensors)[0]->GetViewShape(), (*tensors)[0]->GetDataType());
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Alloc out tensor failed.");
        return nullptr;
    }

    if (IsAiCoreSupport(tensors)) {
        return AddNAiCore(tensors, out, executor);
    } else {
        return AddNAiCpu(tensors, out, executor);
    }
}

} // namespace l0op
