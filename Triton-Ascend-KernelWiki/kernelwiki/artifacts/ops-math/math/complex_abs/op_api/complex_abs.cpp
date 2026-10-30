/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "complex_abs.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ComplexAbs);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {op::DataType::DT_COMPLEX32,
                                                                                         op::DataType::DT_COMPLEX64};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
    }
    return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ComplexAbsAiCore(const aclTensor* self, aclTensor* out, DataType Tout, aclOpExecutor* executor)
{
    L0_DFX(ComplexAbsAiCore, self, out, Tout);
    // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将ComplexAbs算子加入任务队列
    // op::OP_TYPE_ComplexAbs是算子的OpType，self是算子的输入，out是算子的输出
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ComplexAbs, OP_INPUT(self), OP_OUTPUT(out), OP_ATTR(Tout));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ComplexAbsAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
    return out;
}

const aclTensor* ComplexAbs(const aclTensor* self, aclOpExecutor* executor)
{
    // 根据推导出的输出shape申请输出tensor
    // 第一个参数是输出shape，第二个参数是输出的dtype
    DataType outType{DataType::DT_FLOAT};
    if (self->GetDataType() == DataType::DT_COMPLEX32) {
        outType = DataType::DT_FLOAT16;
    }
    auto out = executor->AllocTensor(self->GetViewShape(), outType);

    if (IsAiCoreSupport(self)) {
        return ComplexAbsAiCore(self, out, outType, executor);
    } else {
        return nullptr;
    }
}
} // namespace l0op
