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
 * \file zero_op.cpp
 * \brief experimental 自包含 ZerosLike L0 接口：AICORE（910b/950）与 AICPU 兜底分发。
 */
#include "zero_op.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ZerosLike);

// 本 experimental 实现仅在 ascend910b（DAV_2201）注册 AI Core kernel（见 op_host/zeros_like_def.cpp
// 的 AddConfig("ascend910b")）。其余 SoC 无配套 kernel binary，统一回退 AICPU，避免 kernel-not-found。
static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static constexpr size_t MAX_DIM_LEN = 8;

// 仅 ascend910b 走 AI Core；其余 SoC 无配套 kernel，返回 false 由调用方回退 AICPU。
static bool IsAiCoreSupport(const aclTensor* self)
{
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        return false;
    }
    return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor* ZerosLikeAiCore(const aclTensor* self, aclTensor* zerosLikeOut, aclOpExecutor* executor)
{
    L0_DFX(ZerosLikeAiCore, self);
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ZerosLike, OP_INPUT(self), OP_OUTPUT(zerosLikeOut));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ZerosLike ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return zerosLikeOut;
}

static bool CheckDim(const aclTensor* self)
{
    // self的数据维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    return true;
}

// AICPU算子kernel
static const aclTensor* ZerosLikeAiCpu(const aclTensor* self, aclTensor* zerosLikeOut, aclOpExecutor* executor)
{
    L0_DFX(ZerosLikeAiCpu, self);
    if (!CheckDim(self)) {
        return nullptr;
    }

    static internal::AicpuTaskSpace space("ZerosLike", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(ZerosLike, OP_ATTR_NAMES(), OP_INPUT(self), OP_OUTPUT(zerosLikeOut));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return zerosLikeOut;
}

const aclTensor* ZerosLike(const aclTensor* self, aclOpExecutor* executor)
{
    auto zerosLikeOut = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    if (zerosLikeOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ZerosLike out tensor alloc failed");
        return nullptr;
    }
    if (zerosLikeOut->IsEmpty()) {
        return zerosLikeOut;
    }
    if (IsAiCoreSupport(self)) {
        return ZerosLikeAiCore(self, zerosLikeOut, executor);
    } else {
        return ZerosLikeAiCpu(self, zerosLikeOut, executor);
    }
}

} // namespace l0op
