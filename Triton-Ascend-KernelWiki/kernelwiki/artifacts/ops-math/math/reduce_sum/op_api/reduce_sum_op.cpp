/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "reduce_sum_op.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ReduceSum);

static const std::initializer_list<op::DataType> AICORE310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> AICORE910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> AICORE610LITE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> AICORE910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16,
    op::DataType::DT_INT64, op::DataType::DT_BOOL};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    // 获取芯片类型，判断是910还是910B
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201:
            return CheckType(self->GetDataType(), AICORE910B_DTYPE_SUPPORT_LIST);
        case NpuArch::DAV_3510:
            return CheckType(self->GetDataType(), ARCH3510_DTYPE_SUPPORT_LIST);
        case NpuArch::DAV_2002:
            return CheckType(self->GetDataType(), AICORE310P_DTYPE_SUPPORT_LIST);
        case NpuArch::DAV_1001:
            return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
        case NpuArch::DAV_3102:
            return CheckType(self->GetDataType(), AICORE610LITE_DTYPE_SUPPORT_LIST);
        default:
            return CheckType(self->GetDataType(), AICORE910_DTYPE_SUPPORT_LIST);
    }
}

// AICORE算子kernel
static const aclTensor* ReduceSumOpAiCore(
    const aclTensor* x, const aclTensor* axes, bool keepDim, bool noopWithEmptyAxes, const aclTensor* out,
    aclOpExecutor* executor)
{
    L0_DFX(ReduceSumOpAiCore, x, axes, keepDim, noopWithEmptyAxes, out);
    auto retAicore =
        ADD_TO_LAUNCHER_LIST_AICORE(ReduceSum, OP_INPUT(x, axes), OP_OUTPUT(out), OP_ATTR(keepDim, noopWithEmptyAxes));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ReduceSumOp ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

// AICPU算子kernel
static const aclTensor* ReduceSumOpAiCpu(
    const aclTensor* x, const aclTensor* axes, bool keepDim, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ReduceSumOpAiCpu, x, axes, keepDim, out);
    if (x->GetDataType() == op::DataType::DT_INT64) {
        static internal::AicpuTaskSpace space("ReduceSum", ge::DEPEND_IN_SHAPE);
        auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
            ReduceSum, OP_ATTR_NAMES({"keep_dims"}), OP_INPUT(x, axes), OP_OUTPUT(out), OP_ATTR(keepDim));
        CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    } else {
        static internal::AicpuTaskSpace space("Sum", ge::DEPEND_IN_SHAPE, true);
        auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
            ReduceSum, OP_ATTR_NAMES({"Tidx", "keep_dims"}), OP_INPUT(x, axes), OP_OUTPUT(out),
            OP_ATTR(axes->GetDataType(), keepDim));
        CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    }
    return out;
}

const aclTensor* ReduceSumOp(const aclTensor* x, const aclIntArray* axes, bool keepDim, aclOpExecutor* executor)
{
    auto axesTensor = executor->ConvertToTensor(axes, op::ToOpDataType(ACL_INT64));
    aclTensor* out = nullptr;

    if (x->GetDataType() == op::DataType::DT_BOOL) {
        out = executor->AllocTensor(op::DataType::DT_INT64, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    } else {
        out = executor->AllocTensor(x->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    }

    // dim为空时，默认保留所有轴
    bool noopWithEmptyAxes = true;
    INFER_SHAPE(ReduceSum, OP_INPUT(x, axesTensor), OP_OUTPUT(out), OP_ATTR(keepDim, noopWithEmptyAxes));
    op::Shape outShape = x->GetViewShape();
    auto count = axes->Size();
    size_t dimNum = outShape.GetDimNum();
    if (keepDim) {
        for (uint64_t i = 0; i < count; i++) {
            int64_t dimIndex = static_cast<int64_t>((*axes)[i]);
            int64_t dimNew = dimIndex >= 0 ? dimIndex : dimIndex + dimNum;
            outShape.SetDim(dimNew, 1);
        }
        out->SetViewShape(outShape);
    }

    if (IsAiCoreSupport(x)) {
        return ReduceSumOpAiCore(x, axesTensor, keepDim, noopWithEmptyAxes, out, executor);
    } else {
        return ReduceSumOpAiCpu(x, axesTensor, keepDim, out, executor);
    }
}
} // namespace l0op
