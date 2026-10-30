/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_bias_add_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_grouped_bias_add_grad.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;
static constexpr int64_t GROUP_INDEX_SHAPE = 1;
static constexpr int64_t GRAD_BIAS_SHAPE_SIZE = 2;
static constexpr int64_t INPUT_MAX_GROUP = 2048;

static const std::initializer_list<DataType> group_bias_in_out_dtype_list = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType> group_idx_dtype_list = {op::DataType::DT_INT32, op::DataType::DT_INT64};

static inline bool CheckNotNull(const aclTensor* gradY, const aclTensor* out)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* groupIdxOptional, const aclTensor* out)
{
    // 检查gradY的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradY, group_bias_in_out_dtype_list, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, group_bias_in_out_dtype_list, return false);

    // 检查输入和输出的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(out, gradY->GetDataType(), return false);

    if (groupIdxOptional != nullptr) {
        // 检查groupIdxOptional的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(groupIdxOptional, group_idx_dtype_list, return false);
    }

    return true;
}

static bool CheckShapeValid(const aclTensor* gradY, const aclTensor* groupIdxOptional, const aclTensor* out)
{
    auto gradYDimNum = gradY->GetViewShape().GetDimNum();
    auto groupNum = gradY->GetViewShape().GetDim(0);
    int64_t hNum = 0;

    if (groupIdxOptional == nullptr) {
        if (gradYDimNum != GRAD_Y_SHAPE_NO_GROUP_IDX) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the grad_y of input should be 3D tensor when group_idx is null.");
            return false;
        }

        hNum = gradY->GetViewShape().GetDim(GRAD_Y_SHAPE_NO_GROUP_IDX - 1);
    } else {
        int64_t groupIdxDimNum = groupIdxOptional->GetViewShape().GetDimNum();
        if (gradYDimNum != GRAD_Y_SHAPE_WITH_GROUP_IDX || groupIdxDimNum != GROUP_INDEX_SHAPE) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "the gradY of input should be 2D tensor when groupIdxOptional is not null \
                                    and groupIdxOptional must be 1D tensor.");
            return false;
        }

        auto groupIdxSize = groupIdxOptional->GetViewShape().GetDim(0);
        OP_CHECK(
            groupIdxSize <= INPUT_MAX_GROUP,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "input group_idx shape not support more than %ld, but got %ld.",
                INPUT_MAX_GROUP, groupIdxSize),
            return false);

        hNum = gradY->GetViewShape().GetDim(GRAD_Y_SHAPE_WITH_GROUP_IDX - 1);
        groupNum = groupIdxSize;
    }

    auto outDimNum = out->GetViewShape().GetDimNum();
    OP_CHECK(
        outDimNum == GRAD_BIAS_SHAPE_SIZE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimensions of output should be two, but got %lu.", outDimNum),
        return false);

    auto outGroupNum = out->GetViewShape().GetDim(0);
    auto outHNum = out->GetViewShape().GetDim(1);
    OP_CHECK(
        outDimNum == GRAD_BIAS_SHAPE_SIZE && outGroupNum == groupNum && outHNum == hNum,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The shape of out should be [%ld, %ld], but got [%ld, %ld].", groupNum, hNum,
            outGroupNum, outHNum),
        return false);

    return true;
}

static bool checkAttrValid(int64_t groupIdxType)
{
    if (groupIdxType < 0 || groupIdxType > 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "group_idx_type should be 0 or 1, but current is %ld", groupIdxType);
        return false;
    }
    return true;
}

static void CheckFormatNZ(const aclTensor* x)
{
    if (x == nullptr) { return; }
    op::Format format = x->GetStorageFormat();
    if (format == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of input gets [%s], this format may lead to precision failure",
        op::ToString(format).GetString());
    }
}

static aclnnStatus CheckParams(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradY, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradY, groupIdxOptional, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShapeValid(gradY, groupIdxOptional, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(checkAttrValid(groupIdxType), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入是否为NZ，若为NZ则提示告警
    CheckFormatNZ(gradY);
    CheckFormatNZ(groupIdxOptional);
    CheckFormatNZ(out);
    return ACLNN_SUCCESS;
}

aclnnStatus ExecGroupedBiasAddGradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(gradY, groupIdxOptional, groupIdxType, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (gradY->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // gradY如果非连续，需要转连续
    auto gradYContiguous = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(gradYContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入groupIdxOptional转换成连续的tensor
    auto groupIdxOptionalContiguous =
        groupIdxOptional == nullptr ? nullptr : l0op::Contiguous(groupIdxOptional, uniqueExecutor.get());
    CHECK_RET(groupIdxOptional == nullptr || groupIdxOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子GroupedBiasAddGrad行计算
    auto gradBias =
        l0op::GroupedBiasAddGrad(gradYContiguous, groupIdxOptionalContiguous, groupIdxType, uniqueExecutor.get());
    CHECK_RET(gradBias != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(gradBias, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGroupedBiasAddGrad, DFX_IN(gradY, groupIdxOptional), DFX_OUT(out));
    return ExecGroupedBiasAddGradGetWorkspaceSize(gradY, groupIdxOptional, 0, out, workspaceSize, executor);
}

aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(
    const aclTensor* gradY, const aclTensor* groupIdxOptional, int64_t groupIdxType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGroupedBiasAddGradV2, DFX_IN(gradY, groupIdxOptional, groupIdxType), DFX_OUT(out));
    return ExecGroupedBiasAddGradGetWorkspaceSize(gradY, groupIdxOptional, groupIdxType, out, workspaceSize, executor);
}

aclnnStatus aclnnGroupedBiasAddGrad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnGroupedBiasAddGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnGroupedBiasAddGradV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnGroupedBiasAddGradV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
