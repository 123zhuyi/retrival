/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_left_shift.h"
#include "left_shift.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t MAX_INPUT_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,  op::DataType::DT_INT16,  op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    return DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* shiftBits, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(shiftBits, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* shiftBits, aclTensor* out)
{
    const auto& dTypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dTypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(shiftBits, dTypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dTypeSupportList, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* shiftBits, aclTensor* out)
{
    const int64_t selfDim = self->GetViewShape().GetDimNum();
    const int64_t shiftBitsDim = shiftBits->GetViewShape().GetDimNum();
    OP_CHECK(
        selfDim <= MAX_INPUT_DIM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dim num should be less than or equal to 8."),
        return false);
    OP_CHECK(
        shiftBitsDim <= MAX_INPUT_DIM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ShiftBits dim num should be less than or equal to 8."), return false);
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, shiftBits, broadcastShape, return false);
    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormatValid(const aclTensor* input)
{
    if (IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* shiftBits, aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, shiftBits, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, shiftBits, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, shiftBits, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(shiftBits), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckPromoteType(
    const aclTensor* self, const aclTensor* shiftBits, aclTensor* out, op::DataType promoteType)
{
    // 检查self和shiftBits能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype [%s] and shiftBits dtype [%s] can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(shiftBits->GetDataType()).GetString());
        return false;
    }
    // 检查推导后的数据类型是否能转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckNotNullScalar(const aclTensor* self, const aclScalar* shiftBits, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(shiftBits, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValidScalar(const aclTensor* self, const aclScalar* shiftBits, aclTensor* out)
{
    const auto& dTypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dTypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(shiftBits, dTypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dTypeSupportList, return false);
    return true;
}

static bool CheckShapeScalar(const aclTensor* self, aclTensor* out)
{
    const int64_t selfDim = self->GetViewShape().GetDimNum();
    OP_CHECK(
        selfDim <= MAX_INPUT_DIM, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dim num should be less than or equal to 8."),
        return false);
    const auto selfShape = self->GetViewShape();
    const auto outShape = out->GetViewShape();
    OP_CHECK(
        selfShape == outShape, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self shape should be the same as out."), return false);
    return true;
}

static aclnnStatus CheckParamsScalar(const aclTensor* self, const aclScalar* shiftBits, aclTensor* out)
{
    CHECK_RET(CheckNotNullScalar(self, shiftBits, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidScalar(self, shiftBits, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeScalar(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckPromoteTypeScalar(
    const aclTensor* self, const aclScalar* shiftBits, aclTensor* out, op::DataType promoteType)
{
    // 检查self和shiftBits能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype [%s] and shiftBits dtype [%s] can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(shiftBits->GetDataType()).GetString());
        return false;
    }
    // 检查推导后的数据类型是否能转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

aclnnStatus aclnnLeftShiftsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* shiftBits, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLeftShifts, DFX_IN(self, shiftBits), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数dtype检查
    auto ret = CheckParamsScalar(self, shiftBits, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        OP_LOGD("The self is empty, skip LeftShift.");
        return ACLNN_SUCCESS;
    }
    //
    auto promoteType = op::PromoteType(self->GetDataType(), shiftBits->GetDataType());
    CHECK_RET(CheckPromoteTypeScalar(self, shiftBits, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入shiftBits转换为数据类型为promoteType的tensor
    const aclTensor* shiftBitsdTensor = (uniqueExecutor.get())->ConvertToTensor(shiftBits, promoteType);
    CHECK_RET(shiftBitsdTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shiftBitsdBroadcast = shiftBitsdTensor;
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(out->GetViewShape());
    auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (shiftBitsdBroadcast->GetViewShape() != out->GetViewShape()) {
        shiftBitsdBroadcast = l0op::BroadcastTo(shiftBitsdTensor, broadcastShapeArray, uniqueExecutor.get());
        CHECK_RET(shiftBitsdBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outShifted = l0op::LeftShift(selfCasted, shiftBitsdBroadcast, uniqueExecutor.get());
    CHECK_RET(outShifted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto outCasted = l0op::Cast(outShifted, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果outCasted拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(outCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeftShifts(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLeftShifts);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnLeftShiftGetWorkspaceSize(
    const aclTensor* self, const aclTensor* shiftBits, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLeftShift, DFX_IN(self, shiftBits), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数dtype检查
    auto ret = CheckParams(self, shiftBits, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || shiftBits->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        OP_LOGD("The self or shiftBits is empty, skip LeftShift.");
        return ACLNN_SUCCESS;
    }
    //
    auto promoteType = op::PromoteType(self->GetDataType(), shiftBits->GetDataType());
    CHECK_RET(CheckPromoteType(self, shiftBits, out, promoteType), ACLNN_ERR_PARAM_INVALID);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shiftBitsContiguous = l0op::Contiguous(shiftBits, uniqueExecutor.get());
    CHECK_RET(shiftBitsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto shiftBitsdCasted = l0op::Cast(shiftBitsContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(shiftBitsdCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfBroadcast = selfCasted;
    auto shiftBitsdBroadcast = shiftBitsdCasted;
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(out->GetViewShape());
    auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (selfBroadcast->GetViewShape() != out->GetViewShape()) {
        selfBroadcast = l0op::BroadcastTo(selfBroadcast, broadcastShapeArray, uniqueExecutor.get());
        CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (shiftBitsdBroadcast->GetViewShape() != out->GetViewShape()) {
        shiftBitsdBroadcast = l0op::BroadcastTo(shiftBitsdCasted, broadcastShapeArray, uniqueExecutor.get());
        CHECK_RET(shiftBitsdBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outShifted = l0op::LeftShift(selfBroadcast, shiftBitsdBroadcast, uniqueExecutor.get());
    CHECK_RET(outShifted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转换成输出out的数据类型
    auto outCasted = l0op::Cast(outShifted, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果outCasted拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(outCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeftShift(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLeftShift);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
