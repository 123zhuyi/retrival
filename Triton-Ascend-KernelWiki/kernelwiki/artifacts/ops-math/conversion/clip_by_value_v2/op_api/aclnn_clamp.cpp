/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_clamp.h"
#include <climits>
#include <limits>
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "clip_by_value.h"

using namespace op;

static const std::initializer_list<DataType> Ascend910_dtype_support_list = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_INT64,
    DataType::DT_INT16,   DataType::DT_INT8,  DataType::DT_UINT8, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> Ascend910B_dtype_support_list = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_INT32,  DataType::DT_INT64, DataType::DT_INT16,
    DataType::DT_INT8,    DataType::DT_UINT8, DataType::DT_DOUBLE, DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,
    DataType::DT_BF16};

const float NPU_HALF_MAX = std::numeric_limits<float>::infinity();
const float NPU_HALF_MIN = -std::numeric_limits<float>::infinity();
const int64_t NPU_S64_MAX = std::numeric_limits<int64_t>::max(); // 9223372036854775807
const int64_t NPU_S64_MIN = std::numeric_limits<int64_t>::min(); // -9223372036854775808
const double NPU_F64_MAX = std::numeric_limits<double>::infinity();
const double NPU_F64_MIN = -std::numeric_limits<double>::infinity();
static const int NPU_S16_MAX = std::numeric_limits<int16_t>::max(); // 32767
static const int NPU_S16_MIN = std::numeric_limits<int16_t>::min(); // -32768
const int NPU_S8_MAX = std::numeric_limits<int8_t>::max();          // 127
const int NPU_S8_MIN = std::numeric_limits<int8_t>::min();          // -128
const int NPU_U8_MAX = std::numeric_limits<uint8_t>::max();         // 255
const int NPU_U8_MIN = std::numeric_limits<uint8_t>::min();         // 0

static op::DataType InnerTypeToComplexType(const op::DataType input)
{
    switch (input) {
        case op::DataType::DT_BF16:
            // BFloat16 has range equivalent to Float,
            // so we map it to ComplexFloat.
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_FLOAT16:
            return op::DataType::DT_COMPLEX32;
        case op::DataType::DT_FLOAT:
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_DOUBLE:
            return op::DataType::DT_COMPLEX128;
        case op::DataType::DT_COMPLEX32:
            return op::DataType::DT_COMPLEX32;
        case op::DataType::DT_COMPLEX64:
            return op::DataType::DT_COMPLEX64;
        case op::DataType::DT_COMPLEX128:
            return op::DataType::DT_COMPLEX128;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown Complex ScalarType for [%s]", ToString(input).GetString());
            return op::DataType::DT_UNDEFINED;
    }
}

static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower)
{
    if (IsComplexType(higher)) {
        return higher;
    } else if (IsComplexType(lower)) {
        // preserve value type of higher if it is floating type.
        if (IsFloatingType(higher)) {
            return InnerTypeToComplexType(higher);
        }
        // in case of integral input
        // lower complex takes precedence.
        return lower;
    } else if (IsFloatingType(higher)) {
        return higher;
    }
    if (higher == op::DataType::DT_BOOL || IsFloatingType(lower)) {
        return op::PromoteType(higher, lower);
    }
    if (higher != op::DataType::DT_UNDEFINED) {
        return higher;
    }
    return lower;
}

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || IsRegBase(curArch)) {
        return Ascend910B_dtype_support_list;
    } else {
        return Ascend910_dtype_support_list;
    }
}

static bool CheckNotNull(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    if (clipValueMin == nullptr && clipValueMax == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "clamp: At least one of 'min' or 'max' must not be None.");
        return false;
    }
    return true;
}

static bool CheckNotNull(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    if (clipValueMin == nullptr && clipValueMax == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "clamp: At least one of 'min' or 'max' must not be None.");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static bool CheckShape(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    if (clipValueMin != nullptr) {
        OP_CHECK_MAX_DIM(clipValueMin, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    if (clipValueMax != nullptr) {
        OP_CHECK_MAX_DIM(clipValueMax, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, clipValueMin, clipValueMax, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, clipValueMin, clipValueMax, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, clipValueMin, clipValueMax, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckReformat(const aclTensor* input)
{
    input = l0op::ReFormat(input, static_cast<op::Format>(ACL_FORMAT_ND));
    OP_CHECK(
        input != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input tensor with TransData return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static bool CheckSelfCanCastOut(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否可以转化成out
    if (!CanCast(self->GetDataType(), out->GetDataType())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self type %s can't be cast to the output type %s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    return true;
}

const aclTensor* ScalarToTensor(const aclScalar* other, const op::DataType dataType, aclOpExecutor* executor)
{
    auto otherTensor = executor->ConvertToTensor(other, dataType);
    return otherTensor;
}

const aclScalar* NormalizeMaxScalar(const aclScalar* valueMax, const op::DataType dataType, aclOpExecutor* executor)
{
    if (valueMax != nullptr) {
        return valueMax;
    }
    switch (dataType) {
        case DataType::DT_FLOAT:
            return executor->AllocScalar(NPU_HALF_MAX);
        case DataType::DT_INT32:
            return executor->AllocScalar(INT_MAX);
        case DataType::DT_INT64:
            return executor->AllocScalar(NPU_S64_MAX);
        case DataType::DT_DOUBLE:
            return executor->AllocScalar(&NPU_F64_MAX, DataType::DT_DOUBLE);
        case DataType::DT_INT16:
            return executor->AllocScalar(NPU_S16_MAX);
        case DataType::DT_INT8:
            return executor->AllocScalar(NPU_S8_MAX);
        case DataType::DT_UINT8:
            return executor->AllocScalar(NPU_U8_MAX);
        default:
            return executor->AllocScalar(NPU_HALF_MAX);
    }
}

const aclScalar* NormalizeMinScalar(const aclScalar* valueMin, const op::DataType dataType, aclOpExecutor* executor)
{
    if (valueMin != nullptr) {
        return valueMin;
    }
    switch (dataType) {
        case DataType::DT_FLOAT:
            return executor->AllocScalar(NPU_HALF_MIN);
        case DataType::DT_INT32:
            return executor->AllocScalar(INT_MIN);
        case DataType::DT_INT64:
            return executor->AllocScalar(NPU_S64_MIN);
        case DataType::DT_DOUBLE:
            return executor->AllocScalar(&NPU_F64_MIN, DataType::DT_DOUBLE);
        case DataType::DT_INT16:
            return executor->AllocScalar(NPU_S16_MIN);
        case DataType::DT_INT8:
            return executor->AllocScalar(NPU_S8_MIN);
        case DataType::DT_UINT8:
            return executor->AllocScalar(NPU_U8_MIN);
        default:
            return executor->AllocScalar(NPU_HALF_MIN);
    }
}

const aclTensor* NormalizeMaxTensor(const aclTensor* valueMax, const op::DataType dataType, aclOpExecutor* executor)
{
    if (valueMax != nullptr) {
        return l0op::Contiguous(valueMax, executor);
    }
    const aclScalar* scalar = NormalizeMaxScalar(nullptr, dataType, executor);
    return executor->ConvertToTensor(scalar, dataType);
}

const aclTensor* NormalizeMinTensor(const aclTensor* valueMin, const op::DataType dataType, aclOpExecutor* executor)
{
    if (valueMin != nullptr) {
        return l0op::Contiguous(valueMin, executor);
    }
    const aclScalar* scalar = NormalizeMinScalar(nullptr, dataType, executor);
    return executor->ConvertToTensor(scalar, dataType);
}

static bool ClampPromoteType(
    const aclScalar* clipValueMin, const aclScalar* clipValueMax, const aclTensor* out, op::DataType& promoteType)
{
    if ((clipValueMin != nullptr) && (clipValueMax != nullptr)) {
        auto clipValueMinDtype = GetScalarDefaultDtype(clipValueMin->GetDataType());
        auto clipValueMaxDtype = GetScalarDefaultDtype(clipValueMax->GetDataType());
        auto scalarValueDtype = op::PromoteType(clipValueMinDtype, clipValueMaxDtype);
        promoteType = CombineCategoriesWithComplex(promoteType, scalarValueDtype);
    } else if (clipValueMin != nullptr) {
        auto clipValueMinDtype = GetScalarDefaultDtype(clipValueMin->GetDataType());
        promoteType = CombineCategoriesWithComplex(promoteType, clipValueMinDtype);
    } else if (clipValueMax != nullptr) {
        auto clipValueMaxDtype = GetScalarDefaultDtype(clipValueMax->GetDataType());
        promoteType = CombineCategoriesWithComplex(promoteType, clipValueMaxDtype);
    }
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promote type fail.");
        return false;
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

    auto supportList = GetDtypeSupportList();
    if (!CheckType(promoteType, supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s not implemented for %s, should be in dtype support list %s.", "promote dtype",
            op::ToString(promoteType).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    OP_LOGI("clamp tensor scalar process with promote dtype %s.", op::ToString(promoteType).GetString());

    return true;
}

aclnnStatus aclnnClampCommon(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, clipValueMin, clipValueMax, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    op::DataType promoteType = self->GetDataType();
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        CHECK_RET(ClampPromoteType(clipValueMin, clipValueMax, out, promoteType), ACLNN_ERR_PARAM_INVALID);
    } else {
        promoteType = out->GetDataType();
        CHECK_RET(CheckSelfCanCastOut(self, out), ACLNN_ERR_PARAM_INVALID);
    }

    auto value_max = NormalizeMaxScalar(clipValueMax, promoteType, uniqueExecutor.get());
    auto value_min = NormalizeMinScalar(clipValueMin, promoteType, uniqueExecutor.get());

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto minTensor = ScalarToTensor(value_min, promoteType, uniqueExecutor.get());
    auto maxTensor = ScalarToTensor(value_max, promoteType, uniqueExecutor.get());

    auto ceilResult = l0op::ClipByValueV2(selfCasted, minTensor, maxTensor, uniqueExecutor.get());
    CHECK_RET(ceilResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 针对Arch3510芯片进行处理
    if (IsRegBase(curArch)) {
        auto ceilCastResult = l0op::Cast(ceilResult, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(ceilCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(ceilCastResult, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto viewCopyResult = l0op::ViewCopy(ceilResult, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClampGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, const aclScalar* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClamp, DFX_IN(self, clipValueMin, clipValueMax), DFX_OUT(out));
    return aclnnClampCommon(self, clipValueMin, clipValueMax, out, workspaceSize, executor);
}

aclnnStatus aclnnClamp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClamp);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnClampMinGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClampMin, DFX_IN(self, clipValueMin), DFX_OUT(out));
    OP_CHECK_NULL(clipValueMin, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampCommon(self, clipValueMin, nullptr, out, workspaceSize, executor);
}

aclnnStatus aclnnClampMin(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClampMin);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

const aclTensor* AddBroadcastNode(const op::Shape& broadcastShape, const aclTensor* clipValue, aclOpExecutor* executor)
{
    // no need broadcast
    if (broadcastShape == clipValue->GetViewShape()) {
        return clipValue;
    }
    // aicore can do broadcast in clip_by_value
    if (op::CheckType(clipValue->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
        return clipValue;
    }
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
    auto broadcastDstTensor = executor->AllocTensor(broadcastShape, clipValue->GetDataType());
    auto shape =
        executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(), static_cast<op::DataType>(ACL_INT64));
    return l0op::BroadcastTo(clipValue, broadcastDstTensor, shape, executor);
}

static bool ClampTensorPromoteShape(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, const aclTensor* out,
    op::Shape& broadcastShape)
{
    if (clipValueMin != nullptr && clipValueMax != nullptr) {
        op::Shape broadcastShapeMin;
        if (!BroadcastInferShape(self->GetViewShape(), clipValueMin->GetViewShape(), broadcastShapeMin)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the size of tensor %s %s must match the size of tensor %s %s.", "self",
                op::ToString(self->GetViewShape()).GetString(), "min",
                op::ToString(clipValueMin->GetViewShape()).GetString());
            return false;
        }
        if (!BroadcastInferShape(clipValueMax->GetViewShape(), broadcastShapeMin, broadcastShape)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the size of tensor %s %s must match the size of tensor %s %s.", "self/min",
                op::ToString(broadcastShapeMin).GetString(), "max",
                op::ToString(clipValueMax->GetViewShape()).GetString());
            return false;
        }
    } else if (clipValueMin != nullptr) {
        if (!BroadcastInferShape(self->GetViewShape(), clipValueMin->GetViewShape(), broadcastShape)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the size of tensor %s %s must match the size of tensor %s %s.", "self",
                op::ToString(self->GetViewShape()).GetString(), "min",
                op::ToString(clipValueMin->GetViewShape()).GetString());
            return false;
        }
    } else {
        if (!BroadcastInferShape(self->GetViewShape(), clipValueMax->GetViewShape(), broadcastShape)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "the size of tensor %s %s must match the size of tensor %s %s.", "self",
                op::ToString(self->GetViewShape()).GetString(), "max",
                op::ToString(clipValueMax->GetViewShape()).GetString());
            return false;
        }
    }
    if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "expected tensor for %s to have same size as tensor for %s, but %s does not "
            "equal %s.",
            "out", "broadcast", op::ToString(out->GetViewShape()).GetString(),
            op::ToString(broadcastShape).GetString());
        return false;
    }
    return true;
}

static bool ClampTensorPromoteType(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, const aclTensor* out,
    op::DataType& promoteType)
{
    promoteType = self->GetDataType();
    if (clipValueMin != nullptr) {
        promoteType = op::PromoteType(promoteType, clipValueMin->GetDataType());
    }
    if (clipValueMax != nullptr) {
        promoteType = op::PromoteType(promoteType, clipValueMax->GetDataType());
    }
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promote type fail.");
        return false;
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    auto supportList = GetDtypeSupportList();
    if (!CheckType(promoteType, supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "%s not implemented for %s, should be in dtype support list %s.", "promote dtype",
            op::ToString(promoteType).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    OP_LOGI("clamp process with promote dtype %s.", op::ToString(promoteType).GetString());

    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

static aclnnStatus aclnnClampTensorCommon(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, clipValueMin, clipValueMax, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || (clipValueMin != nullptr && clipValueMin->IsEmpty()) ||
        (clipValueMax != nullptr && clipValueMax->IsEmpty())) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    op::DataType promoteType;
    CHECK_RET(ClampTensorPromoteType(self, clipValueMin, clipValueMax, out, promoteType), ACLNN_ERR_PARAM_INVALID);
    op::Shape promoteShape;
    CHECK_RET(ClampTensorPromoteShape(self, clipValueMin, clipValueMax, out, promoteShape), ACLNN_ERR_PARAM_INVALID);

    auto minTensor = NormalizeMinTensor(clipValueMin, promoteType, uniqueExecutor.get());
    auto maxTensor = NormalizeMaxTensor(clipValueMax, promoteType, uniqueExecutor.get());

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto minTensorCasted = l0op::Cast(minTensor, promoteType, uniqueExecutor.get());
    CHECK_RET(minTensorCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto maxTensorCasted = l0op::Cast(maxTensor, promoteType, uniqueExecutor.get());
    CHECK_RET(maxTensorCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    ret = CheckReformat(selfCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckReformat(minTensorCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckReformat(maxTensorCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto selfTensorCastedBrc = AddBroadcastNode(promoteShape, selfCasted, uniqueExecutor.get());
    CHECK_RET(selfTensorCastedBrc != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto minTensorCastedBrc = AddBroadcastNode(promoteShape, minTensorCasted, uniqueExecutor.get());
    CHECK_RET(minTensorCastedBrc != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto maxTensorCastedBrc = AddBroadcastNode(promoteShape, maxTensorCasted, uniqueExecutor.get());
    CHECK_RET(maxTensorCastedBrc != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ceilResult =
        l0op::ClipByValueV2(selfTensorCastedBrc, minTensorCastedBrc, maxTensorCastedBrc, uniqueExecutor.get());
    CHECK_RET(ceilResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(ceilResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOutFormat = l0op::ReFormat(castOut, out->GetStorageFormat());
    CHECK_RET(castOutFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOutFormat, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClampTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin, const aclTensor* clipValueMax, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClampTensor, DFX_IN(self, clipValueMin, clipValueMax), DFX_OUT(out));
    return aclnnClampTensorCommon(self, clipValueMin, clipValueMax, out, workspaceSize, executor);
}

aclnnStatus aclnnClampTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClampTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnClampMinTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* clipValueMin, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClampMinTensor, DFX_IN(self, clipValueMin), DFX_OUT(out));
    OP_CHECK_NULL(clipValueMin, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampTensorCommon(self, clipValueMin, nullptr, out, workspaceSize, executor);
}

aclnnStatus aclnnClampMinTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClampMinTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceClampMinTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* clipValueMin, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceClampMinTensor, DFX_IN(selfRef, clipValueMin), DFX_OUT(selfRef));
    OP_CHECK_NULL(clipValueMin, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampTensorCommon(selfRef, clipValueMin, nullptr, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceClampMinTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceClampMinTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnClampMaxGetWorkspaceSize(
    const aclTensor* self, const aclScalar* clipValueMax, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClampMax, DFX_IN(self, clipValueMax), DFX_OUT(out));
    OP_CHECK_NULL(clipValueMax, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampCommon(self, nullptr, clipValueMax, out, workspaceSize, executor);
}

aclnnStatus aclnnClampMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClampMax);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceClampMaxGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* clipValueMax, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceClampMax, DFX_IN(selfRef, clipValueMax), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    OP_CHECK_NULL(clipValueMax, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampCommon(selfRef, nullptr, clipValueMax, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceClampMax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceClampMax);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceClampMaxTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* max, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceClampMaxTensor, DFX_IN(selfRef, max), DFX_OUT(selfRef));
    OP_CHECK_NULL(max, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampTensorCommon(selfRef, nullptr, max, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceClampMaxTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceClampMaxTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnClampMaxTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* max, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClampMaxTensor, DFX_IN(self, max), DFX_OUT(out));
    OP_CHECK_NULL(max, return ACLNN_ERR_PARAM_NULLPTR);
    return aclnnClampTensorCommon(self, nullptr, max, out, workspaceSize, executor);
}

aclnnStatus aclnnClampMaxTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClampMaxTensor);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
