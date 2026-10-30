/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_floor_divide.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "floordiv.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,  op::DataType::DT_INT64,
    op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16,   op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

[[maybe_unused]] static op::DataType InnerTypeToComplexType(const op::DataType input)
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

[[maybe_unused]] static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower)
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

[[maybe_unused]] static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

[[maybe_unused]] static op::DataType GetOpMathDtype(const op::DataType input)
{
    // only handel DT_BF16, DT_FLOAT16, DT_COMPLEX32
    switch (input) {
        case op::DataType::DT_BF16:
            return op::DataType::DT_FLOAT;
        case op::DataType::DT_FLOAT16:
            return op::DataType::DT_FLOAT;
        case op::DataType::DT_COMPLEX32:
            return op::DataType::DT_COMPLEX64;
        default:
            return input;
    }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline op::DataType InferFloorDivTensorScalarDtype(const op::DataType selfDtype, const op::DataType otherDtype)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        auto scalarDefaultDtype = GetScalarDefaultDtype(otherDtype);
        auto promoteType = CombineCategoriesWithComplex(selfDtype, scalarDefaultDtype);
        if ((promoteType == op::DataType::DT_FLOAT16) || (promoteType == op::DataType::DT_BF16) ||
            (promoteType == op::DataType::DT_COMPLEX32)) {
            promoteType = GetOpMathDtype(promoteType);
        }
        return promoteType;
    }
    auto promoteType = op::PromoteType(selfDtype, otherDtype);
    promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;
    return promoteType;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在FloorDiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查other的数据类型是否在FloorDiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
    return true;
}

static bool CheckDtypeValidScalar(const aclTensor* self, const aclScalar* other)
{
    auto supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在FloorDiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查other的数据类型是否在FloorDiv算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
    return true;
}

static bool CheckPromoteType(
    const op::DataType selfDtype, const op::DataType otherDtype, const op::DataType outDtype, const bool isOtherScalar)
{
    auto promoteType =
        isOtherScalar ? InferFloorDivTensorScalarDtype(selfDtype, otherDtype) : op::PromoteType(selfDtype, otherDtype);
    // 检查self和other能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(otherDtype).GetString());
        return false;
    }
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        // 增加cast能力检查
        auto supportList = GetDtypeSupportList();
        if (!CheckType(promoteType, supportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "aclnnFloorDivide not implemented for input dtype %s,"
                "should be in dtype support list %s.",
                ToString(promoteType).GetString(), op::ToString(supportList).GetString());
            return false;
        }
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDtype, promoteType, return false);
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(otherDtype, promoteType, return false);
        // 检查计算结果的数据类型能否转换为输出的数据类型
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
    } else {
        promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* y)
{
    CHECK_RET(CheckNotNull(self, other, y), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(
        CheckPromoteType(self->GetDataType(), other->GetDataType(), y->GetDataType(), false), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(self, other, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CalcFloorDivide(
    const aclTensor* self, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || other->IsEmpty()) {
        return ACLNN_SUCCESS;
    }

    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;

    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor);
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto otherContiguous = l0op::Contiguous(other, executor);
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto otherCasted = l0op::Cast(otherContiguous, promoteType, executor);
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* divOpOut = l0op::FloorDiv(selfCasted, otherCasted, executor);
    CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFloorDivideGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFloorDivide, DFX_IN(self, other), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CalcFloorDivide(self, other, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFloorDivide(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFloorDivide);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static bool CheckNotNullScalar(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShapeScalar(const aclTensor* self, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    if (self->GetViewShape() != y->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParamsScalar(const aclTensor* self, const aclScalar* other, const aclTensor* y)
{
    CHECK_RET(CheckNotNullScalar(self, other, y), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValidScalar(self, other), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(
        CheckPromoteType(self->GetDataType(), other->GetDataType(), y->GetDataType(), true), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeScalar(self, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CalcFloorDivides(
    const aclTensor* self, const aclScalar* other, aclTensor* out, aclOpExecutor* executor)
{
    // 调用适配aclScalar参数检查
    auto ret = CheckParamsScalar(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        return ACLNN_SUCCESS;
    }

    auto promoteType = InferFloorDivTensorScalarDtype(self->GetDataType(), other->GetDataType());
    promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;

    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, executor);
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // aclScalar转aclTensor
    auto otherConvert = executor->ConvertToTensor(other, promoteType);
    CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* divOpOut = l0op::FloorDiv(selfCasted, otherConvert, executor);
    CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFloorDividesGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFloorDivides, DFX_IN(self, other), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CalcFloorDivides(self, other, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFloorDivides(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFloorDivides);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static inline aclnnStatus CheckInplace(const aclTensor* selfRef, const aclTensor* other)
{
    OP_CHECK(
        selfRef != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected selfRef not to be null."),
        return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(
        other != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected other not to be null."),
        return ACLNN_ERR_PARAM_NULLPTR);
    op::Shape broadcastShape;
    OP_CHECK(
        BroadcastInferShape(selfRef->GetViewShape(), other->GetViewShape(), broadcastShape),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of selfRef and other can't broadcast, got %s, %s.",
            op::ToString(selfRef->GetViewShape()).GetString(), op::ToString(other->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(
        selfRef->GetViewShape() == broadcastShape,
        OP_LOGE(
            ACLNN_ERR_PARAM_NULLPTR, "Expected shape of selfRef should be %s, but got %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(selfRef->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFloorDivideGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFloorDivide, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckInplace(selfRef, other);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto out = const_cast<aclTensor*>(selfRef);
    ret = CalcFloorDivide(selfRef, other, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFloorDivide(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFloorDivide);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceFloorDividesGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFloorDivides, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CalcFloorDivides(selfRef, other, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceFloorDivides(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFloorDivides);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif