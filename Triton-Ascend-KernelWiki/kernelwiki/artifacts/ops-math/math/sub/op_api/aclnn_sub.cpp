/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_sub.h"
#include "sub.h"
#include "math/axpy/op_api/axpy.h"
#include "math/axpy_v2/op_api/axpy_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/mul/op_api/mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Sub 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             Sub(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ARCH3510_AXPY_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_AXPY_V2_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,     op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> AXPY_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

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

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
    }
}

static inline float GetCastedFloat(const op::DataType tensorDtype, const aclScalar* scalar)
{
    float castedRes = 0;
    switch (tensorDtype) {
        case DataType::DT_FLOAT16:
            castedRes = static_cast<float>(scalar->ToFp16());
            break;
        case DataType::DT_BF16:
            castedRes = static_cast<float>(scalar->ToBf16());
            break;
        default:
            castedRes = scalar->ToFloat();
            break;
    }
    return castedRes;
}

static inline bool IsFloatEqual(float a, float b)
{
    return std::abs(a - b) <= std::numeric_limits<float>::epsilon();
}

static inline bool IsEqualToOne(const op::DataType calcType, const aclScalar* alpha)
{
    if (!IsRegBase()) {
        return !(alpha->ToFloat() > 1 || alpha->ToFloat() < 1);
    }
    if (IsComplexType(alpha->GetDataType()) || IsComplexType(calcType)) {
        return false;
    }

    if (calcType == DataType::DT_DOUBLE) {
        return !(alpha->ToDouble() > 1 || alpha->ToDouble() < 1);
    }
    return !(alpha->ToFloat() > 1 || alpha->ToFloat() < 1);
}

static bool CheckPromoteType(
    const op::DataType selfDtype, const op::DataType otherDtype, const aclScalar* alpha, const op::DataType outDtype,
    op::DataType promoteType)
{
    // 检查self和other能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(otherDtype).GetString());
        return false;
    }
    if (promoteType == op::DataType::DT_BOOL) {
        OP_CHECK(
            IsIntegralType(DataType(alpha->GetDataType()), true),
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Alpha dtype %s can't be cast to the promote dtype %s.",
                op::ToString(DataType(alpha->GetDataType())).GetString(), op::ToString(promoteType).GetString()),
            return false);
    } else {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType(alpha->GetDataType()), promoteType, return false);
    }
    if (IsRegBase()) {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDtype, promoteType, return false);
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(otherDtype, promoteType, return false);
        const auto& supportList = GetDtypeSupportListBySocVersion();
        if (!CheckType(promoteType, supportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Input dtype %s is not implemented, it"
                "should be in dtype support list %s.",
                ToString(promoteType).GetString(), op::ToString(supportList).GetString());
            return false;
        }
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(other, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    CHECK_RET(
        CheckPromoteType(self->GetDataType(), other->GetDataType(), alpha, out->GetDataType(), promoteType),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查双输入是否能broadcast
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool UseAxpy(const DataType promoteType, [[maybe_unused]] const aclScalar* alpha)
{
    if (IsRegBase()) {
        return CheckType(promoteType, ARCH3510_AXPY_DTYPE_SUPPORT_LIST);
    }
    return CheckType(promoteType, AXPY_DTYPE_SUPPORT_LIST);
}

static bool UseAxpyV2(const DataType promoteType, [[maybe_unused]] const aclScalar* alpha)
{
    if (IsRegBase()) {
        return CheckType(promoteType, ARCH3510_AXPY_V2_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static void CheckFormat(const aclTensor* self, const aclTensor* other){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  ge::Format otherStorageFormat = other->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND || otherStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnSub/aclnnInplaceSub only support format ND.");
  }
}

static void SubsCheckFormat(const aclTensor* self) {
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnSubs/aclnnInplaceSubs doesn't support format NZ.");
    }
}

aclnnStatus aclnnSubGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSub, DFX_IN(self, other, alpha), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查self和other的format是否支持
    CheckFormat(self, other);

    // Sub算子的空tensor在kernel中支持
    if (self->IsEmpty() || other->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 混合数据类型场景类型推导，且alpha不为1时需进一步做数据类型提升
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    if (!IsRegBase() && (alpha->ToFloat() > 1 || alpha->ToFloat() < 1) &&
        IsFloatingType(promoteType)) {
        promoteType = promoteType == DataType::DT_DOUBLE ? DataType::DT_DOUBLE : DataType::DT_FLOAT;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // alpha非1时右输入带缩放计算
    const aclTensor* subOpOut = nullptr;
    if (IsEqualToOne(promoteType, alpha)) {
        subOpOut = l0op::Sub(selfCasted, otherCasted, uniqueExecutor.get());
    } else if (UseAxpy(promoteType, alpha)) {
        float alphaNeg = alpha->ToFloat() * (-1.0f);
        subOpOut = l0op::Axpy(selfCasted, otherCasted, alphaNeg, uniqueExecutor.get());
    } else if (UseAxpyV2(promoteType, alpha)) {
        int64_t alphaNeg = alpha->ToInt64() * (static_cast<int64_t>(-1));
        aclScalar* alphaNegPtr = uniqueExecutor.get()->AllocScalar(alphaNeg);
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alphaNegPtr, promoteType);
        subOpOut = l0op::AxpyV2(selfCasted, otherCasted, alphaTensor, uniqueExecutor.get());
    } else {
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, promoteType);
        auto otherRes = l0op::Mul(otherCasted, alphaTensor, uniqueExecutor.get());
        CHECK_RET(otherRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        subOpOut = l0op::Sub(selfCasted, otherRes, uniqueExecutor.get());
    }
    CHECK_RET(subOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(subOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static bool CheckNotNullScalar(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShapeScalar(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static DataType PromoteTypeScalar(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, const aclTensor* out)
{
    if (IsRegBase()) {
        auto otherDefaultDtype = GetScalarDefaultDtype(other->GetDataType());
        auto promoteType = CombineCategoriesWithComplex(self->GetDataType(), otherDefaultDtype);
        if (promoteType == op::DataType::DT_FLOAT16 || promoteType == op::DataType::DT_BF16) {
            bool isKeepB16 = IsFloatEqual(GetCastedFloat(promoteType, other), other->ToFloat()) &&
                             IsFloatEqual(GetCastedFloat(promoteType, alpha), alpha->ToFloat());
            promoteType = isKeepB16 ? promoteType : op::DataType::DT_FLOAT;
        }
        if (promoteType == op::DataType::DT_COMPLEX32) {
            promoteType = op::DataType::DT_COMPLEX64;
        }
        return promoteType;
    }

    if (IsComplexType(self->GetDataType()) || IsComplexType(other->GetDataType())) {
        return op::PromoteType(self->GetDataType(), other->GetDataType());
    }
    if (IsFloatingType(self->GetDataType())) {
        return self->GetDataType();
    }

    if (other->GetDataType() == op::DataType::DT_DOUBLE && out->GetDataType() == op::DataType::DT_FLOAT) {
        return op::DataType::DT_FLOAT;
    }

    if (IsFloatingType(other->GetDataType()) || self->GetDataType() == op::DataType::DT_BOOL) {
        return op::PromoteType(self->GetDataType(), other->GetDataType());
    }

    return self->GetDataType();
}

static aclnnStatus CheckParamsScalar(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullScalar(self, other, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    auto promoteType = PromoteTypeScalar(self, other, alpha, out);
    CHECK_RET(
        CheckPromoteType(self->GetDataType(), other->GetDataType(), alpha, out->GetDataType(), promoteType),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckShapeScalar(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSubsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSubs, DFX_IN(self, other, alpha), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsScalar(self, other, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查self的format是否支持
    SubsCheckFormat(self);

    // Sub算子的空tensor在kernel中支持
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Sub算子需要对self和other两个输入做隐式数据类型转换
    auto promoteType = PromoteTypeScalar(self, other, alpha, out);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other从scalar转换成tensor
    auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);

    // 调用Sub算子kernel
    const aclTensor* subOpOut = nullptr;
    if (IsEqualToOne(promoteType, alpha)) {
        subOpOut = l0op::Sub(selfCasted, otherTensor, uniqueExecutor.get());
    } else if (UseAxpy(promoteType, alpha)) {
        float alphaNeg = alpha->ToFloat() * (-1.0f);
        subOpOut = l0op::Axpy(selfCasted, otherTensor, alphaNeg, uniqueExecutor.get());
    } else if (UseAxpyV2(promoteType, alpha)) {
        int64_t alphaNeg = alpha->ToInt64() * (static_cast<int64_t>(-1));
        aclScalar* alphaNegPtr = uniqueExecutor.get()->AllocScalar(alphaNeg);
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alphaNegPtr, promoteType);
        subOpOut = l0op::AxpyV2(selfCasted, otherTensor, alphaTensor, uniqueExecutor.get());
    } else {
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, promoteType);
        auto otherRes = l0op::Mul(otherTensor, alphaTensor, uniqueExecutor.get());
        CHECK_RET(otherRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
        subOpOut = l0op::Sub(selfCasted, otherRes, uniqueExecutor.get());
    }
    CHECK_RET(subOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(subOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckInplace(const aclTensor* selfRef, const aclTensor* other)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(other, return false);
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, other, broadcastShape, return false);
    OP_CHECK_BROADCAST_WITH_SHAPE(selfRef, broadcastShape, return false);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceSubGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckInplace(selfRef, other);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return aclnnSubGetWorkspaceSize(selfRef, other, alpha, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSubsGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, const aclScalar* alpha, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    return aclnnSubsGetWorkspaceSize(selfRef, other, alpha, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnSub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSub);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceSub);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnSubs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSubs);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSubs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceSubs);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
