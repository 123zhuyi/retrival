/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rsub.h"
#include "sub.h"
#include "math/axpy/op_api/axpy.h"
#include "math/axpy_v2/op_api/axpy_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/mul/op_api/mul.h"
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
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ARCH3510_AXPY_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_AXPY_V2_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    // 如下数据类型会走AICPU
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    // 如下数据类型会走AICPU
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

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

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_2201:
        case NpuArch::DAV_3510: {
            return ASCEND910B_DTYPE_SUPPORT_LIST;
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

static inline bool IsScalarOne(const aclScalar* alpha)
{
    if (IsIntegralType(alpha->GetDataType())) {
        return alpha->ToInt32() == 1;
    } else if (IsFloatingType(alpha->GetDataType())) {
        return alpha->ToFloat() == 1.0;
    }
    return false;
}

static inline bool IsEqualToOne(const op::DataType calcType, const aclScalar* alpha)
{
    if (!IsRegBase()) {
        return IsScalarOne(alpha);
    }
    if (IsComplexType(alpha->GetDataType()) || IsComplexType(calcType)) {
        return false;
    }

    if (calcType == DataType::DT_DOUBLE) {
        return !(alpha->ToDouble() > 1 || alpha->ToDouble() < 1);
    }
    return !(alpha->ToFloat() > 1 || alpha->ToFloat() < 1);
}

static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* other)
{
    // 检查self和other的数据类型是否在算子的支持列表内
    const auto& supportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
    return true;
}

static inline bool CheckRsubsDtypeValid(const aclTensor* self, const aclScalar* other)
{
    // 检查self和other的数据类型是否在算子的支持列表内
    const auto& supportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
    return true;
}

static inline bool CheckPromoteType(
    const op::DataType selfDtype, const op::DataType otherDtype, const aclScalar* alpha, const op::DataType outDtype,
    op::DataType promoteType)
{
    // 检查self和other能否做数据类型推导
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected self dtype %s and other dtype %s can promote dtype but check failed.",
            op::ToString(selfDtype).GetString(), op::ToString(otherDtype).GetString());
        return false;
    }

    // 检查alpha能否转换为推导后的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(alpha->GetDataType(), promoteType, return false);
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
    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

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
    return false;
}

static bool UseAxpyV2(const DataType promoteType, [[maybe_unused]] const aclScalar* alpha)
{
    if (IsRegBase()) {
        return CheckType(promoteType, ARCH3510_AXPY_V2_DTYPE_SUPPORT_LIST);
    }
    return false;
}

aclnnStatus aclnnRsubGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRsub, DFX_IN(self, other, alpha), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || other->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Add算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行rsub计算
    const aclTensor* rsubOut = nullptr;
    if (IsEqualToOne(promoteType, alpha)) {
        rsubOut = l0op::Sub(otherCasted, selfCasted, uniqueExecutor.get());
    } else if (UseAxpy(promoteType, alpha)) {
        float alphaNeg = alpha->ToFloat() * (-1.0f);
        rsubOut = l0op::Axpy(otherCasted, selfCasted, alphaNeg, uniqueExecutor.get());
    } else if (UseAxpyV2(promoteType, alpha)) {
        int64_t alphaNeg = alpha->ToInt64() * (static_cast<int64_t>(-1));
        aclScalar* alphaNegPtr = uniqueExecutor.get()->AllocScalar(alphaNeg);
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alphaNegPtr, promoteType);
        rsubOut = l0op::AxpyV2(otherCasted, selfCasted, alphaTensor, uniqueExecutor.get());
    } else {
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, promoteType);
        auto selfRes = l0op::Mul(selfCasted, alphaTensor, uniqueExecutor.get());
        CHECK_RET(selfRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        rsubOut = l0op::Sub(otherCasted, selfRes, uniqueExecutor.get());
    }
    CHECK_RET(rsubOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(rsubOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static inline bool CheckNotNullScalar(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckShapeScalar(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    return true;
}

static inline DataType PromoteTypeScalar(const aclTensor* self, const aclScalar* other, const aclScalar* alpha)
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
    return IsFloatingType(self->GetDataType()) ?
               self->GetDataType() :
               (IsFloatingType(other->GetDataType()) ? op::PromoteType(self->GetDataType(), other->GetDataType()) :
                                                       self->GetDataType());
}

static inline aclnnStatus CheckParamsScalar(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullScalar(self, other, alpha, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckRsubsDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    auto promoteType = PromoteTypeScalar(self, other, alpha);
    CHECK_RET(
        CheckPromoteType(self->GetDataType(), other->GetDataType(), alpha, out->GetDataType(), promoteType),
        ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape
    CHECK_RET(CheckShapeScalar(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRsubsGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, const aclScalar* alpha, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRsubs, DFX_IN(self, other, alpha), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParamsScalar(self, other, alpha, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = PromoteTypeScalar(self, other, alpha);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行rsub计算
    const aclTensor* rsubOut = nullptr;
    if (IsEqualToOne(promoteType, alpha)) {
        rsubOut = l0op::Sub(otherTensor, selfCasted, uniqueExecutor.get());
    } else if (UseAxpy(promoteType, alpha)) {
        float alphaNeg = alpha->ToFloat() * (-1.0f);
        rsubOut = l0op::Axpy(otherTensor, selfCasted, alphaNeg, uniqueExecutor.get());
    } else if (UseAxpyV2(promoteType, alpha)) {
        int64_t alphaNeg = alpha->ToInt64() * (static_cast<int64_t>(-1));
        aclScalar* alphaNegPtr = uniqueExecutor.get()->AllocScalar(alphaNeg);
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alphaNegPtr, promoteType);
        rsubOut = l0op::AxpyV2(otherTensor, selfCasted, alphaTensor, uniqueExecutor.get());
    } else {
        auto alphaTensor = uniqueExecutor.get()->ConvertToTensor(alpha, promoteType);
        auto selfRes = l0op::Mul(selfCasted, alphaTensor, uniqueExecutor.get());
        CHECK_RET(selfRes != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        rsubOut = l0op::Sub(otherTensor, selfRes, uniqueExecutor.get());
    }
    CHECK_RET(rsubOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(rsubOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRsubs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRsubs);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnRsub(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRsub);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif