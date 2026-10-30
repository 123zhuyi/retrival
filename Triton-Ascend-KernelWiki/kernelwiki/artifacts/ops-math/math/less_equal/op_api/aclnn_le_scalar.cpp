/**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_le_scalar.h"
#include "less_equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_INT16,   op::DataType::DT_UINT16,
    op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,     op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,      op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,    op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> REGBASE_OUT_DTYPE_SUPPORT_LIST  = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,     op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

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

static bool CheckNotNull(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool HasEmptyTensor(const aclTensor* self)
{
    if (self->IsEmpty()) {
        return true;
    }

    return false;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        return REGBASE_DTYPE_SUPPORT_LIST;
    }
    if (npuArch == NpuArch::DAV_2201) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline const std::initializer_list<op::DataType>& GetOutputDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910E) {
        return REGBASE_OUT_DTYPE_SUPPORT_LIST ;
    }
    return OUT_DTYPE_SUPPORT_910_LIST;
}

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto outSuportList = IsRegBase(npuArch) ?
                         GetOutputDtypeSupportList() :
                         supportList;
    op::DataType outType = out->GetDataType();
    if ((!CheckType(outType, outSuportList)) && (outType != DataType::DT_BOOL)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "out dtype %s should be in dtype support list [%s].",
            op::ToString(out->GetDataType()).GetString(), op::ToString(outSuportList).GetString());
        return false;
    }

    return true;
}

static DataType PromoteTypeScalar(const aclTensor* self, const aclScalar* other)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        auto scalarDefaultDtype = GetScalarDefaultDtype(other->GetDataType());
        auto promoteType = CombineCategoriesWithComplex(self->GetDataType(), scalarDefaultDtype);
        return promoteType;
    }
    if (IsFloatingType(self->GetDataType())) {
        return self->GetDataType();
    }

    if (other->GetDataType() == op::DataType::DT_DOUBLE) {
        return op::DataType::DT_FLOAT;
    }

    return op::PromoteType(self->GetDataType(), other->GetDataType());
}

static bool CheckPromoteType(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    // 检查self和other能否做数据类型推导
    op::DataType promoteType = PromoteTypeScalar(self, other);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        const auto& supportList = GetDtypeSupportList();
        if (!CheckType(promoteType, supportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "aclnnLeScalar not implemented for input dtype %s,"
                "should be in dtype support list [%s].",
                ToString(promoteType).GetString(), op::ToString(supportList).GetString());
            return false;
        }
    }

    // 检查BOOL类型能否转换为输出的数据类型(算子返回的都是BOOL类型)
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    const size_t MAX_DIM = 8;
    OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM, return false);

    // self和out的shape必须一致
    if (self->GetViewShape() != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self shape is different with out shape, self [%s], out [%s].",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnLeScalarCommon(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空Tensor处理
    if (HasEmptyTensor(self)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto promoteType = PromoteTypeScalar(self, other);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (promoteType == DataType::DT_BOOL && !IsRegBase(npuArch)) {
        promoteType = DataType::DT_FLOAT;
    }
    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子LessEqual进行计算
    auto leResult = l0op::LessEqual(selfCasted, otherTensor, uniqueExecutor.get());
    CHECK_RET(leResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果(BOOL)转换成输出out的数据类型
    auto castOut = l0op::Cast(leResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLeScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLeScalar, DFX_IN(self, other), DFX_OUT(out));
    return aclnnLeScalarCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnLeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLeScalar);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLeScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceLeScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    return aclnnLeScalarCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceLeScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceLeScalar);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
