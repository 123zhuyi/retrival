/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_eq_scalar.h"
#include "equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t EQ_MAX_DIM_NUM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_UINT64,
    op::DataType::DT_UINT32,    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,     op::DataType::DT_INT64,      op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 950相对于之前，增加了DT_UINT64
static const std::initializer_list<op::DataType> REGBASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,     op::DataType::DT_INT64,      op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16,
    op::DataType::DT_UINT64};

//其他output支持的类型
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,  op::DataType::DT_UINT8,  op::DataType::DT_UINT64,
    op::DataType::DT_UINT32,    op::DataType::DT_BOOL,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64,
    op::DataType::DT_COMPLEX128};

// 910B-910_93的output支持的类型
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,     op::DataType::DT_INT64,      op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

// 950的output支持的类型
static const std::initializer_list<op::DataType> REGBASE_OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_UINT32,    op::DataType::DT_UINT64,     op::DataType::DT_BOOL,  op::DataType::DT_UINT16,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static inline double GetCastedDouble(const aclTensor* self, const aclScalar* other)
{
    double res = 0;
    switch (self->GetDataType()) {
        case DataType::DT_FLOAT:
            res = static_cast<double>(other->ToFloat());
            break;
        case DataType::DT_FLOAT16:
            res = static_cast<double>(other->ToFp16());
            break;
        default:
            res = other->ToDouble();
            break;
    }
    return res;
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

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

static inline bool IsDoubleEqual(double a, double b)
{
    if (std::abs(a - b) <= std::numeric_limits<float>::epsilon()) {
        return true;
    }
    return false;
}

static op::DataType PromoteTypeScalar(op::DataType selfDtype, op::DataType otherDtype)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        op::DataType promoteType;
        auto scalarDefaultDtype = GetScalarDefaultDtype(otherDtype);
        promoteType = CombineCategoriesWithComplex(selfDtype, scalarDefaultDtype);
        if (promoteType == op::DataType::DT_COMPLEX32) {
            promoteType = op::DataType::DT_COMPLEX64;
        }
        return promoteType;
    }

    if (IsComplexType(selfDtype)) {
        return selfDtype;
    }

    if (!IsComplexType(otherDtype) && IsFloatingType(selfDtype)) {
        return selfDtype;
    }

    if (selfDtype == op::DataType::DT_BOOL || IsFloatingType(otherDtype) || IsComplexType(otherDtype)) {
        return op::PromoteType(selfDtype, otherDtype);
    }

    return selfDtype;
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

static const std::initializer_list<op::DataType> GetInputDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    switch (npuArch) {
        case NpuArch::DAV_2201: {
            return DTYPE_SUPPORT_910B_LIST;
        }
        case NpuArch::DAV_3510: {
            return REGBASE_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return DTYPE_SUPPORT_910_LIST;
        }
        default: {
            // 如果既不是1971也不是1980，先暂且默认当做1971处理
            return DTYPE_SUPPORT_910B_LIST;
        }
    }
}

static inline const std::initializer_list<op::DataType>& GetOutputDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_1001) {
        return DTYPE_SUPPORT_910_LIST;
    } else if (IsRegBase(npuArch)) {
        return REGBASE_OUT_DTYPE_SUPPORT_LIST;
    }
    return DTYPE_SUPPORT_910B_LIST;
}

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    const std::initializer_list<op::DataType> dtypeSupportList = GetInputDtypeSupportList();
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

    // 检查other的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, dtypeSupportList, return false);

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto outSuportList = IsRegBase(npuArch) ?
                         GetOutputDtypeSupportList() :
                         dtypeSupportList;
    // 检查out的数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(out, outSuportList, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    // 检查self和other能否做数据类型推导
    op::DataType promoteType;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        promoteType = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
        auto inputSupportList = GetInputDtypeSupportList();
        // 检查self的数据类型是否在Equal算子的支持列表内
        if (!CheckType(promoteType, inputSupportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "promote dtype %s should be in dtype support list [%s].",
                op::ToString(promoteType).GetString(), op::ToString(inputSupportList).GetString());
            return false;
        }
    } else {
        promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    }
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), promoteType, return false);

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), promoteType, return false);

    // 检查BOOL类型能否转换为输出的数据类型(算子返回的都是BOOL类型)
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(DataType::DT_BOOL, out->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, EQ_MAX_DIM_NUM, return false);
    OP_CHECK_MAX_DIM(out, EQ_MAX_DIM_NUM, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

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

aclnnStatus aclnnEqScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnEqScalar, DFX_IN(self, other), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (HasEmptyTensor(self)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* otherTensor = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Equal进行计算
    auto equalResult = l0op::Equal(selfCasted, otherTensor, uniqueExecutor.get());
    CHECK_RET(equalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果(BOOL)转换成输出out的数据类型
    auto castOut = l0op::Cast(equalResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceEqScalarGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnEqScalarGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnEqScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEqScalar);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceEqScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceEqScalar);
    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
