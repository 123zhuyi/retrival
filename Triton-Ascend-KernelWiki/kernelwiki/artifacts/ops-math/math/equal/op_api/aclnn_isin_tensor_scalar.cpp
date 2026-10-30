/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_isin_tensor_scalar.h"
#include "equal.h"

#include "math/not_equal/op_api/not_equal.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

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
static inline DataType PromoteTypeScalar(const aclTensor* element, const aclScalar* testElement)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        op::DataType promoteType;
        auto scalarDefaultDtype = GetScalarDefaultDtype(testElement->GetDataType());
        promoteType = CombineCategoriesWithComplex(element->GetDataType(), scalarDefaultDtype);
        return promoteType;
    }
    if (IsFloatingType(element->GetDataType())) {
        return element->GetDataType();
    }
    if (IsFloatingType(testElement->GetDataType())) {
        return op::PromoteType(testElement->GetDataType(), element->GetDataType());
    }
    return element->GetDataType();
}

static inline bool CheckNotNull(const aclTensor* element, const aclScalar* testElement, const aclTensor* out)
{
    OP_CHECK_NULL(element, return false);
    OP_CHECK_NULL(testElement, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* element, const aclScalar* testElement, const aclTensor* out)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查element的数据类型是否在在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(element, supportList, return false);
    // 检查testElement的数据类型是否在在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(testElement, supportList, return false);
    // 检查out的数据类型是否在在支持列表内
    OP_CHECK_DTYPE_NOT_MATCH(out, op::DataType::DT_BOOL, return false);

    // 检查element和testElement能否做数据类型推导
    op::DataType promoteType = PromoteTypeScalar(element, testElement);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "element dtype %s and testElement dtype %s can not promote dtype.",
            op::ToString(element->GetDataType()).GetString(), op::ToString(testElement->GetDataType()).GetString());
        return false;
    }

    // 检查promoteType的数据类型是否在equal算子的支持列表内
    if (!CheckType(promoteType, supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "element dtype %s and testElement dtype %s get promoteType dtype %s should be in "
            "dtype support list [%s].",
            op::ToString(element->GetDataType()).GetString(), op::ToString(testElement->GetDataType()).GetString(),
            op::ToString(promoteType).GetString(), op::ToString(supportList).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor* element, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(element, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(out, element, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* element, const aclScalar* testElement, const aclTensor* out)
{
    // 1.检查参数是否为空指针
    CHECK_RET(CheckNotNull(element, testElement, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(element, testElement, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape
    CHECK_RET(CheckShape(element, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(
    const aclTensor* element, const aclScalar* testElement, [[maybe_unused]] bool assumeUnique, bool invert,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsInTensorScalar, DFX_IN(element, testElement), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(element, testElement, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空进空出
    if (element->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto promoteType = PromoteTypeScalar(element, testElement);

    // 固定写法，将输入element转换成连续的tensor
    auto elementContiguous = l0op::Contiguous(element, uniqueExecutor.get());
    CHECK_RET(elementContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入element的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto elementCasted = l0op::Cast(elementContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(elementCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto testElementTensor = uniqueExecutor.get()->ConvertToTensor(testElement, promoteType);
    CHECK_RET(testElementTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* isInOut = nullptr;
    if (invert) {
        isInOut = l0op::NotEqual(elementCasted, testElementTensor, uniqueExecutor.get());
    } else {
        isInOut = l0op::Equal(elementCasted, testElementTensor, uniqueExecutor.get());
    }

    CHECK_RET(isInOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(isInOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsInTensorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsInTensorScalar);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
