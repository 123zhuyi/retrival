/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_isin.h"
#include "aclnn_kernels/contiguous.h"
#include "equal.h"
#include "math/reduce_any/op_api/reduce_any.h"
#include "math/logical_not/op_api/logical_not.h"
#include "aclnn_kernels/cast.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t MAX_DIM_LEN = 8;
static const int64_t ELEMENTS_THRESHOLD = 10; // 当testElements的元素个数大于等于10，直接进行类型推导

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    // AICORE可以支持的数据类型
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,
    // 走AICPU的数据类型
    op::DataType::DT_DOUBLE, op::DataType::DT_INT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    // AICORE可以支持的数据类型
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8, op::DataType::DT_BF16,
    // 走AICPU的数据类型
    op::DataType::DT_DOUBLE, op::DataType::DT_INT16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline bool CheckNotNull(const aclScalar* element, const aclTensor* testElements, const aclTensor* out)
{
    OP_CHECK_NULL(element, return false);
    OP_CHECK_NULL(testElements, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclScalar* element, const aclTensor* testElements, const aclTensor* out)
{
    const auto& supportList = GetDtypeSupportList();
    // 检查element的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(element, supportList, return false);
    // 检查testElements的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(testElements, supportList, return false);
    // 检查out的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_MATCH(out, op::DataType::DT_BOOL, return false);
    return true;
}

static inline DataType PromoteTypeScalar(const aclScalar* element, const aclTensor* testElements)
{
    if (testElements->Size() >= ELEMENTS_THRESHOLD) {
        return op::PromoteType(testElements->GetDataType(), element->GetDataType());
    }

    if (IsFloatingType(testElements->GetDataType())) {
        return testElements->GetDataType();
    }

    if (IsFloatingType(element->GetDataType())) {
        return op::PromoteType(testElements->GetDataType(), element->GetDataType());
    }

    return testElements->GetDataType();
}

static inline bool CheckPromoteType(const aclScalar* element, const aclTensor* testElements)
{
    // 检查element和testElements能否做数据类型推导
    auto promoteType = PromoteTypeScalar(element, testElements);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected element %s and testElements %s can promote dtype but check failed.",
            op::ToString(element->GetDataType()).GetString(), op::ToString(testElements->GetDataType()).GetString());
        return false;
    }

    OP_CHECK_RESULT_DTYPE_CAST_FAILED(element->GetDataType(), promoteType, return false);
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(testElements->GetDataType(), promoteType, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* testElements, const aclTensor* out)
{
    // 检查testElements的维度是否大于8维
    OP_CHECK_MAX_DIM(testElements, MAX_DIM_LEN, return false);
    // 检查out是否为scalar
    OP_CHECK_WRONG_DIMENSION(out, 0, return false);
    return true;
}

static inline aclnnStatus CheckParams(const aclScalar* element, const aclTensor* testElements, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(element, testElements, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(element, testElements, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查element和testElements能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(element, testElements), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查tensor的维度
    CHECK_RET(CheckShape(testElements, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline aclnnStatus FillScalar(aclTensor* out, bool val, aclOpExecutor* executor)
{
    FVector<int64_t> shape;
    shape.push_back(1);
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<bool> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclIntArray* GetAllDims(const aclTensor* tensor, aclOpExecutor* executor)
{
    const size_t input_dim_num = tensor->GetViewShape().GetDimNum();
    std::vector<int64_t> dims(input_dim_num);
    for (size_t idx = 0; idx < input_dim_num; idx++) {
        dims[idx] = idx;
    }
    return executor->AllocIntArray(dims.data(), input_dim_num);
}

aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(
    const aclScalar* element, const aclTensor* testElements, bool assumeUnique, bool invert, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsInScalarTensor, DFX_IN(element, testElements, assumeUnique, invert), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(element, testElements, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入testElements为空tensor或者输入element是nan时，输出是值为invert的标量tensor
    if (testElements->IsEmpty() ||
        (IsFloatingType(element->GetDataType()) && element->ToFloat() != element->ToFloat())) {
        // 填充invert
        ret = FillScalar(out, invert, uniqueExecutor.get());
        if (ret == ACLNN_SUCCESS) {
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
            uniqueExecutor.ReleaseTo(executor);
        }
        return ret;
    }

    if (testElements->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Format only support ND");
    }
    // testElement如果非连续，需要转连续
    auto testElementsContiguous = l0op::Contiguous(testElements, uniqueExecutor.get());
    CHECK_RET(testElementsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto promoteType = PromoteTypeScalar(element, testElements);
    // 将输入testElements的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto testElementsCasted = l0op::Cast(testElementsContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(testElementsCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* elementTensor = uniqueExecutor.get()->ConvertToTensor(element, promoteType);
    CHECK_RET(elementTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Equal进行计算
    auto equalResult = l0op::Equal(testElementsCasted, elementTensor, uniqueExecutor.get());
    CHECK_RET(equalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto dim = GetAllDims(equalResult, uniqueExecutor.get());
    CHECK_RET(dim != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto anyResult = l0op::ReduceAny(equalResult, dim, false, uniqueExecutor.get());
    CHECK_RET(anyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = invert ? l0op::LogicalNot(anyResult, uniqueExecutor.get()) : anyResult;
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyMinResult = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(viewCopyMinResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsInScalarTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsInScalarTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif