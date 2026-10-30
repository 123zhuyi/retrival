/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <bitset>
#include "aclnn_reduce_log_sum.h"
#include "reduce_log_sum.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* data, const aclIntArray* axes, const aclTensor* reduce)
{
    OP_CHECK_NULL(data, return false);
    OP_CHECK_NULL(axes, return false);
    OP_CHECK_NULL(reduce, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* data, const aclTensor* reduce)
{
    // 检查data和reduce的数据类型是否在支持列表内
    if (IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(data, DTYPE_SUPPORT_LIST_950, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(reduce, DTYPE_SUPPORT_LIST_950, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(data, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(reduce, DTYPE_SUPPORT_LIST, return false);
    }

    return true;
}

static bool CheckMaxDimension(const aclTensor* data)
{
    OP_CHECK_MAX_DIM(data, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static inline uint64_t GetPosDim(int64_t dim, int64_t dimNum)
{
    if (dimNum <= 0) {
        dimNum = 1;
    }
    return dim >= 0 ? dim : dim + dimNum;
}

static bool CheckAxesValid(const aclTensor* data, const aclIntArray* axes)
{
    auto dataViewShape = data->GetViewShape();
    auto dataDimNum = static_cast<int64_t>(dataViewShape.GetDimNum());
    // data为标量时，axes range [-1, 0]
    if (dataDimNum <= 0) {
        dataDimNum = 1;
    }
    // axes为负时需要转正校验
    std::bitset<MAX_MASK_LEN> axesMask = std::bitset<MAX_MASK_LEN>();

    for (size_t i = 0; i < axes->Size(); i++) {
        int64_t curDim = (*axes)[i];
        if (curDim >= dataDimNum || curDim < (-dataDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Provided axes %ld not in the range of input tensor size %ld.", curDim,
                dataDimNum);
            return false;
        }
        uint64_t index = GetPosDim(curDim, dataDimNum);
        if (axesMask[index]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Axes %lu appears multiple times in the list of axes", index);
        }
        axesMask.set(index);
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* data, const aclIntArray* axes, const aclTensor* reduce)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(data, axes, reduce), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查data、reduce的数据类型是否合法
    CHECK_RET(CheckDtypeValid(data, reduce), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查最大维度是否超过8
    CHECK_RET(CheckMaxDimension(data), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出data维度范围
    CHECK_RET(CheckAxesValid(data, axes), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* reduce, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape;
    size_t axesNum = reduce->GetViewShape().GetDimNum();

    if (reduce->IsEmpty()) {
        return ACLNN_SUCCESS;
    }

    for (size_t idx = 0; idx < axesNum; idx++) {
        int64_t tmpVal = reduce->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }

    auto axes = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), reduce->GetDataType());
    auto fillOut = l0op::Fill(axes, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, reduce, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReduceLogSumGetWorkspaceSize(
    const aclTensor* data, const aclIntArray* axes, bool keepDims, bool noopWithEmptyAxes, aclTensor* reduce,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnReduceLogSum, DFX_IN(data, axes, keepDims, noopWithEmptyAxes), DFX_OUT(reduce));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(data, axes, reduce);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入self为空tensor时，直接返回dtype类型的空tensor
    if (data->IsEmpty()) {
        ret = FillScalar(reduce, 0.0f, uniqueExecutor.get());
        if (ret == ACLNN_SUCCESS) {
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
            uniqueExecutor.ReleaseTo(executor);
        }
        return ret;
    }

    op::Shape shape = data->GetViewShape();

    //固定写法，将输入的data转换成连续的tensor
    auto dataContiguous = l0op::Contiguous(data, uniqueExecutor.get());
    const aclTensor* reduceOut = nullptr;

    CHECK_RET(dataContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (axes->Size() == 0) {
        if (noopWithEmptyAxes == false) {
            size_t axesDum = shape.GetDimNum();
            std::vector<int64_t> appendDim(axesDum);
            for (size_t i = 0; i < axesDum; i++) {
                appendDim[i] = i;
            }
            axes = uniqueExecutor.get()->AllocIntArray(appendDim.data(), axesDum);
            reduceOut = l0op::ReduceLogSum(dataContiguous, axes, keepDims, uniqueExecutor.get());
        } else {
            //固定写法，将计算结果拷贝到输出reduce上，reduce可能是非连续的tensor
            auto viewCopyResult = l0op::ViewCopy(dataContiguous, reduce, uniqueExecutor.get());
            CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            //固定写法，获取计算过程中需要使用的workspace大小
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    } else {
        reduceOut = l0op::ReduceLogSum(dataContiguous, axes, keepDims, uniqueExecutor.get());
    }

    CHECK_RET(reduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(reduceOut, reduce), ACLNN_ERR_PARAM_INVALID);

    //固定写法，将计算结果拷贝到输出reduce上，reduce可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(reduceOut, reduce, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    //固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReduceLogSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReduceLogSum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
