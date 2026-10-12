/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_unfold_grad.h"
#include "unfold_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/transdata.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MIN_SUPPORT_DIMS_NUMS_FOR_UNFOLD_GRAD = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,  op::DataType::DT_BF16};

static const aclTensor* GetInputSizesTensor(const aclIntArray* inputSizes, aclOpExecutor* executor)
{
    FVector<int64_t, op::MAX_DIM_NUM> inputSizesVector;
    size_t dimNum = static_cast<size_t>(inputSizes->Size());
    for (size_t i = 0; i<dimNum; i++) {
        inputSizesVector.emplace_back((*inputSizes)[i]);
    }
    auto newInputSizes = executor->AllocIntArray(inputSizesVector.data(), dimNum);
    auto inputSizesTensor = executor->ConvertToTensor(newInputSizes, static_cast<op::DataType>(ACL_INT64));
    return inputSizesTensor;
}

static bool CheckNotNull(const aclTensor *gradOut, const aclIntArray *inputSizes, const aclTensor *gradIn) {
  OP_CHECK_NULL(gradOut, return false);
  OP_CHECK_NULL(inputSizes, return false);
  OP_CHECK_NULL(gradIn, return false);
  return true;
}

static bool CheckShapeValid(const aclTensor *gradOut, const aclIntArray *inputSizes, const aclTensor *gradIn)
{
    // 检查维度是否小于等于8
    OP_CHECK_MAX_DIM(gradOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradIn, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查维度是否大于等于1
    OP_CHECK_MIN_DIM(gradOut, MIN_SUPPORT_DIMS_NUMS_FOR_UNFOLD_GRAD, return false);
    OP_CHECK_MIN_DIM(gradIn, MIN_SUPPORT_DIMS_NUMS_FOR_UNFOLD_GRAD, return false);

    // 检查gradOut的维度是否等于inputSizes的维度 + 1
    uint64_t gradOutDimNum = static_cast<uint64_t>(gradOut->GetViewShape().GetDimNum());

    OP_CHECK(
        inputSizes->Size() == gradOutDimNum - 1,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
                "inputSizes length should be equal to Dim of gradOut - 1, but got %zu.", 
                inputSizes->Size()), 
            return false);

    op::Shape expectShape;
    expectShape.SetDimNum(inputSizes->Size());
    for (size_t i = 0; i < inputSizes->Size(); i++) {
        OP_LOGD("aclnnUnfoldGradGetWorkspaceSize:inputSizes[%zu]=%ld.\n", i, (*inputSizes)[i]);
        expectShape.SetDim(i, (*inputSizes)[i]);
    }

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradIn, expectShape, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *gradIn) {
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(gradIn, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckFormat(const aclTensor* gradOut)
{
    // 检查format，若是NZ格式，则拦截
    if (gradOut->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradOut gets [%s], this format not support.",
        op::ToString(gradOut->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOut, const aclIntArray* inputSizes, int64_t dim, int64_t size, int64_t step, 
    const aclTensor* gradIn) {
  CHECK_RET(CheckNotNull(gradOut, inputSizes, gradIn), ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(CheckDtypeValid(gradOut, gradIn), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShapeValid(gradOut, inputSizes, gradIn), ACLNN_ERR_PARAM_INVALID);
 
  int64_t dimNum = gradOut->GetViewShape().GetDimNum();
  CHECK_RET(dim >= 0 && dim < dimNum-1, ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(size > 0, ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(step > 0, ACLNN_ERR_PARAM_INVALID);

  // 检查format，若是NZ格式，则拦截
  CHECK_RET(CheckFormat(gradOut), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnfoldGradGetWorkspaceSize(
    const aclTensor* gradOut, const aclIntArray* inputSizes, int64_t dim, int64_t size, int64_t step, 
    const aclTensor* gradIn, uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
    L2_DFX_PHASE_1(aclnnUnfoldGrad, DFX_IN(gradOut, inputSizes, dim, size, step), DFX_OUT(gradIn));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, inputSizes, dim, size, step, gradIn);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOut->IsEmpty() || gradIn->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto gradOutContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputSizesTensor = GetInputSizesTensor(inputSizes, uniqueExecutor.get());
    CHECK_RET(inputSizesTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto inputSizesTensorContiguous = l0op::Contiguous(inputSizesTensor, uniqueExecutor.get());
    CHECK_RET(inputSizesTensorContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto unfoldGradOut = l0op::UnfoldGrad(gradOutContiguous, inputSizesTensorContiguous, dim, size, step, uniqueExecutor.get());
    CHECK_RET(unfoldGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outCast = l0op::Cast(unfoldGradOut, gradOutContiguous->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outReformat = l0op::ReFormat(outCast, gradIn->GetStorageFormat());
    CHECK_RET(outReformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(outReformat, gradIn, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnfoldGrad(void *workspace, uint64_t workspaceSize,
                          aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnUnfoldGrad);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif

