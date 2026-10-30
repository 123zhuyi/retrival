/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_batch_norm_stats.h"
#include "aclnn_kernels/cast.h"
#include "conversion/concat_d/op_api/concat_d.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "math/reduce_mean/op_api/reduce_mean.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "reduce_std_with_mean.h"
#include "math/reduce_var/op_api/reduce_var.h"
#include "math/add/op_api/add.h"
#include "math/pow/op_api/pow.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_check.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr float NEGTIVE_SQRT_EXP = -0.5f;

static aclTensor* FillVector(const op::Shape dstShape, const aclTensor* src, float value, aclOpExecutor* executor) {
  op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(dstShape);
  auto shapes = executor->AllocIntArray(fillDims.data(), src->GetViewShape().GetDimNum());
  const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT32);
  const aclScalar* valueScalar = executor->AllocScalar(value);
  const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, src->GetDataType());
  auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
  if (fillTensor == nullptr) {
    return nullptr;
  }
  fillTensor = l0op::ReFormat(fillTensor, op::Format::FORMAT_ND);
  return const_cast<aclTensor*>(fillTensor);
}

static aclnnStatus ProcessEmptyTensorWithValue(aclTensor* src, float initValue, aclOpExecutor* executor) {
  auto srcShape = src->GetViewShape();
  auto dst = FillVector(srcShape, src, initValue, executor);
  auto dstCopyResult = l0op::ViewCopy(dst, src, executor);
  CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (IsRegBase()) {
        return ASCEND950_DTYPE_SUPPORT_LIST;
    } else {
        return DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* input, aclTensor* mean, aclTensor* invstd)
{
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mean, {op::DataType::DT_FLOAT}, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(invstd, {op::DataType::DT_FLOAT}, return false);
    return true;
}

static bool CheckNotNull(const aclTensor* input, aclTensor* mean, aclTensor* invstd)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(mean, return false);
    OP_CHECK_NULL(invstd, return false);
    return true;
}

static bool CheckFormat(const aclTensor* input, aclTensor* mean, aclTensor* invstd)
{
    if (op::IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(mean->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Mean format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    if (op::IsPrivateFormat(invstd->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invstd format only support ND, NCL, NCHW, NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* input, aclTensor* mean, aclTensor* invstd)
{
    OP_CHECK_MAX_DIM(input, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(input, BN_MIN_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_WRONG_DIMENSION(mean, 1, return false);
    OP_CHECK_WRONG_DIMENSION(invstd, 1, return false);
    if (input->GetViewShape()[1] == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Channel size of input should not be zero.");
        return false;
    }
    op::Shape expectShape = {input->GetViewShape()[1]};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(mean, expectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(invstd, expectShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, aclTensor* mean, aclTensor* invstd)
{
    CHECK_RET(CheckNotNull(input, mean, invstd), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(input, mean, invstd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(input, mean, invstd), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(input, mean, invstd), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnBatchNormStatsImplUnify(
    const aclTensor* input, const aclIntArray* dim, double eps, aclTensor* meanOut, aclTensor* invstdOut,
    uint64_t* workspaceSize, UniqueExecutor& uniqueExecutor, aclOpExecutor** executor)
{
    int64_t correction = 0;
    bool keepdim = false;
    bool isMeanOut = true;
    auto reduceVarOut = l0op::ReduceVar(input, dim, correction, keepdim, isMeanOut, uniqueExecutor.get());

    auto varOpOut = std::get<0>(reduceVarOut);
    CHECK_RET(varOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto epsScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(eps));
    CHECK_RET(epsScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto epsTensor = uniqueExecutor.get()->ConvertToTensor(epsScalar, op::DataType::DT_FLOAT);
    CHECK_RET(epsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto addOpOut = l0op::Add(varOpOut, epsTensor, uniqueExecutor.get());
    CHECK_RET(addOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto expScalar = uniqueExecutor.get()->AllocScalar(NEGTIVE_SQRT_EXP);
    CHECK_RET(expScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto expTensor = uniqueExecutor.get()->ConvertToTensor(expScalar, op::DataType::DT_FLOAT);
    CHECK_RET(expTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 调用pow进行计算
    auto powOut = l0op::Pow(addOpOut, expTensor, uniqueExecutor.get());

    auto viewCopyResult = l0op::ViewCopy(powOut, invstdOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto meanOpOut = std::get<1>(reduceVarOut);
    CHECK_RET(meanOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(meanOpOut, meanOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(
    const aclTensor* input, double eps, aclTensor* meanOut, aclTensor* invstdOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBatchNormStats, DFX_IN(input, eps), DFX_OUT(meanOut, invstdOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(input, meanOut, invstdOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input != nullptr && input->IsEmpty()) {
        ret = ProcessEmptyTensorWithValue(meanOut, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = ProcessEmptyTensorWithValue(invstdOut, std::numeric_limits<float>::quiet_NaN(), uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto contiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto reformat = l0op::ReFormat(castOut, Format::FORMAT_ND);
    CHECK_RET(reformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t> dimIndexNoC;
    FVector<int64_t> dimAll;
    op::Shape shapeIn = input->GetViewShape();
    for (size_t idx = 0; idx < shapeIn.GetDimNum(); idx++) {
        if (idx != 1) {
            dimIndexNoC.push_back(static_cast<int64_t>(idx));
        }
        dimAll.push_back(shapeIn.GetDim(idx));
    }
    aclIntArray* axes = uniqueExecutor.get()->AllocIntArray(dimIndexNoC.data(), dimIndexNoC.size());
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsRegBase()) {
        return aclnnBatchNormStatsImplUnify(
            reformat, axes, eps, meanOut, invstdOut, workspaceSize, uniqueExecutor, executor);
    }

    auto reduceMeanResult = l0op::ReduceMean(reformat, axes, false, uniqueExecutor.get());
    CHECK_RET(reduceMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto view_copy_mean = l0op::ViewCopy(reduceMeanResult, meanOut, uniqueExecutor.get());
    CHECK_RET(view_copy_mean != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto unsqueeze = l0op::UnsqueezeNd(reduceMeanResult, axes, uniqueExecutor.get());
    CHECK_RET(unsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclIntArray* shape = uniqueExecutor.get()->AllocIntArray(dimAll.data(), dimAll.size());
    auto broadcast = l0op::BroadcastTo(unsqueeze, shape, uniqueExecutor.get());
    CHECK_RET(broadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t correction = 0;
    auto reduceStdWithMeanResult = l0op::ReduceStdWithMean(
        reformat, broadcast, axes, correction, false, true, static_cast<float>(eps), uniqueExecutor.get());
    CHECK_RET(reduceStdWithMeanResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto view_copy_invstd = l0op::ViewCopy(reduceStdWithMeanResult, invstdOut, uniqueExecutor.get());
    CHECK_RET(view_copy_invstd != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBatchNormStats(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBatchNormStats);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
