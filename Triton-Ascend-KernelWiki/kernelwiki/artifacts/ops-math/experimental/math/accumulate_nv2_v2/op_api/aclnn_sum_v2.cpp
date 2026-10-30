/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "aclnn_sum_v2.h"
#include "accumulate_nv2_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_UINT8};

static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* out)
{
    OP_CHECK_NULL(tensors, return false);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if ((*tensors)[i] == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a proper Tensor but got null for tensor %lu.", i);
            return false;
        }
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), DTYPE_SUPPORT_LIST)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s.", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool GetTensorsBroadcastShape(const aclTensorList* tensors, op::Shape& broadcastShape)
{
    broadcastShape = (*tensors)[0]->GetViewShape();
    for (uint64_t i = 1; i < tensors->Size(); ++i) {
        if (!BroadcastInferShape((*tensors)[i]->GetViewShape(), broadcastShape, broadcastShape)) {
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); ++i) {
        auto dimNum = (*tensors)[i]->GetViewShape().GetDimNum();
        if (dimNum > MAX_SUPPORT_DIMS_NUMS) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Dim of tensor %lu is %zu, can't be greater than %zu.", i, dimNum,
                MAX_SUPPORT_DIMS_NUMS);
            return false;
        }
    }
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    op::Shape broadcastShape;
    if (!GetTensorsBroadcastShape(tensors, broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensors can not broadcast.");
        return false;
    }
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(tensors, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus SplitToSumN(
    const aclTensorList* tensors, const aclIntArray* broadcastShapeArray, const aclTensor** sumOut,
    aclOpExecutor* executor, bool promoteFp16)
{
    size_t MAX_TENSOR_SIZE = 16;
    op::FVector<const aclTensor*> tensorList;
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        auto contiguousOut = l0op::Contiguous((*tensors)[i], executor);
        CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto broadcastOut = l0op::BroadcastTo(contiguousOut, broadcastShapeArray, executor);
        CHECK_RET(broadcastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (promoteFp16) {
            broadcastOut = l0op::Cast(broadcastOut, op::DataType::DT_FLOAT, executor);
            CHECK_RET(broadcastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        tensorList.push_back(broadcastOut);
    }

    while (tensorList.size() >= MAX_TENSOR_SIZE) {
        op::FVector<const aclTensor*> tensorListOnes;
        tensorListOnes.assign(tensorList.end() - MAX_TENSOR_SIZE, tensorList.end());
        auto onesComputeOut =
            l0op::AccumulateNv2V2(executor->AllocTensorList(tensorListOnes.data(), tensorListOnes.size()), executor);
        CHECK_RET(onesComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorList.erase(tensorList.end() - MAX_TENSOR_SIZE, tensorList.end());
        tensorList.push_back(onesComputeOut);
    }
    *sumOut = tensorList[0];
    if (tensorList.size() > 1) {
        *sumOut = l0op::AccumulateNv2V2(executor->AllocTensorList(tensorList.data(), tensorList.size()), executor);
    }
    CHECK_RET(sumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclTensorList* SumAdaptInputZeroDimTensor(const aclTensorList* tensors, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> fTensorList;
    int64_t selfShapeValue[1] = {1};
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, 1);
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        auto oneTensor = (*tensors)[i];
        if (oneTensor->GetViewShape().GetDimNum() == 0) {
            auto reshapeTensor = l0op::Reshape(oneTensor, selfShape, executor);
            CHECK_RET(reshapeTensor != nullptr, nullptr);
            fTensorList.push_back(reshapeTensor);
        } else {
            fTensorList.push_back(oneTensor);
        }
    }
    return executor->AllocTensorList(fTensorList.data(), fTensorList.size());
}

aclnnStatus aclnnSumV2GetWorkspaceSize(
    const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSumV2, DFX_IN(tensors), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(tensors, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if ((*tensors)[0]->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto reshapeTensors = SumAdaptInputZeroDimTensor(tensors, uniqueExecutor.get());
    CHECK_RET(reshapeTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    op::Shape broadcastShape = (*reshapeTensors)[0]->GetViewShape();
    for (uint64_t i = 1; i < reshapeTensors->Size(); i++) {
        BroadcastInferShape((*reshapeTensors)[i]->GetViewShape(), broadcastShape, broadcastShape);
    }
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
    auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // For fp16 inputs, promote to fp32 for accumulation precision and to avoid
    // built-in kernel issues with large fp16 shapes
    bool promoteFp16 = ((*reshapeTensors)[0]->GetDataType() == op::DataType::DT_FLOAT16);

    const aclTensor* sumOut = nullptr;
    aclnnStatus retSplit = SplitToSumN(reshapeTensors, broadcastShapeArray, &sumOut, uniqueExecutor.get(), promoteFp16);
    CHECK_RET(retSplit == ACLNN_SUCCESS, retSplit);

    // Cast result back to fp16 if promoted
    if (promoteFp16) {
        sumOut = l0op::Cast(sumOut, op::DataType::DT_FLOAT16, uniqueExecutor.get());
        CHECK_RET(sumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(sumOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSumV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSumV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
