/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sum.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "accumulate_nv2.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8, op::DataType::DT_INT32,
    op::DataType::DT_UINT8};

// 检查入参是否为nullptr
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
                ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list [%s].", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
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
        // 检查输入tensors里的tensor是否都满足broadcast规则。
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensors can not broadcast.");
        return false;
    }
    // 输出shape应该等于输入tensors broadcast之后的shape
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(tensors, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输出shape
    CHECK_RET(CheckShape(tensors, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus SplitToSumN(
    const aclTensorList* tensors, const aclIntArray* broadcastShapeArray, const aclTensor** sumOut,
    aclOpExecutor* executor)
{
    size_t MAX_TENSOR_SIZE = 16;
    // 将输入转换成连续的tensor, 调用BroadcastTo
    op::FVector<const aclTensor*> tensorList;
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        auto contiguousOut = l0op::Contiguous((*tensors)[i], executor);
        CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto broadcastOut = l0op::BroadcastTo(contiguousOut, broadcastShapeArray, executor);
        CHECK_RET(broadcastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorList.push_back(broadcastOut);
    }

    // 调用L0 AccumulateNv2算子计算
    while (tensorList.size() >= MAX_TENSOR_SIZE) {
        op::FVector<const aclTensor*> tensorListOnes;
        tensorListOnes.assign(tensorList.end() - MAX_TENSOR_SIZE, tensorList.end());
        auto onesComputeOut =
            l0op::AccumulateNv2(executor->AllocTensorList(tensorListOnes.data(), tensorListOnes.size()), executor);
        CHECK_RET(onesComputeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorList.erase(tensorList.end() - MAX_TENSOR_SIZE, tensorList.end());
        tensorList.push_back(onesComputeOut);
    }
    *sumOut = tensorList[0];
    if (tensorList.size() > 1) {
        *sumOut = l0op::AccumulateNv2(executor->AllocTensorList(tensorList.data(), tensorList.size()), executor);
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

aclnnStatus aclnnSumGetWorkspaceSize(
    const aclTensorList* tensors, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSum, DFX_IN(tensors), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(tensors, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if ((*tensors)[0]->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto reshapeTensors = SumAdaptInputZeroDimTensor(tensors, uniqueExecutor.get());
    CHECK_RET(reshapeTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 输入tensors支持broadcast
    op::Shape broadcastShape = (*reshapeTensors)[0]->GetViewShape();
    for (uint64_t i = 1; i < reshapeTensors->Size(); i++) {
        BroadcastInferShape((*reshapeTensors)[i]->GetViewShape(), broadcastShape, broadcastShape);
    }
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
    auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用AccumulateNv2算子计算
    const aclTensor* sumOut = nullptr;
    aclnnStatus retSplit = SplitToSumN(reshapeTensors, broadcastShapeArray, &sumOut, uniqueExecutor.get());
    CHECK_RET(retSplit == ACLNN_SUCCESS, retSplit);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(sumOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
