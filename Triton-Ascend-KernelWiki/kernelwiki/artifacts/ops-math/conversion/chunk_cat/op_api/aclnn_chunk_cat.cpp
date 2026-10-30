/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_cat.cpp
 * \brief
 */
#include "aclnn_chunk_cat.h"
#include "chunk_cat.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
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
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include <iostream>


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t CAT_INPUT_NUM = 512;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch npuArch)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& dTypeSupportList = GetSupportDtypeList(npuArch);
    op::DataType inputType = (*tensors)[0]->GetDataType();
    if (!CheckType(inputType, dTypeSupportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list %s.", 0,
            op::ToString(inputType).GetString(), op::ToString(dTypeSupportList).GetString());
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        if ((*tensors)[i]->GetDataType() != inputType) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expects all input tensors with the same dtype.");
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dTypeSupportList, return false);
    if (inputType == DataType::DT_FLOAT && out->GetDataType() != DataType::DT_FLOAT) {    
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output dtype must be float when input dtype is float.");
        return false;
    }
    return true;
}

static bool CheckNotNull(const aclTensorList* tensors, const aclTensor* out)
{
    OP_CHECK_NULL(tensors, return false);

    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_NULL((*tensors)[i], return false);
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckFormat(const aclTensorList* tensors, const aclTensor* out)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        op::Format format = (*tensors)[i]->GetStorageFormat();
        if (op::IsPrivateFormat(format)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
            return false;
        }
    }
    if (op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensorList* tensors)
{
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        OP_CHECK_MAX_DIM((*tensors)[i], MAX_SUPPORT_DIMS_NUMS, return false);
        op::Shape shape = (*tensors)[i]->GetViewShape();
        if (shape.GetDimNum() == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input tensor %lu dimnum is 0.", i);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(tensors, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* MergeLastDims(const aclTensor* tensor, int64_t dim, aclOpExecutor* executor) {
    op::Shape shapeTensor = tensor->GetViewShape();
    int64_t dimNum = shapeTensor.GetDimNum();
    
    op::Shape newShape;
    for (int64_t i = 0; i <= dim; i++) {
        newShape.AppendDim(static_cast<int64_t>(shapeTensor.GetDim(i)));
    }
    int64_t catdimSize = 1;
    for (int64_t i = dim + 1; i < dimNum; i++) {
        catdimSize *= shapeTensor.GetDim(i);
    }
    newShape.AppendDim(catdimSize);
    auto reshapeTensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    reshapeTensor->SetViewShape(newShape);
    reshapeTensor->SetOriginalShape(newShape);
    reshapeTensor->SetStorageShape(newShape);
    return reshapeTensor;
}

const aclTensor* PostProcess(const aclTensor* tensor, const aclTensor* out, aclOpExecutor* executor) {
    auto reFormatTensor = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetViewOffset());
    reFormatTensor->SetViewFormat(out->GetViewFormat());
    reFormatTensor->SetOriginalFormat(out->GetOriginalFormat());
    reFormatTensor->SetStorageFormat(out->GetStorageFormat());
    return reFormatTensor;
}

static aclnnStatus ProcessOneTensor(const aclTensorList* tensors, int64_t dim, int64_t numChunks,
                                    aclTensor* out, aclOpExecutor* executor)
{
    auto concatTensor = l0op::ChunkCat(tensors, dim, numChunks, out->GetDataType(), executor);
    CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(concatTensor, out), ACLNN_ERR_PARAM_INVALID);
    auto outTensor = PostProcess(concatTensor, out, executor);
    auto viewCopyResult = l0op::ViewCopy(outTensor, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus SplitToChunkCat(const aclTensorList* tensors, int64_t dim, int64_t numChunks,
                                 aclTensor* out, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> tensorListA;
    auto outType = out ->GetDataType();
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!(*tensors)[i]->IsEmpty()) {
            auto contiguous = l0op::Contiguous((*tensors)[i], executor);
            CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            contiguous = MergeLastDims(contiguous, dim, executor);
            tensorListA.emplace_back(contiguous);
        }
    }
    if (tensorListA.size() == 1) {
        auto tensorList = executor->AllocTensorList(tensorListA.data(), tensorListA.size());
        return ProcessOneTensor(tensorList, dim, numChunks, out, executor);
    }
    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorListB;
        for (auto tensor : tensorListA) {
            tensorListOnce.emplace_back(tensor);
            if (tensorListOnce.size() == CAT_INPUT_NUM) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                auto concatTensor = l0op::ChunkCat(tensorList, dim, numChunks, outType, executor);
                CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListB.emplace_back(concatTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
            auto concatTensorTail = l0op::ChunkCat(aclTensorListTail, dim, numChunks, outType, executor);
            CHECK_RET(concatTensorTail != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensorListB.emplace_back(concatTensorTail);
            tensorListOnce.clear();
        }
        tensorListA = tensorListB;
    }
    if (tensorListA.empty()) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckShapeAndScalarSame(tensorListA.front(), out), ACLNN_ERR_PARAM_INVALID);
    auto outTensor = PostProcess(tensorListA.front(), out, executor);
    auto viewCopyResult = l0op::ViewCopy(outTensor, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkCatGetWorkspaceSize(
    const aclTensorList* tensors, int64_t dim, int64_t numChunks, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnChunkCat, DFX_IN(tensors, dim, numChunks), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (dim != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim only support 0 now.");
    }
    if (tensors->Size() == 0) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto ret = CheckParams(tensors, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = SplitToChunkCat(tensors, dim, numChunks, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkCat(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkCat);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
