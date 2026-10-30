/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_stack.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_check.h"
#include "../../concat_d/op_api/concat_d.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "pack.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,      op::DataType::DT_INT16,     op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,     op::DataType::DT_UINT16,    op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,     op::DataType::DT_BOOL,   op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8,      op::DataType::DT_INT16,      op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_UINT8,     op::DataType::DT_UINT16,     op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,      op::DataType::DT_BOOL,   op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensorList* tensors, const int64_t* realDim, const aclTensor* out)
{
    if (tensors == nullptr || realDim == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Input of aclnnStack should not be null.");
        return false;
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensorList* tensors, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    for (uint64_t i = 0; i < tensors->Size(); i++) {
        if (!CheckType((*tensors)[i]->GetDataType(), supportList)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor %lu not implemented for %s, should be in dtype support list [%s].", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(supportList).GetString());
            return false;
        }
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    return true;
}

static bool CheckPromoteType(const aclTensorList* tensors, const aclTensor* out)
{
    op::DataType promoteType = (*tensors)[0]->GetDataType();
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        if (promoteType == DataType::DT_UNDEFINED) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "tensor %lu dtype %s and dtype %s can not promote dtype.", i,
                op::ToString((*tensors)[i]->GetDataType()).GetString(), op::ToString(promoteType).GetString());
            return false;
        }
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensorList* tensors, int64_t* realDim)
{
    op::Shape shapeFirst = (*tensors)[0]->GetViewShape();
    auto dimNum = (int64_t)shapeFirst.GetDimNum();
    static const int64_t MAX_DIM_LEN = 8;
    if (dimNum > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self is %ld, can't be greater than %ld.", dimNum, MAX_DIM_LEN);
        return false;
    }
    auto originDim = *realDim;
    if (*realDim < 0) {
        (*realDim) += dimNum + 1;
    }
    if ((*realDim) < 0 || (*realDim) > dimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "dimnum %ld exceed the dim range of the [%ld, %ld].", originDim, -(dimNum + 1),
            dimNum);
        return false;
    }
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        op::Shape shapeCurrent = (*tensors)[i]->GetViewShape();
        if (dimNum != (int64_t)shapeCurrent.GetDimNum()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "dimnum of tensor %lu is [%zu], should be equal to tensor 0 [%ld].", i,
                shapeCurrent.GetDimNum(), dimNum);
            return false;
        }
        for (int64_t j = 0; j < dimNum; j++) {
            if (shapeFirst.GetDim(j) != shapeCurrent.GetDim(j)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "dim %ld of tensor %lu is [%ld], should be equal to tensor 0 [%ld].", j, i,
                    shapeCurrent.GetDim(j), shapeFirst.GetDim(j));
                return false;
            }
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensorList* tensors, int64_t* realDim, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(tensors, realDim, out), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckDtypeValid(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckPromoteType(tensors, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(tensors, realDim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool EmplaceTensorList(
    const aclTensorList* tensors, op::FVector<const aclTensor*>& tensorListA, DataType& promoteType)
{
    tensorListA.emplace_back((*tensors)[0]);
    for (uint64_t i = 1; i < tensors->Size(); i++) {
        promoteType = op::PromoteType((*tensors)[i]->GetDataType(), promoteType);
        tensorListA.emplace_back((*tensors)[i]);
    }
    return true;
}

static aclnnStatus SplitToStack(
    const aclTensorList* tensors, int64_t dim, aclOpExecutor* executor, const aclTensor** out)
{
    size_t maxInputs = 32;
    op::FVector<const aclTensor*> tensorListA;
    auto promoteType = (*tensors)[0]->GetDataType();
    bool firstLoop = EmplaceTensorList(tensors, tensorListA, promoteType);
    while (tensorListA.size() > 1) {
        op::FVector<const aclTensor*> tensorListOnce;
        op::FVector<const aclTensor*> tensorLlistB;
        for (auto tensor : tensorListA) {
            if (tensor->IsEmpty()) {
                continue;
            }
            if (firstLoop) {
                auto contiguous = l0op::Contiguous(tensor, executor);
                CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
                auto castOut = l0op::Cast(contiguous, promoteType, executor);
                CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorListOnce.emplace_back(castOut);
            } else {
                tensorListOnce.emplace_back(tensor);
            }
            if (tensorListOnce.size() == maxInputs) {
                auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
                const aclTensor* stackTensor = nullptr;
                if (firstLoop) {
                    stackTensor = l0op::Pack(tensorList, dim, promoteType, executor);
                } else {
                    stackTensor = l0op::ConcatD(tensorList, dim, promoteType, executor);
                }
                CHECK_RET(stackTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
                tensorLlistB.emplace_back(stackTensor);
                tensorListOnce.clear();
            }
        }
        if (!tensorListOnce.empty()) {
            auto aclTensorListTail = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
            const aclTensor* stackTensorTail = nullptr;
            if (firstLoop) {
                stackTensorTail = l0op::Pack(aclTensorListTail, dim, promoteType, executor);
            } else {
                stackTensorTail = l0op::ConcatD(aclTensorListTail, dim, promoteType, executor);
            }
            CHECK_RET(stackTensorTail != nullptr, ACLNN_ERR_INNER_NULLPTR);
            tensorLlistB.emplace_back(stackTensorTail);
            tensorListOnce.clear();
        }
        tensorListA = tensorLlistB;
        firstLoop = false;
    }
    *out = tensorListA.empty() ? nullptr : tensorListA.front();
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStackGetWorkspaceSize(
    const aclTensorList* tensors, int64_t dim, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnStack, DFX_IN(tensors, dim), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    int64_t realDim = dim;
    auto ret = CheckParams(tensors, &realDim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* castIn = nullptr;
    if (tensors->Size() == 1) {
        if ((*tensors)[0]->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
        auto contiguous = l0op::Contiguous((*tensors)[0], uniqueExecutor.get());
        CHECK_RET(contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        op::FVector<const aclTensor*> tensorList{contiguous};
        auto oneTensor = uniqueExecutor.get()->AllocTensorList(tensorList.data(), tensorList.size());
        castIn = l0op::Pack(oneTensor, realDim, (*tensors)[0]->GetDataType(), uniqueExecutor.get());
    } else {
        aclnnStatus retSplit = SplitToStack(tensors, realDim, uniqueExecutor.get(), &castIn);
        CHECK_RET(retSplit == ACLNN_SUCCESS, retSplit);
    }
    if (castIn == nullptr) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckShapeAndScalarSame(castIn, out), ACLNN_ERR_PARAM_INVALID);

    auto castOut = l0op::Cast(castIn, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnStack(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnStack);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
