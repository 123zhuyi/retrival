/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_searchsorted.h"
#include "searchsorted.h"
#include "aclnn_kernels/cast.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_DOUBLE};

static bool searchsortedDimsMatchedBeforeLastDim(const aclTensor* sortedSequence, const aclTensor* self)
{
    auto dimsSortedSeq = sortedSequence->GetViewShape();
    auto dimsSelf = self->GetViewShape();
    if (dimsSortedSeq.GetDimNum() != dimsSelf.GetDimNum()) {
        return false;
    }
    for (size_t dim = 0; dim + 1 < dimsSortedSeq.GetDimNum(); ++dim) {
        if (dimsSortedSeq.GetDim(dim) != dimsSelf.GetDim(dim)) {
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* sortedSequence, const aclTensor* self, const aclTensor* out,
                            const bool outInt32, const aclTensor* sorter)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(sortedSequence, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    auto outDtype = outInt32 ? op::DataType::DT_INT32 : op::DataType::DT_INT64;
    OP_CHECK_DTYPE_NOT_MATCH(out, outDtype, return false);

    if (sorter != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(sorter, op::DataType::DT_INT64, return false);
    }

    return true;
}

static bool CheckPromoteType(const op::DataType sortedSequenceDtype, const op::DataType selfDtype,
                             op::DataType promoteType)
{
    OP_CHECK(promoteType != DataType::DT_UNDEFINED,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sortedSequence dtype %s and self dtype %s can not promote dtype.",
                     op::ToString(sortedSequenceDtype).GetString(), op::ToString(selfDtype).GetString()),
             return false);

    OP_CHECK(CheckType(promoteType, DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promoteType dtype %s not in dtype support list [%s].",
                     op::ToString(promoteType).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString()),
             return false);

    return true;
}

static bool CheckNotNull(const aclTensor* sortedSequence, const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(sortedSequence, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckShape(const aclTensor* sortedSequence, const aclTensor* self, const aclTensor* out,
                       const aclTensor* sorter)
{
    auto sortedSequenceDimNum = sortedSequence->GetViewShape().GetDimNum();
    auto selfDimNum = self->GetViewShape().GetDimNum();
    OP_CHECK_MAX_DIM(sortedSequence, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    if (sorter != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(sortedSequence, sorter, return false);
    }

    if (selfDimNum == 1 && self->GetViewShape().GetDim(0) == 1) {
        OP_CHECK(sortedSequenceDimNum == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "self value can be a scalar only when sortedSequence tensor dimension is 1, "
                         "but we got sortedSequence tensor dim(%s) and self value's dim(%s).",
                         op::ToString(sortedSequence->GetViewShape()).GetString(),
                         op::ToString(self->GetViewShape()).GetString()),
                 return false);
    }

    OP_CHECK(sortedSequenceDimNum == 1 || searchsortedDimsMatchedBeforeLastDim(sortedSequence, self),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "sortedSequence tensor should be 1 dimension or the first N-1 dimension of sortedSequence tensor "
                     "and self value tensor must match, but we got sortedSequence tensor %s and self value tensor %s.",
                     op::ToString(sortedSequence->GetViewShape()).GetString(),
                     op::ToString(self->GetViewShape()).GetString()),
             return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* sortedSequence, const aclTensor* self, const aclTensor* out,
                               const bool outInt32, const aclTensor* sorter)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(sortedSequence, self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(sortedSequence, self, out, outInt32, sorter), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(sortedSequence, self, out, sorter), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查sortedSequence和self能否做数据类型推导
    op::DataType promoteType = op::PromoteType(sortedSequence->GetDataType(), self->GetDataType());
    CHECK_RET(CheckPromoteType(sortedSequence->GetDataType(), self->GetDataType(), promoteType),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor* sortedSequence, const aclTensor* self, bool outInt32,
                                              bool right, const aclTensor* sorter, aclTensor* out,
                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSearchSorted, DFX_IN(sortedSequence, self, outInt32, right, sorter), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(sortedSequence, self, out, outInt32, sorter);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (sortedSequence->IsEmpty() || self->IsEmpty() || out->IsEmpty()) {
        auto outContiguous = l0op::Contiguous(out, uniqueExecutor.get());
        CHECK_RET(outContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用ZerosLike算子kernel
        auto zeroOut = l0op::ZerosLike(outContiguous, uniqueExecutor.get());
        CHECK_RET(zeroOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyOut = l0op::ViewCopy(zeroOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (op::ToShapeVector(self->GetViewShape()).size() == 0) {
        int64_t shape = 1;
        self = l0op::Reshape(self, uniqueExecutor.get()->AllocIntArray(&shape, 1), uniqueExecutor.get());
    }
    auto promoteType = op::PromoteType(sortedSequence->GetDataType(), self->GetDataType());

    auto sortedSequenceContiguous = l0op::Contiguous(sortedSequence, uniqueExecutor.get());
    CHECK_RET(sortedSequenceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sortedSequenceContiguousCasted = l0op::Cast(sortedSequenceContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(sortedSequenceContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguousCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* sorterContiguous = nullptr;
    if (sorter != nullptr) {
        sorterContiguous = l0op::Contiguous(sorter, uniqueExecutor.get());
        CHECK_RET(sorterContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outDtype = outInt32 ? op::DataType::DT_INT32 : op::DataType::DT_INT64;
    auto SearchSortedOut = l0op::SearchSorted(sortedSequenceContiguousCasted, selfContiguousCasted, outDtype, right,
                                              sorterContiguous, uniqueExecutor.get());
    CHECK_RET(SearchSortedOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(SearchSortedOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor* sortedSequence, const aclScalar* self,
                                               const bool outInt32, const bool right, const aclTensor* sorter,
                                               aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSearchSorteds, DFX_IN(sortedSequence, self, outInt32, right, sorter), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const aclTensor* selfTensor = nullptr;
    try {
        float selfValue = self->ToFloat();
        selfTensor = uniqueExecutor.get()->ConvertToTensor(&selfValue, 1, sortedSequence->GetDataType());
    } catch (std::invalid_argument& e) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Exception when convert to tensor. exception is %s."
                "Convert to tensor failed. not support sortedSequence data type[%s].",
                e.what(), op::ToString(sortedSequence->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(selfTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK(!out->IsEmpty(), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of out tensor should be {1}."),
             return ACLNN_ERR_PARAM_INVALID);
    out->SetViewShape(op::Shape{1});
    uniqueExecutor.ReleaseTo(executor);
    return aclnnSearchSortedGetWorkspaceSize(sortedSequence, selfTensor, outInt32, right, sorter, out, workspaceSize,
                                             executor);
}

aclnnStatus aclnnSearchSorted(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSearchSorted);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnSearchSorteds(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSearchSorteds);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
