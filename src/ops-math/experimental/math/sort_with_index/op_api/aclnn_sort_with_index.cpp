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
 * \file aclnn_sort_with_index.cpp
 * \brief SortWithIndex L2 aclnn public interface implementation (experimental ascend910b native).
 *
 * Two-stage flow:
 *   CREATE_EXECUTOR -> CheckParams -> empty-tensor fast path -> Contiguous(x,index) ->
 *   l0op::SortWithIndex -> ViewCopy(y, sortedIndex) -> GetWorkspaceSize.
 *
 * Supported dtypes (D1=1A iteration-3 decision, spec.yaml v1.2): x in {float16, float, bfloat16,
 * int32}, index in {int32} (4 combinations). int64-index is NOT supported on 910B through aclnn: the
 * binary-select runtime forces sorted_index = int32 for this sort family on non-RegBase, so an int64
 * binary is never matched (docs/ITER3_DECISIONS.md problem 1). int32-value sort is only exact for
 * |x| <= 2^24 (float-domain sort key, documented limitation).
 */

#include "aclnn_sort_with_index.h"
#include "sort_with_index_l0.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_DIM = 8;

// D1=1A: 4 combinations = 4 value x 1 index (int32 only). int64-index removed (spec.yaml v1.2).
static const std::initializer_list<op::DataType> VALUE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* x, const aclTensor* index, const aclTensor* y, const aclTensor* sortedIndex)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(sortedIndex, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x, const aclTensor* index, const aclTensor* y, const aclTensor* sortedIndex)
{
    // value dtype: x / y must match and be in the supported value list.
    OP_CHECK_DTYPE_NOT_SUPPORT(x, VALUE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);

    // index dtype: index / sortedIndex must match and be in the supported index list.
    OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(sortedIndex, index->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* index, const aclTensor* y, const aclTensor* sortedIndex)
{
    OP_CHECK_MAX_DIM(x, MAX_DIM, return false);
    OP_CHECK_MAX_DIM(index, MAX_DIM, return false);

    // x and index must have the same shape (spec: shape consistency).
    OP_CHECK(
        x->GetViewShape() == index->GetViewShape(),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "x shape %s and index shape %s must be the same.",
            op::ToString(x->GetViewShape()).GetString(), op::ToString(index->GetViewShape()).GetString()),
        return false);

    // y shape == x shape; sortedIndex shape == index shape.
    OP_CHECK(
        y->GetViewShape() == x->GetViewShape(),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "y shape %s must equal x shape %s.", op::ToString(y->GetViewShape()).GetString(),
            op::ToString(x->GetViewShape()).GetString()),
        return false);
    OP_CHECK(
        sortedIndex->GetViewShape() == index->GetViewShape(),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "sortedIndex shape %s must equal index shape %s.",
            op::ToString(sortedIndex->GetViewShape()).GetString(), op::ToString(index->GetViewShape()).GetString()),
        return false);
    return true;
}

static bool CheckAxis(const aclTensor* x, int64_t axis)
{
    int64_t rank = static_cast<int64_t>(x->GetViewShape().GetDimNum());
    int64_t signedRank = (rank == 0) ? 1 : rank;
    // Only last-axis sort is supported (axis == -1 or axis == rank-1).
    OP_CHECK(
        axis >= -signedRank && axis < signedRank,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis %ld is out of range [%ld, %ld).", axis, -signedRank, signedRank),
        return false);
    int64_t normAxis = axis < 0 ? axis + signedRank : axis;
    OP_CHECK(
        normAxis == signedRank - 1,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "only last-axis sort is supported (axis must be -1 or rank-1). axis=%ld, rank=%ld.", axis, rank),
        return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x, const aclTensor* index, int64_t axis, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    CHECK_RET(CheckNotNull(x, index, valuesOut, indicesOut), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, index, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, index, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAxis(x, axis), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSortWithIndexGetWorkspaceSize(
    const aclTensor* x, const aclTensor* index, const int64_t axis, const bool descending, const bool stable, aclTensor* valuesOut,
    aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSortWithIndex, DFX_IN(x, index, axis, descending, stable), DFX_OUT(valuesOut, indicesOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, index, axis, valuesOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty tensor: nothing to compute, outputs stay empty.
    if (x->IsEmpty() || index->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make inputs contiguous (kernel requires contiguous inputs).
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
    CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // L0 launch: returns (sortedValues, sortedIndices).
    auto opResult = l0op::SortWithIndex(xContiguous, indexContiguous, axis, descending, stable, uniqueExecutor.get());
    auto sortedValues = std::get<0>(opResult);
    auto sortedIndices = std::get<1>(opResult);
    CHECK_RET(sortedValues != nullptr && sortedIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Copy back to user-provided outputs (handles non-contiguous user out tensors).
    auto valuesViewCopy = l0op::ViewCopy(sortedValues, valuesOut, uniqueExecutor.get());
    CHECK_RET(valuesViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesViewCopy = l0op::ViewCopy(sortedIndices, indicesOut, uniqueExecutor.get());
    CHECK_RET(indicesViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSortWithIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSortWithIndex);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
