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
 * \file aclnn_sort_with_index.h
 * \brief SortWithIndex L2 aclnn public interface (experimental ascend910b native).
 *
 * Two-stage interface (signature aligned with docs/aclnnSortWithIndex.md):
 *   1. aclnnSortWithIndexGetWorkspaceSize - param check + workspace size + executor
 *   2. aclnnSortWithIndex                 - execute
 *
 * 910B first-version scope (spec.yaml v1.2): value dtype = {float16, float32, bfloat16, int32},
 * index dtype = int32 only (4 supported combinations). int64-index is not exposed on 910B
 * (framework forces int32 index for the sort family on non-RegBase); int64 kernel code is retained
 * but not registered. See docs/ITER3_DECISIONS.md (D1=1A).
 */
#ifndef OP_API_INC_ACLNN_SORT_WITH_INDEX_H_
#define OP_API_INC_ACLNN_SORT_WITH_INDEX_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnSortWithIndex first-stage interface: compute workspace size and create executor.
 * @param [in] x: value tensor to sort, ND format.
 * @param [in] index: index tensor following the sort, ND format. Same shape as x.
 * @param [in] axis: sort axis. Only the last axis is supported (-1 or rank-1). Default -1.
 * @param [in] descending: true = descending, false = ascending. Default false.
 * @param [in] stable: true = stable sort (equal elements keep original order). Default false.
 * @param [out] valuesOut: sorted value tensor. Same shape/dtype as x.
 * @param [out] indicesOut: index tensor permuted by x's sort order. Same shape/dtype as index.
 * @param [out] workspaceSize: returned device workspace size in bytes.
 * @param [out] executor: returned op executor containing the compute flow.
 * @return aclnnStatus status code.
 */
ACLNN_API aclnnStatus aclnnSortWithIndexGetWorkspaceSize(const aclTensor* x, const aclTensor* index, const int64_t axis,
                                                         const bool descending, const bool stable, aclTensor* valuesOut,
                                                         aclTensor* indicesOut, uint64_t* workspaceSize,
                                                         aclOpExecutor** executor);

/**
 * @brief aclnnSortWithIndex second-stage interface: execute the SortWithIndex op.
 * @param [in] workspace: device workspace memory address.
 * @param [in] workspaceSize: device workspace size from the first-stage interface.
 * @param [in] executor: op executor from the first-stage interface.
 * @param [in] stream: execution stream.
 * @return aclnnStatus status code.
 */
ACLNN_API aclnnStatus aclnnSortWithIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_ACLNN_SORT_WITH_INDEX_H_
