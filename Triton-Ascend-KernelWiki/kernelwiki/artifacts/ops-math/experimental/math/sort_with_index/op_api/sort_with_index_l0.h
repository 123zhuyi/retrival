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
 * \file sort_with_index_l0.h
 * \brief SortWithIndex L0 (internal) op-api interface (experimental ascend910b native).
 *
 * L0 responsibilities: output shape derivation + Kernel launch via ADD_TO_LAUNCHER_LIST_AICORE.
 * Semantics mirror truth source math/sort_with_index, EXCEPT sorted_index dtype follows the
 * `index` input dtype (int32/int64) instead of the hard-coded INT32 of the truth source
 * (DESIGN §5.4/§6 L0 correction item; spec rule outputs[sorted_index].dtype = index.dtype).
 */
#ifndef OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_OP_API_SORT_WITH_INDEX_L0_H_
#define OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_OP_API_SORT_WITH_INDEX_L0_H_

#include "opdev/op_executor.h"

namespace l0op {

// Returns (y, sorted_index). y dtype == self dtype; sorted_index dtype == index dtype.
const std::tuple<aclTensor*, aclTensor*> SortWithIndex(
    const aclTensor* self, const aclTensor* index, const int64_t axis, const bool descending, const bool stable, aclOpExecutor* executor);

} // namespace l0op

#endif // OPS_MATH_EXPERIMENTAL_SORT_WITH_INDEX_OP_API_SORT_WITH_INDEX_L0_H_
