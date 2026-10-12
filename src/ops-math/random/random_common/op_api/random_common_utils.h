/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_RANDOM_RANDOM_COMMON_UTILS_H_
#define OP_API_INC_RANDOM_RANDOM_COMMON_UTILS_H_

// random 类算子 aclnn 实现的公共头文件引入（aclnn_random/aclnn_normal/aclnn_uniform 共用）
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "math/add/op_api/add.h"
#include "op_api/aclnn_check.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "../../../conversion/concat_d/op_api/concat_d.h"
#include "aclnn_set_pytorch_random.h"

// DSA 路径公共处理：将 offsetTensor 与 offset 相加后，与 0 做 concat，构造 [0, offsetTensor+offset] 张量
static inline const aclTensor* ProcessOffsetTensor(const aclTensor* offsetTensor, int64_t offset,
                                                   aclOpExecutor* executor)
{
    op::FVector<int64_t> tmpVector = {static_cast<int64_t>(offset)};
    auto offsetTmpTensor = executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), offsetTensor->GetDataType());
    CHECK_RET(offsetTmpTensor != nullptr, nullptr);
    auto offsetAddOut = l0op::Add(offsetTensor, offsetTmpTensor, executor);
    CHECK_RET(offsetAddOut != nullptr, nullptr);
    // concat
    op::FVector<int64_t> zeroVector = {static_cast<int64_t>(0)};
    auto zeroTensor = executor->ConvertToTensor(zeroVector.data(), zeroVector.size(), offsetTensor->GetDataType());
    CHECK_RET(zeroTensor != nullptr, nullptr);
    op::FVector<const aclTensor*> tensorListOnce;
    tensorListOnce.emplace_back(zeroTensor);
    tensorListOnce.emplace_back(offsetAddOut);
    auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
    auto concatTensor = l0op::ConcatD(tensorList, 0, executor);
    CHECK_RET(concatTensor != nullptr, nullptr);

    return concatTensor;
}

#endif // OP_API_INC_RANDOM_RANDOM_COMMON_UTILS_H_
