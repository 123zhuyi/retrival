/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_std_v2_update.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_REDUCE_STD_V2_UPDATE_H
#define OP_API_INC_LEVEL0_REDUCE_STD_V2_UPDATE_H
#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *ReduceStdV2Update(const aclTensor *self, const aclTensor *mean, const aclIntArray *dim,
                                   bool unbiased, bool keepdim, aclOpExecutor *executor);
const aclTensor* ReduceStdV2UpdateCorrection(const aclTensor* self, const aclTensor* mean, const aclIntArray* dim,
                                             int64_t correction, bool keepdim, aclOpExecutor* executor);
}

#endif