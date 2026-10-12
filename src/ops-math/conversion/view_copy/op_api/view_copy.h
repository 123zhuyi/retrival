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
 * \file view_copy.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_VIEW_COPY_H
#define OP_API_INC_LEVEL0_VIEW_COPY_H

#include "opdev/op_def.h"
#include "opdev/common_types.h"

namespace l0op {

const aclTensor *ViewCopy(const aclTensor *y,
                          const aclTensor *dstSize, const aclTensor *dstStride, const aclTensor *dstOffset,
                          const aclTensor *x,
                          const aclTensor *srcSize, const aclTensor *srcStride, const aclTensor *srcOffset,
                          aclOpExecutor *executor);
const aclTensor *ViewCopy(const aclTensor *y,
                          const op::Shape &dstSize, const op::Strides &dstStride, int64_t dstOffset,
                          const aclTensor *x,
                          const op::Shape &srcSize, const op::Strides &srcStride, int64_t srcOffset,
                          aclOpExecutor *executor);

} // l0op

#endif  // OP_API_INC_LEVEL0_VIEW_COPY_H
