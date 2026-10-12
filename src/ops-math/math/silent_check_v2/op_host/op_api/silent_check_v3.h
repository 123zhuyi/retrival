/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_SILENT_CHECK_V3_H_
#define OP_API_INC_LEVEL0_SILENT_CHECK_V3_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *SilentCheckV3(const aclTensor *val, const aclTensor *max, const aclTensor *avgRef, const aclTensor *inputGradRef, const aclTensor *stepRef, 
                                const aclTensor *dstSize, const aclTensor *dstStride, const aclTensor *dstOffset, const float cThreshL1, 
                                const float cThreshL2, const float beta1, const int32_t npuAsdDetect, aclOpExecutor *executor);
} // namespace l0op

#endif // OP_API_INC_LEVEL0_SILENT_CHECK_V3_H_

