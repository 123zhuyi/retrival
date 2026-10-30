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
 * \file chunk_cat.cpp
 * \brief
 */
#include "chunk_cat.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
namespace l0op {

OP_TYPE_REGISTER(ChunkCat);

aclTensor* ChunkCat(const aclTensorList* inputs, int64_t dim, int64_t numChunks, op::DataType outDtype, aclOpExecutor* executor)
{
    L0_DFX(ChunkCat, inputs, dim, numChunks);

    auto output = executor->AllocTensor(outDtype, Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret = INFER_SHAPE(ChunkCat, OP_INPUT(inputs), OP_OUTPUT(output), OP_ATTR(dim, numChunks));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "infershape failed.");
        return nullptr;
    }
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(ChunkCat, OP_INPUT(inputs), OP_OUTPUT(output), OP_ATTR(dim, numChunks));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ChunkCat ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return output;
}

} // namespace l0op
