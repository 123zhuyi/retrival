/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cdist.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
constexpr size_t MIN_DIM_LEN = 2;
static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

OP_TYPE_REGISTER(Cdist);
const Shape InferShapeForA2A3(const aclTensor* x1)
{
    Shape x1Shape = x1->GetViewShape();
    size_t dimNum = x1Shape.GetDimNum();
    Shape outShape;
    for (size_t i = 0; i < dimNum - 1; i++) {
        outShape.AppendDim(x1Shape.GetDim(i));
    }
    return outShape;
}

const Shape InferShapeForA5(const aclTensor* x1, const aclTensor* x2)
{
    op::Shape outShape;
    int64_t x1DimNum = x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = x2->GetViewShape().GetDimNum();
    for(int64_t i = 0; i < x1DimNum - 1; i++) {
        int64_t dim = x1->GetViewShape().GetDim(i);
        outShape.AppendDim(dim);
    }
    outShape.AppendDim(x2->GetViewShape().GetDim(x2DimNum - MIN_DIM_LEN));
    return outShape;
}

// A2A3的AICORE算子kernel
static inline const aclTensor* CdistAiCore(
    const aclTensor* x1, const aclTensor* x2, float p, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(CdistAiCore, x1, x2, p, output);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Cdist算子加入任务队列
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cdist, OP_INPUT(x1, x2), OP_OUTPUT(output), OP_ATTR(p));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CdistAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}
// David的AICORE算子kernel
static inline const aclTensor* CdistAiCore(
    const aclTensor* x1, const aclTensor* x2, float p, int64_t compute_mode, aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(CdistAiCore, x1, x2, p, compute_mode, output);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Cdist, OP_INPUT(x1, x2), OP_ATTR(p, compute_mode), OP_OUTPUT(output));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CdistAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return output;
}


const aclTensor* Cdist(const aclTensor *x1, const aclTensor *x2, float p, int64_t compute_mode, 
                       aclOpExecutor *executor) {
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    Shape outShape;
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        outShape = InferShapeForA2A3(x1);
    } else if (IsRegBase()) {
        outShape = InferShapeForA5(x1, x2);
    }
    auto out = executor->AllocTensor(outShape, x1->GetDataType(), op::Format::FORMAT_ND);
    CHECK_RET(out != nullptr, nullptr);

    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
        if (CheckType(x1->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST) &&
            CheckType(x2->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST)) {
            return CdistAiCore(x1, x2, p, out, executor);
        }
    } else if (IsRegBase()) {
        if (CheckType(x1->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST) &&
            CheckType(x2->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST)) {
            INFER_SHAPE(Cdist, OP_INPUT(x1, x2), OP_OUTPUT(out), OP_ATTR(p, compute_mode));
            return CdistAiCore(x1, x2, p, compute_mode, out, executor);
        }
    } else {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Data type not supported.");
        return nullptr;
    }
    return nullptr;
}
}   // namespace l0op