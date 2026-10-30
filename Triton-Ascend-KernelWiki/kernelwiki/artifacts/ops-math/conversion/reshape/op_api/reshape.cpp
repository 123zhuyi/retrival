/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/transdata.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Reshape);

bool Validate(const aclTensor* x, const op::Shape& shape)
{
    if (!op::IsContiguous(x)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor should be contiguous. %s", x->ToString().GetString());
        return false;
    }
    if (shape.GetShapeSize() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Can not reshape to empty tensor.");
        return false;
    }
    return true;
}

const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    if (!Validate(x, shape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid reshape params.");
        return nullptr;
    }
    if (x->GetViewShape() == shape && x->GetStorageShape() == shape && x->GetOriginalShape() == shape) {
        return x;
    }
    int64_t negativeAxis = -1;
    int64_t remainShapeSize = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.GetDimNum()); i++) {
        if (shape[i] == -1) {
            if (negativeAxis != -1) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "Only one dim may be -1, not both dim[%ld] and dim[%ld].", negativeAxis,
                    shape[i]);
                return nullptr;
            }
            negativeAxis = i;
        } else {
            remainShapeSize *= shape[i];
        }
    }
    // support []->[1] []->[-1]
    int64_t shapeSize = x->GetViewShape().GetShapeSize();
    aclTensor* newTensor = nullptr;
    if (negativeAxis >= 0) {
        if (shapeSize % remainShapeSize != 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Input shape [%s] cannot be reshape to [%s].",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(shape).GetString());
            return nullptr;
        }
        op::Shape newShape(shape);
        newShape[negativeAxis] = shapeSize / remainShapeSize;
        newTensor = executor->CreateView(x, newShape, x->GetViewOffset());
    } else {
        if (shape.GetShapeSize() != shapeSize) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Input shape [%s] cannot be reshape to [%s]. ShapeSize mismatch.",
                op::ToString(x->GetViewShape()).GetString(), op::ToString(shape).GetString());
            return nullptr;
        }
        newTensor = executor->CreateView(x, shape, x->GetViewOffset());
    }
    if (newTensor->GetViewShape().GetDimNum() != x->GetViewShape().GetDimNum()) {
        return l0op::ReFormat(newTensor, static_cast<op::Format>(ACL_FORMAT_ND));
    }
    return newTensor;
}

const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor)
{
    L0_DFX(Reshape, x, shape);
    op::Shape newShape;
    for (int64_t i = 0; i < static_cast<int64_t>(shape->Size()); i++) {
        newShape.AppendDim((*shape)[i]);
    }
    return Reshape(x, newShape, executor);
}

} // namespace l0op