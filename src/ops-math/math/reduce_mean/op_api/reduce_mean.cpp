/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_mean.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ReduceMean);

static int64_t MakeWrapDim(int64_t dim, int64_t rank)
{
    if (rank <= 0) {
        rank = 1;
    }
    if (dim < 0) {
        dim += rank;
    }
    return dim;
}

constexpr size_t MAX_MASK_LEN = 64;

bool ReduceMeanInferShape(const op::Shape& self_shape, const aclIntArray* dim, bool keep_dims, op::Shape& reduce_shape)
{
    uint64_t mask[MAX_MASK_LEN] = {0};
    if (dim->Size() == 0) {
        // 空dim mask位全部置1
        for (size_t i = 0; i < MAX_MASK_LEN; i++) {
            mask[i] = 1;
        }
    } else {
        for (size_t i = 0; i < dim->Size(); i++) {
            int64_t index = MakeWrapDim(dim->operator[](i), self_shape.GetDimNum());
            // 如果有重复dim值，返错
            if (mask[index] == 1) {
                OP_LOGE(ACL_ERROR_INVALID_PARAM, "dim value[%ld] repeat", dim->operator[](i));
                return false;
            }
            mask[index] = 1;
        }
    }
    for (size_t i = 0; i < self_shape.GetDimNum(); i++) {
        if (keep_dims) {
            if (mask[i] == 0) {
                reduce_shape.AppendDim(self_shape.GetDim(i));
            } else {
                reduce_shape.AppendDim(1);
            }
        } else {
            if (mask[i] == 0) {
                reduce_shape.AppendDim(self_shape.GetDim(i));
            }
        }
    }
    return true;
}

static const std::initializer_list<op::DataType> aicore_dtype_support_list = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> aicpu_dtype_support_list = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,  op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,   op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,  op::DataType::DT_UINT16, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), aicore_dtype_support_list);
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCpuSupport(const aclTensor* self)
{
    return CheckType(self->GetDataType(), aicpu_dtype_support_list);
}

static const aclTensor* GenerateDimTensor(const aclTensor* self, const aclIntArray* dim, aclOpExecutor* executor)
{
    if (dim->Size() == 0) {
        FVector<int64_t> dimVector;
        for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
            dimVector.emplace_back(i);
        }
        return executor->ConvertToTensor(dimVector.data(), dimVector.size(), DataType::DT_INT64);
    } else {
        return executor->ConvertToTensor(dim, DataType::DT_INT64);
    }
}

// AICORE算子kernel
static const aclTensor* ReduceMeanAiCore(
    const aclTensor* self, const aclTensor* dim, bool keepDim, bool noopWithEmptyAxes, const aclTensor* meanOut,
    aclOpExecutor* executor)
{
    L0_DFX(ReduceMeanAiCore, self, dim, keepDim, noopWithEmptyAxes, meanOut);
    // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将Mean算子加入任务队列
    auto retAicore = ADD_TO_LAUNCHER_LIST_AICORE(
        ReduceMean, OP_INPUT(self, dim), OP_OUTPUT(meanOut), OP_ATTR(keepDim, noopWithEmptyAxes));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAicore != ACLNN_SUCCESS, return nullptr, "ReduceMean ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return meanOut;
}

// AICPU算子kernel
static const aclTensor* ReduceMeanAiCpu(
    const aclTensor* self, const aclTensor* dim, bool keepDim, bool noopWithEmptyAxes, const aclTensor* meanOut,
    aclOpExecutor* executor)
{
    // 使用框架宏ADD_AICPU_TO_KERNEL_OBJ_LIST，将Mean算子加入任务队列
    L0_DFX(ReduceMeanAiCpu, self, dim, keepDim, noopWithEmptyAxes, meanOut);

    static internal::AicpuTaskSpace space("Mean", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        ReduceMean, OP_ATTR_NAMES({"keep_dims", "noop_with_empty_axes", "Tidx"}), OP_INPUT(self, dim),
        OP_OUTPUT(meanOut), OP_ATTR(keepDim, noopWithEmptyAxes, dim->GetDataType()));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return meanOut;
}

const aclTensor* ReduceMean(const aclTensor* self, const aclIntArray* dim, bool keepDim, aclOpExecutor* executor)
{
    op::Shape reduceShape;
    if (!ReduceMeanInferShape(self->GetViewShape(), dim, keepDim, reduceShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Reduce %s shape failed. Reduceshape dimnum is %zu ",
            op::ToString(self->GetViewShape()).GetString(), reduceShape.GetDimNum());
        return nullptr;
    }
    // self如果为标量，不调用算子，直接返回
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }
    auto meanOut = executor->AllocTensor(reduceShape, self->GetDataType());
    auto dimTensor = GenerateDimTensor(self, dim, executor);

    if (IsAiCoreSupport(self)) {
        return ReduceMeanAiCore(self, dimTensor, keepDim, true, meanOut, executor);
    } else {
        CHECK_RET(IsAiCpuSupport(self), nullptr);
        return ReduceMeanAiCpu(self, dimTensor, keepDim, true, meanOut, executor);
    }
}

const aclTensor* ReduceMean(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, bool noopWithEmptyAxes, aclOpExecutor* executor)
{
    op::Shape reduceShape;
    if (!ReduceMeanInferShape(self->GetViewShape(), dim, keepDim, reduceShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Reduce %s shape failed. Reduceshape dimnum is %zu ",
            op::ToString(self->GetViewShape()).GetString(), reduceShape.GetDimNum());
        return nullptr;
    }
    // self如果为标量，不调用算子，直接返回
    if (self->GetViewShape().GetDimNum() == 0) {
        return self;
    }
    // dim如果为空且noopWithEmptyAxes为true，不调用算子，直接返回
    if (dim->Size() == 0 && noopWithEmptyAxes) {
        return self;
    }
    auto meanOut = executor->AllocTensor(reduceShape, self->GetDataType());
    auto dimTensor = GenerateDimTensor(self, dim, executor);

    if (IsAiCoreSupport(self)) {
        return ReduceMeanAiCore(self, dimTensor, keepDim, noopWithEmptyAxes, meanOut, executor);
    } else {
        CHECK_RET(IsAiCpuSupport(self), nullptr);
        return ReduceMeanAiCpu(self, dimTensor, keepDim, noopWithEmptyAxes, meanOut, executor);
    }
}
} // namespace l0op
