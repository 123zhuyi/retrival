/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_slice_v2.h"
#include <bitset>
#include "strided_slice_v3.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_INT32, op::DataType::DT_UINT8,   op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static inline int64_t GetPosDim(int64_t dim, int64_t dimNum)
{
    return dim >= 0 ? dim : dim + dimNum;
}

inline static bool CheckNotNull(
    const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes,
    const aclIntArray* steps, aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(starts, return false);
    OP_CHECK_NULL(ends, return false);
    OP_CHECK_NULL(axes, return false);
    OP_CHECK_NULL(steps, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // self和out数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of aclnnSliceV2 is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查self的数据类型是否在API支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckAxesValid(const aclTensor* self, const aclIntArray* axes)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());

    // dim为负时需要转正校验
    bitset<MAX_DIM_LEN> dimMask = bitset<MAX_DIM_LEN>();

    for (uint64_t i = 0; i < axes->Size(); i++) {
        if (axes->operator[](i) >= selfDimNum || axes->operator[](i) < (-selfDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Provided aclnnSliceV2 axes %ld not in the range of input tensor size %ld.",
                axes->operator[](i), selfDimNum);
            return false;
        }
        int64_t index = GetPosDim(axes->operator[](i), selfDimNum);
        // dim重复
        if (dimMask[index]) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "aclnnSliceV2 axes %ld appears multiple times in the list of dims.", index);
            return false;
        }

        dimMask.set(index);
    }

    return true;
}

static bool CheckArray(
    const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes, const aclIntArray* steps)
{
    if (starts->Size() != axes->Size()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_NULLPTR, "Expected aclnnSliceV2 starts.size() %lu to be equal to axes.size() %lu.",
            starts->Size(), axes->Size());
        return false;
    }
    if (ends->Size() != axes->Size()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_NULLPTR, "Expected aclnnSliceV2 ends.size() %lu to be equal to axes.size() %lu.",
            ends->Size(), axes->Size());
        return false;
    }
    for (uint64_t i = 0; i < steps->Size(); i++) {
        if (steps->operator[](i) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnSliceV2 step must be positive.");
            return false;
        }
    }
    return true;
}

static bool CheckMaxDim(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    return true;
}

static void CalculateValuesSliceLowerForSliceV2(const aclTensor* self, int64_t& dim, int64_t& start, int64_t& end)
{
    if (start < 0) {
        start = 0;
    } else if (start >= self->GetViewShape().GetDim(dim)) {
        start = self->GetViewShape().GetDim(dim);
    }
    if (end < start) {
        end = start;
    } else if (end >= self->GetViewShape().GetDim(dim)) {
        end = self->GetViewShape().GetDim(dim);
    }
}

static bool CheckShape(
    const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes,
    const aclIntArray* steps, const aclTensor* out)
{
    auto sliceShape = self->GetViewShape();
    int64_t start, end, dim, step, sliceNum;
    int64_t selfDimNum = sliceShape.GetDimNum();
    for (uint64_t i = 0; i < steps->Size(); i++) {
        dim = GetPosDim(axes->operator[](i), selfDimNum);
        start = GetPosDim(starts->operator[](i), self->GetViewShape().GetDim(dim));
        end = GetPosDim(ends->operator[](i), self->GetViewShape().GetDim(dim));
        step = steps->operator[](i);
        CalculateValuesSliceLowerForSliceV2(self, dim, start, end);
        sliceNum = (end - start + step - 1) / step;
        sliceShape.SetDim(dim, sliceNum);
    }
    auto outShape = out->GetViewShape();
    // 校验输出shape
    if (sliceShape != outShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of aclnnSliceV2 out should be %s, but current is %s.",
            op::ToString(sliceShape).GetString(), op::ToString(outShape).GetString());
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes,
    const aclIntArray* steps, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, starts, ends, axes, steps, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入与输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的shape支持能力
    CHECK_RET(CheckMaxDim(self), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的axes是否超出self维度范围或者重复
    CHECK_RET(CheckAxesValid(self, axes), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入的IntArray支持能力
    CHECK_RET(CheckArray(starts, ends, axes, steps), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输出的shape支持能力
    CHECK_RET(CheckShape(self, starts, ends, axes, steps, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSliceV2GetWorkspaceSize(
    const aclTensor* self, const aclIntArray* starts, const aclIntArray* ends, const aclIntArray* axes,
    const aclIntArray* steps, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSliceV2, DFX_IN(self, starts, ends, axes, steps), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, starts, ends, axes, steps, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto sliceOut = l0op::StridedSliceV3V2(selfContiguous, starts, ends, axes, steps, uniqueExecutor.get());
    CHECK_RET(sliceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(sliceOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSliceV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSliceV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
