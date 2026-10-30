/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_slice.h"
#include "conversion/strided_slice_v3/op_api/strided_slice_v3.h"
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

constexpr int64_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_INT32, op::DataType::DT_UINT8,   op::DataType::DT_BOOL,  op::DataType::DT_BF16};

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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of aclnnSlice is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查self的数据类型是否在API支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static void CalculateValuesSliceUpperForSlice(const aclTensor* self, int64_t& dim, int64_t& start, int64_t& end)
{
    if (dim < 0) {
        dim = dim + self->GetViewShape().GetDimNum();
    }
    if (start < 0) {
        start = start + self->GetViewShape().GetDim(dim);
    }
    if (end < 0) {
        end = end + self->GetViewShape().GetDim(dim);
    }
}

static void CalculateValuesSliceLowerForSlice(const aclTensor* self, int64_t& dim, int64_t& start, int64_t& end)
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

static bool CheckNotNull2TensorForSlice(const aclTensor* t0, const aclTensor* t1)
{
    OP_CHECK_NULL(t0, return false);
    OP_CHECK_NULL(t1, return false);

    return true;
}

static bool CheckShape(
    const aclTensor* self, int64_t dim, int64_t start, int64_t end, int64_t step, const aclTensor* out)
{
    int64_t dimNum = self->GetViewShape().GetDimNum();
    // 校验输入的长度
    if (dimNum > MAX_DIM_LEN) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected aclnnSlice self len %ld to not be greater than %ld but check failed.",
            dimNum, MAX_DIM_LEN);
        return false;
    }
    if (dimNum <= 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected aclnnSlice self len %ld to not be less than one but check failed.",
            dimNum);
        return false;
    }

    if ((-dimNum > dim) || ((dimNum - 1) < dim)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_NULLPTR, "Expected aclnnSlice dim value %ld to be in range [%ld, %ld] but check failed.",
            dim, -dimNum, dimNum - 1);
        return false;
    }

    if (step <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnSlice step must be positive.");
        return false;
    }
    CalculateValuesSliceUpperForSlice(self, dim, start, end);
    CalculateValuesSliceLowerForSlice(self, dim, start, end);
    int64_t sliceNum = (end - start + step - 1) / step;
    auto sliceShape = self->GetViewShape();
    sliceShape.SetDim(dim, sliceNum);
    auto outShape = out->GetViewShape();
    // 校验输出shape
    if (sliceShape != outShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(sliceShape).GetString(), op::ToString(outShape).GetString());
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2TensorForSlice(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入与输出的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入输出的shape支持能力
    CHECK_RET(CheckShape(self, dim, start, end, step, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSliceGetWorkspaceSize(
    const aclTensor* self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSlice, DFX_IN(self, dim, start, end, step), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, start, end, step, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CalculateValuesSliceUpperForSlice(self, dim, start, end);
    CalculateValuesSliceLowerForSlice(self, dim, start, end);
    auto sliceOut = l0op::StridedSliceV3(selfContiguous, start, end, dim, step, uniqueExecutor.get());
    CHECK_RET(sliceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(sliceOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSlice(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSlice);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
