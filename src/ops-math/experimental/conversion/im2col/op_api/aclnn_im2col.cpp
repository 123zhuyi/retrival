/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conversion/im2col/op_api/aclnn_im2col.h"
#include "conversion/im2col/op_api/im2col.h"
#include <limits>
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

static constexpr size_t NEED_SQUEEZE = 3;
static constexpr size_t NO_NEED_SQUEEZE = 4;
static constexpr size_t ARRAY_SIZE = 2;
static constexpr size_t PADDING_SIZE = 4;
static constexpr size_t DIM_N = 0;
static constexpr size_t DIM_C = 1;
static constexpr size_t DIM_H = 2;
static constexpr size_t DIM_W = 3;
static constexpr size_t CHW_DIM_C = 0;
static constexpr size_t CHW_DIM_H = 1;
static constexpr size_t CHW_DIM_W = 2;
static constexpr size_t PAIR_HEIGHT = 0;
static constexpr size_t PAIR_WIDTH = 1;
static constexpr size_t OUTPUT_CHANNEL_DIM = 1;
static constexpr size_t OUTPUT_SPATIAL_DIM = 2;
static constexpr int64_t SYMMETRIC_PADDING_SIDE_COUNT = 2;
static constexpr int64_t UNSQUEEZED_BATCH_SIZE = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_BF16, op::DataType::DT_BOOL};

static bool SafeMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CalculateOutputDim(int64_t input, int64_t kernel, int64_t dilation, int64_t padding, int64_t stride,
                               int64_t& output)
{
    if (stride <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride must be positive.");
        return false;
    }
    const __int128 effectiveKernel = static_cast<__int128>(dilation) * (kernel - 1) + 1;
    const __int128 numerator = static_cast<__int128>(input) +
                               SYMMETRIC_PADDING_SIDE_COUNT * static_cast<__int128>(padding) - effectiveKernel;
    if (numerator < 0) {
        return false;
    }
    const __int128 result = numerator / stride + 1;
    if (result <= 0 || result > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    output = static_cast<int64_t>(result);
    return true;
}

#ifdef __cplusplus
extern "C" {
#endif

static inline bool CheckNotNull(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                                const aclIntArray* padding, const aclIntArray* stride, const aclTensor* out,
                                const uint64_t* workspaceSize, aclOpExecutor* const* executor)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool CheckInputDims(const aclTensor* self)
{
    auto selfDimNum = self->GetViewShape().GetDimNum();
    if (selfDimNum != NEED_SQUEEZE && selfDimNum != NO_NEED_SQUEEZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected self dim [%zu] to be 3 or 4 but check failed.",
                self->GetViewShape().GetDimNum());
        return false;
    }

    const op::Shape selfShape = self->GetViewShape();

    if (selfDimNum == NO_NEED_SQUEEZE && selfShape.GetDim(DIM_N) < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self batch dimension must be non-negative.");
        return false;
    }
    size_t index = selfDimNum == NO_NEED_SQUEEZE ? DIM_C : DIM_N;
    for (size_t i = index; i < selfDimNum; i++) {
        if (selfShape.GetDim(i) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self'dims is invalid, self No.[%lu] dim is [%ld].", i + 1,
                    selfShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckOutputDims(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                            const aclIntArray* padding, const aclIntArray* stride, const aclTensor* out)
{
    bool isNeedSqueeze = (self->GetViewShape().GetDimNum() == NEED_SQUEEZE);
    int64_t inputHeight = isNeedSqueeze ? self->GetViewShape().GetDim(CHW_DIM_H) : self->GetViewShape().GetDim(DIM_H);
    int64_t inputWidth = isNeedSqueeze ? self->GetViewShape().GetDim(CHW_DIM_W) : self->GetViewShape().GetDim(DIM_W);
    int64_t outputHeight = 0;
    int64_t outputWidth = 0;
    if (!CalculateOutputDim(inputHeight, (*kernelSize)[PAIR_HEIGHT], (*dilation)[PAIR_HEIGHT], (*padding)[PAIR_HEIGHT],
                            (*stride)[PAIR_HEIGHT], outputHeight)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The calculated output height is invalid.");
        return false;
    }
    if (!CalculateOutputDim(inputWidth, (*kernelSize)[PAIR_WIDTH], (*dilation)[PAIR_WIDTH], (*padding)[PAIR_WIDTH],
                            (*stride)[PAIR_WIDTH], outputWidth)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The calculated output width is invalid.");
        return false;
    }
    const int64_t inputChannels = isNeedSqueeze ? self->GetViewShape().GetDim(CHW_DIM_C) :
                                                  self->GetViewShape().GetDim(DIM_C);
    int64_t outputChannels = 0;
    if (!SafeMul(inputChannels, (*kernelSize)[PAIR_HEIGHT], outputChannels) ||
        !SafeMul(outputChannels, (*kernelSize)[PAIR_WIDTH], outputChannels)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The calculated output channel size overflows.");
        return false;
    }
    int64_t outputSpatial = 0;
    if (!SafeMul(outputHeight, outputWidth, outputSpatial)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The calculated output spatial size overflows.");
        return false;
    }
    const op::Shape outShape = isNeedSqueeze ?
                                   op::Shape({outputChannels, outputSpatial}) :
                                   op::Shape({self->GetViewShape().GetDim(DIM_N), outputChannels, outputSpatial});
    if (outShape != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expect out shape [%s], but got: [%s].", op::ToString(outShape).GetString(),
                op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}
static bool CheckArray(const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
                       const aclIntArray* stride)
{
    if (kernelSize->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected kernelSize equals to 2, but got size %lu.",
                kernelSize->Size());
        return false;
    }
    if (dilation->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected dilation equals to 2, but got size %lu.", dilation->Size());
        return false;
    }
    if (padding->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected padding equals to 2, but got size %lu.", padding->Size());
        return false;
    }
    if (stride->Size() != ARRAY_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is expected stride equals to 2, but got size %lu.", stride->Size());
        return false;
    }
    for (size_t i = 0; i < kernelSize->Size(); i++) {
        if ((*kernelSize)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "It is expected kernelSize be greater than zero, "
                    "but kernelSize No.[%lu] dim is [%ld].",
                    i + 1, (*kernelSize)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < stride->Size(); i++) {
        if ((*stride)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "It is expected stride be greater than zero, "
                    "but stride No.[%lu] dim is [%ld].",
                    i + 1, (*stride)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < dilation->Size(); i++) {
        if ((*dilation)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "It is expected dilation be greater than zero, "
                    "but dilation No.[%lu] dim is [%ld].",
                    i + 1, (*dilation)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < padding->Size(); i++) {
        if ((*padding)[i] < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "It is expected padding be greater than or equal to zero, "
                    "but padding No.[%lu] dim is [%ld].",
                    i + 1, (*padding)[i]);
            return false;
        }
    }
    return true;
}

static void CheckFormat(const aclTensor* self)
{
    // 检查format，若是NZ格式，则添加警告
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of self gets [%s], this format may lead to precision failure.",
                op::ToString(self->GetStorageFormat()).GetString());
    }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation,
                               const aclIntArray* padding, const aclIntArray* stride, const aclTensor* out,
                               const uint64_t* workspaceSize, aclOpExecutor* const* executor)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, kernelSize, dilation, padding, stride, out, workspaceSize, executor),
              ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    if (out->GetDataType() != self->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtype must match self dtype.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 检查输入Tensor self
    CHECK_RET(CheckInputDims(self), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数组是否满足要求
    CHECK_RET(CheckArray(kernelSize, dilation, padding, stride), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入输出Tensor out
    CHECK_RET(CheckOutputDims(self, kernelSize, dilation, padding, stride, out), ACLNN_ERR_PARAM_INVALID);

    // 检查format，若是NZ格式，则添加警告
    CheckFormat(self);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
                                        const aclIntArray* dilation, const aclIntArray* padding,
                                        const aclIntArray* stride, const aclTensor* out, uint64_t* workspaceSize,
                                        aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIm2col, DFX_IN(self, kernelSize, dilation, padding, stride), DFX_OUT(out));
    auto ret = CheckParams(self, kernelSize, dilation, padding, stride, out, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = 0;
    *executor = nullptr;

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    bool isNeedSqueeze = (self->GetViewShape().GetDimNum() == NEED_SQUEEZE);

    // 固定写法，将输入转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* selfUnsqueeze = selfContiguous;
    if (isNeedSqueeze) {
        const op::Shape self4dShape({UNSQUEEZED_BATCH_SIZE, selfContiguous->GetViewShape().GetDim(CHW_DIM_C),
                                     selfContiguous->GetViewShape().GetDim(CHW_DIM_H),
                                     selfContiguous->GetViewShape().GetDim(CHW_DIM_W)});
        selfUnsqueeze = uniqueExecutor.get()->CreateView(selfContiguous, self4dShape, selfContiguous->GetViewOffset());
        CHECK_RET(selfUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto selfReFormat = l0op::ReFormat(selfUnsqueeze, op::Format::FORMAT_NCHW);
    CHECK_RET(selfReFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t> padding4d = {(*padding)[PAIR_HEIGHT], (*padding)[PAIR_HEIGHT], (*padding)[PAIR_WIDTH],
                                  (*padding)[PAIR_WIDTH]};
    const aclIntArray* newPadding = uniqueExecutor.get()->AllocIntArray(padding4d.data(), PADDING_SIZE);
    CHECK_RET(newPadding != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto im2colOut = l0op::Im2col(selfReFormat, kernelSize, dilation, newPadding, stride, uniqueExecutor.get());
    CHECK_RET(im2colOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* outSqueeze = im2colOut;
    if (isNeedSqueeze) {
        const op::Shape out2dShape({im2colOut->GetViewShape().GetDim(OUTPUT_CHANNEL_DIM),
                                    im2colOut->GetViewShape().GetDim(OUTPUT_SPATIAL_DIM)});
        outSqueeze = uniqueExecutor.get()->CreateView(im2colOut, out2dShape, im2colOut->GetViewOffset());
        CHECK_RET(outSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto outView = uniqueExecutor.get()->CreateView(outSqueeze, outSqueeze->GetViewShape(),
                                                    outSqueeze->GetViewOffset());
    CHECK_RET(outView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outReFormat = l0op::ReFormat(outView, out->GetViewFormat());
    CHECK_RET(outReFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto outCast = l0op::Cast(outReFormat, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(outCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(outCast, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIm2col(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIm2col);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
