/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_im2col.h"
#include "im2col.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
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

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_INT16,    op::DataType::DT_UINT16,
    op::DataType::DT_INT32, op::DataType::DT_UINT32,    op::DataType::DT_INT64,    op::DataType::DT_UINT64,
    op::DataType::DT_BF16,  op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,    op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_COMPLEX32, op::DataType::DT_COMPLEX64};

// Integer division rounding to -Infinity
template <typename T>
static inline auto div_rtn(T x, T y) -> T
{
    if (y == 0) {
        OP_LOGE(ACL_ERROR_INVALID_PARAM, "Division by zero!");
        return -1;
    }
    int q = x / y;
    int r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    };
    return q;
}

#ifdef __cplusplus
extern "C" {
#endif

static inline bool CheckNotNull(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(dilation, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(stride, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckInputDims(const aclTensor* self)
{
    auto selfDimNum = self->GetViewShape().GetDimNum();
    if (selfDimNum != NEED_SQUEEZE && selfDimNum != NO_NEED_SQUEEZE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expected self dim [%zu] to be 3 or 4 but check failed.",
            self->GetViewShape().GetDimNum());
        return false;
    }

    const op::Shape selfShape = self->GetViewShape();

    size_t index = selfDimNum == NO_NEED_SQUEEZE ? 1 : 0;
    for (size_t i = index; i < selfDimNum; i++) {
        if (selfShape.GetDim(i) <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "self'dims is invalid, self No.[%lu] dim is [%ld].", i + 1,
                selfShape.GetDim(i));
            return false;
        }
    }
    return true;
}

static bool CheckOutputDims(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, const aclTensor* out)
{
    bool isNeedSqueeze = (self->GetViewShape().GetDimNum() == NEED_SQUEEZE);
    int64_t inputHeight = isNeedSqueeze ? self->GetViewShape().GetDim(1) : self->GetViewShape().GetDim(2);
    int64_t inputWidth = isNeedSqueeze ? self->GetViewShape().GetDim(2) : self->GetViewShape().GetDim(3);
    int64_t outputHeight =
        div_rtn<int64_t>(
            (inputHeight + 2 * (*padding)[0] - ((*dilation)[0] * ((*kernelSize)[0] - 1) + 1)), (*stride)[0]) +
        1;
    int64_t outputWidth =
        div_rtn<int64_t>(
            (inputWidth + 2 * (*padding)[1] - ((*dilation)[1] * ((*kernelSize)[1] - 1) + 1)), (*stride)[1]) +
        1;
    if (outputHeight < 1 || outputWidth < 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The shape (%ld, %ld) of the array calculated by other parameters "
            "must be at least one.",
            outputHeight, outputWidth);
        return false;
    }
    const op::Shape outShape =
        isNeedSqueeze ?
            op::Shape(
                {self->GetViewShape().GetDim(0) * (*kernelSize)[0] * (*kernelSize)[1], outputHeight * outputWidth}) :
            op::Shape(
                {self->GetViewShape().GetDim(0), self->GetViewShape().GetDim(1) * (*kernelSize)[0] * (*kernelSize)[1],
                 outputHeight * outputWidth});
    if (outShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Expect out shape [%s], but got: [%s].", op::ToString(outShape).GetString(),
            op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}
static bool CheckArray(
    const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding, const aclIntArray* stride)
{
    if (kernelSize->Size() != ARRAY_SIZE) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "It is expected kernelSize equals to 2, but got size %lu.", kernelSize->Size());
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
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "It is expected kernelSize be greater than zero, "
                "but kernelSize No.[%lu] dim is [%ld].",
                i + 1, (*kernelSize)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < stride->Size(); i++) {
        if ((*stride)[i] <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "It is expected stride be greater than zero, "
                "but stride No.[%lu] dim is [%ld].",
                i + 1, (*stride)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < dilation->Size(); i++) {
        if ((*dilation)[i] <= 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "It is expected dilation be greater than zero, "
                "but dilation No.[%lu] dim is [%ld].",
                i + 1, (*dilation)[i]);
            return false;
        }
    }
    for (size_t i = 0; i < padding->Size(); i++) {
        if ((*padding)[i] < 0) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
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

static aclnnStatus CheckParams(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, kernelSize, dilation, padding, stride, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_REGBASE, return ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
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

aclnnStatus aclnnIm2colGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* dilation, const aclIntArray* padding,
    const aclIntArray* stride, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIm2col, DFX_IN(self, kernelSize, dilation, padding, stride), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(self, kernelSize, dilation, padding, stride, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    bool isNeedSqueeze = (self->GetViewShape().GetDimNum() == NEED_SQUEEZE);

    // 固定写法，将输入转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfUnsqueeze = isNeedSqueeze ?
                             l0op::UnsqueezeNd(selfContiguous, static_cast<int64_t>(0), uniqueExecutor.get()) :
                             selfContiguous;
    CHECK_RET(selfUnsqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfReFormat = l0op::ReFormat(selfUnsqueeze, op::Format::FORMAT_NCHW);
    CHECK_RET(selfReFormat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<int64_t> padding4d = {(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1]};
    const aclIntArray* newPadding = uniqueExecutor.get()->AllocIntArray(padding4d.data(), PADDING_SIZE);

    auto im2colOut = l0op::Im2col(selfReFormat, kernelSize, dilation, newPadding, stride, uniqueExecutor.get());
    CHECK_RET(im2colOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outSqueeze =
        isNeedSqueeze ? l0op::SqueezeNd(im2colOut, static_cast<int64_t>(0), uniqueExecutor.get()) : im2colOut;
    CHECK_RET(outSqueeze != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outView =
        uniqueExecutor.get()->CreateView(outSqueeze, outSqueeze->GetViewShape(), outSqueeze->GetViewOffset());
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
