/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bincount.h"
#include "bincount.h"
#include "aclnn_kernels/cast.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int32_t MAXIMUM_SIZE = 2147483647;

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> WEIGHTS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,   op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_INT32, op::DataType::DT_INT64};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* weights, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    if (weights) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weights, WEIGHTS_DTYPE_SUPPORT_LIST, return false);
    }

    return true;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* weights, const aclTensor* out)
{
    if (op::IsPrivateFormat(self->GetViewFormat()) || op::IsPrivateFormat(out->GetViewFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format only support ND,NCL,NCHW,NHWC,HWCN,NDHWC,NCDHW,"
            "input format is [%s] and out format is [%s].",
            ToString(self->GetViewFormat()).GetString(), ToString(out->GetViewFormat()).GetString());
        return false;
    }

    // self只支持一维
    OP_CHECK_WRONG_DIMENSION(self, 1, return false);

    if (!weights) {
        return true;
    }

    if (self->GetViewFormat() != weights->GetViewFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input and weights should have same format,"
            "but input format is [%s] and weights format is [%s].",
            ToString(self->GetViewFormat()).GetString(), ToString(weights->GetViewFormat()).GetString());
        return false;
    }

    // self和weights shape必须相同
    OP_CHECK_SHAPE_NOT_EQUAL(self, weights, return false);

    if (self->GetStorageFormat() != Format::FORMAT_ND || weights->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGW("Only support ND format for bincount operator.");
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* weights, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, weights, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, weights, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static const aclTensor* dealWeightsTensor(const aclTensor* self, const aclTensor* weights, aclOpExecutor* executor)
{
    const aclTensor* weightsTensor;
    // 如果weights为空指针，则构造一个全为1的tensor
    if (!weights) {
        aclScalar* value = executor->AllocScalar(1);
        const aclTensor* valueTensor = executor->ConvertToTensor(value, op::DataType::DT_INT64);
        // fill dims tensor
        FVector<int64_t> dimTmp{self->GetViewShape().GetDim(0)};

        aclIntArray* shapeArray = executor->AllocIntArray(dimTmp.data(), dimTmp.size());
        const aclTensor* dims = executor->ConvertToTensor(dimTmp.data(), dimTmp.size(), op::DataType::DT_INT64);
        weightsTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
        CHECK_RET(weightsTensor != nullptr, nullptr);
    } else {
        auto weightsContiguous = l0op::Contiguous(weights, executor);
        CHECK_RET(weightsContiguous != nullptr, nullptr);
        if (weights->GetDataType() != op::DataType::DT_FLOAT && weights->GetDataType() != op::DataType::DT_DOUBLE) {
            weightsTensor = l0op::Cast(weightsContiguous, op::DataType::DT_DOUBLE, executor);
            CHECK_RET(weightsTensor != nullptr, nullptr);
        } else {
            weightsTensor = weightsContiguous;
        }
    }
    return weightsTensor;
}

static const aclTensor* dealWeightsTensor_950(
    const aclTensor* self, const aclTensor* weights, aclOpExecutor* executor)
{
    OP_LOGD("dealWeightsTensor begin");
    const aclTensor* weightsTensor;
    // 如果weights为空指针，则构造一个全为1的tensor
    if (!weights) {
        if (IsRegBase()) {
            op::Shape weightShape = {0};
            weightsTensor = executor->AllocTensor(weightShape, op::DataType::DT_FLOAT);
            CHECK_RET(weightsTensor != nullptr, nullptr);
        } else {
            aclScalar* value = executor->AllocScalar(1);
            const aclTensor* valueTensor = executor->ConvertToTensor(value, op::DataType::DT_INT64);
            // fill dims tensor
            FVector<int64_t> dimTmp{self->GetViewShape().GetDim(0)};

            aclIntArray* shapeArray = executor->AllocIntArray(dimTmp.data(), dimTmp.size());
            const aclTensor* dims = executor->ConvertToTensor(dimTmp.data(), dimTmp.size(), op::DataType::DT_INT64);
            weightsTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
            CHECK_RET(weightsTensor != nullptr, nullptr);
        }
    } else {
        auto weightsContiguous = l0op::Contiguous(weights, executor);
        CHECK_RET(weightsContiguous != nullptr, nullptr);
        if (weights->GetDataType() != op::DataType::DT_FLOAT && weights->GetDataType() != op::DataType::DT_DOUBLE) {
            weightsTensor = l0op::Cast(weightsContiguous, op::DataType::DT_DOUBLE, executor);
            CHECK_RET(weightsTensor != nullptr, nullptr);
        } else {
            weightsTensor = weightsContiguous;
        }
    }
    return weightsTensor;
}

static aclnnStatus DealEmptyTensorWithMinlength(aclTensor* out, aclOpExecutor* executor)
{
    auto outShape = out->GetViewShape();
    op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(outShape);
    auto shapes = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
    const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT64);
    const aclScalar* valueScalar = executor->AllocScalar(0);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
    CHECK_RET(fillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dstCopyResult = l0op::ViewCopy(fillTensor, out, executor);
    CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBincountGetWorkspaceSize(
    const aclTensor* self, const aclTensor* weights, int64_t minlength, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBincount, DFX_IN(self, weights, minlength), DFX_OUT(out));

    auto ret = CheckParams(self, weights, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        if (minlength > 0) {
            auto res = DealEmptyTensorWithMinlength(out, uniqueExecutor.get());
            CHECK_RET(res == ACLNN_SUCCESS, res);
            *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        } else {
            *workspaceSize = 0;
        }
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCast = l0op::Cast(selfContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const int64_t sizes = out->GetViewShape().GetDim(0);
    int64_t size = (sizes > minlength) ? sizes : minlength;

    if (size > MAXIMUM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum output size cannot exceed 2147483647.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto weightsTensor = (IsRegBase()) ?
                             dealWeightsTensor_950(self, weights, uniqueExecutor.get()) :
                             dealWeightsTensor(self, weights, uniqueExecutor.get());
    CHECK_RET(weightsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto BincountOut = l0op::Bincount(selfCast, weightsTensor, size, uniqueExecutor.get());
    CHECK_RET(BincountOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castResult = l0op::Cast(BincountOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBincount);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
