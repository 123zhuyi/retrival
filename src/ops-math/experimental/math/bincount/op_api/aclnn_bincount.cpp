/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "aclnn_bincount.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "bincount.h"
#include "conversion/fill/op_api/fill.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 输出桶数上限。放宽到 2^32 以支持 self 取到 INT32_MAX 时的输出（L=2^31）；
// kernel 内部用 int64 下标寻址,真正的上限是输出显存。更大的 L（如 INT64_MAX）
// 其输出张量在框架侧就无法分配,到不了这里。
static constexpr int64_t MAXIMUM_SIZE = (1LL << 32);

// 用户接口（torch 风格）支持的 dtype；底层 def 算子原型为
// array/size/weights/bins。
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> WEIGHTS_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,   op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out, const uint64_t* workspaceSize,
                         aclOpExecutor** executor)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* weights, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    if (weights != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weights, WEIGHTS_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* weights, const aclTensor* out)
{
    if (op::IsPrivateFormat(self->GetViewFormat()) || op::IsPrivateFormat(out->GetViewFormat()) ||
        (weights != nullptr && op::IsPrivateFormat(weights->GetViewFormat()))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format only support ND,NCL,NCHW,NHWC,HWCN,NDHWC,NCDHW,"
                "self format is [%s], weights format is [%s], out format is [%s].",
                ToString(self->GetViewFormat()).GetString(),
                weights == nullptr ? "NULL" : ToString(weights->GetViewFormat()).GetString(),
                ToString(out->GetViewFormat()).GetString());
        return false;
    }

    OP_CHECK_WRONG_DIMENSION(self, 1, return false);
    OP_CHECK_WRONG_DIMENSION(out, 1, return false);
    if (weights != nullptr) {
        OP_CHECK_WRONG_DIMENSION(weights, 1, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(self, weights, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* weights, int64_t minlength, const aclTensor* out,
                               const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(CheckNotNull(self, out, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, weights, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, weights, out), ACLNN_ERR_PARAM_INVALID);

    if (minlength < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "minlength should be >= 0, but got %ld.", minlength);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
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

// 空输入(self 长度 0)：torch 语义返回全 0；out 有实际长度则清零，否则
// workspace=0。
static aclnnStatus HandleEmptySelf(aclTensor* out, aclOpExecutor* executor, uint64_t* workspaceSize)
{
    if (out->GetViewShape().GetDim(0) > 0) {
        auto res = DealEmptyTensorWithMinlength(out, executor);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        *workspaceSize = executor->GetWorkspaceSize();
    } else {
        *workspaceSize = 0;
    }
    return ACLNN_SUCCESS;
}

// weights 转连续 + 统一 Cast 成 float。无 weights 返回 nullptr（计数模式）；
// weights 非空但处理失败也返回 nullptr，由调用方据 (weights != nullptr)
// 区分校验。
static const aclTensor* PrepareWeights(const aclTensor* weights, aclOpExecutor* executor)
{
    if (weights == nullptr) {
        return nullptr;
    }
    auto weightsContiguous = l0op::Contiguous(weights, executor);
    if (weightsContiguous == nullptr) {
        return nullptr;
    }
    if (weightsContiguous->GetDataType() != op::DataType::DT_FLOAT) {
        return l0op::Cast(weightsContiguous, op::DataType::DT_FLOAT, executor);
    }
    return weightsContiguous;
}

// 选择底层 kernel 直接产出的计算 dtype：out=double 走双精度位拼接；其余加权用
// float、计数用 out dtype。
static op::DataType SelectComputeDtype(op::DataType outDtype, bool weighted)
{
    if (outDtype == op::DataType::DT_DOUBLE) {
        return op::DataType::DT_DOUBLE;
    }
    return weighted ? op::DataType::DT_FLOAT : outDtype;
}

aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor* self, const aclTensor* weights, int64_t minlength,
                                          aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnBincount, DFX_IN(self, weights, minlength), DFX_OUT(out));

    auto ret = CheckParams(self, weights, minlength, out, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = 0;
    *executor = nullptr;

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        auto res = HandleEmptySelf(out, uniqueExecutor.get(), workspaceSize);
        CHECK_RET(res == ACLNN_SUCCESS, res);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (out->GetViewShape().GetDim(0) > MAXIMUM_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The maximum output size cannot exceed %lld.",
                static_cast<long long>(MAXIMUM_SIZE));
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // weights 任意 dtype 统一 Cast 成 float 再喂底层算子（无 weights 为
    // nullptr，计数模式）
    const aclTensor* weightsForKernel = PrepareWeights(weights, uniqueExecutor.get());
    CHECK_RET(weights == nullptr || weightsForKernel != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 底层 def 算子原型为 array/size/weights/bins：将 host 侧 minlength 构造成
    // size 输入（int64, 1 维 [1]） 下发给底层算子。kernel 不依赖 size
    // 的值（输出长度由 out shape 决定），size 仅为对齐底层原型。
    int64_t sizeData = minlength;
    auto sizeIntArray = uniqueExecutor->AllocIntArray(&sizeData, 1);
    CHECK_RET(sizeIntArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* sizeTensor = uniqueExecutor->ConvertToTensor(sizeIntArray, op::DataType::DT_INT64);
    CHECK_RET(sizeTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // computeDtype = 让底层 kernel 直接产出的 dtype（详见
    // SelectComputeDtype）：out=double 走双精度 位拼接（910B AI Core 无
    // fp64）；其余加权用 float 计算后再 Cast 收敛，计数直接用 out dtype。
    op::DataType outDtype = out->GetDataType();
    op::DataType computeDtype = SelectComputeDtype(outDtype, weights != nullptr);

    auto bincountOut = l0op::Bincount(selfContiguous, sizeTensor, weightsForKernel, out->GetViewShape(), computeDtype,
                                      uniqueExecutor.get());
    CHECK_RET(bincountOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* finalOut = bincountOut;
    if (computeDtype != outDtype) {
        finalOut = l0op::Cast(bincountOut, outDtype, uniqueExecutor.get());
        CHECK_RET(finalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewCopyResult = l0op::ViewCopy(finalOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBincount(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBincount);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
