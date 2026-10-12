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
 * \file aclnn_bernoulli.cpp
 * \brief
 */

#include <cmath>
#include <limits>

#include "aclnn_bernoulli.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "random/stateless_bernoulli/op_api/stateless_bernoulli.h"
#include "experimental/random/bernoulli/op_api/bernoulli.h"
#include "math/zero_op/op_api/zero_op.h"
#include "math/ones_like/op_api/ones_like.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SHAPE_LENGTH = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SUPPORTED_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> SUPPORTED_PROB_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static bool CheckNotNullTensor(const aclTensor* self, const aclTensor* prob, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(prob, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckNotNull(const aclTensor* self, const aclScalar* prob, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(prob, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return SUPPORTED_DTYPE_LIST;
    } else {
        OP_LOGW("Unknown NpuArch.");
        return EMPTY_LIST;
    }
}

static const std::initializer_list<DataType>& GetProbDtypeSupportList()
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201) {
        return SUPPORTED_PROB_DTYPE_LIST;
    } else {
        OP_LOGW("Unknown NpuArch.");
        return EMPTY_LIST;
    }
}

static bool IsDoubleEqual(double f1, double f2) { return std::abs(f1 - f2) <= std::numeric_limits<double>::epsilon(); }

static bool UseAivConstantKernel(op::DataType dtype, bool isOne)
{
    return dtype == op::DataType::DT_DOUBLE || dtype == op::DataType::DT_INT16 ||
           (dtype == op::DataType::DT_INT64 && isOne);
}

static bool CheckDtypeValidTensor(const aclTensor* self, const aclTensor* prob, const aclTensor* out)
{
    // 检查self的数据类型是否在tanh算子的支持列表内
    const std::initializer_list<op::DataType> currentDtypeSupportList = GetOutDtypeSupportList();
    const std::initializer_list<op::DataType> currentProbDtypeSupportList = GetProbDtypeSupportList();

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, currentDtypeSupportList, return false);

    // 检查prob的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(prob, currentProbDtypeSupportList, return false);

    // 检查self的数据类型是否和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    // 非浮点数输出不保证精度
    if (!CheckType(self->GetDataType(), SUPPORTED_PROB_DTYPE_LIST)) {
        OP_LOGW("Self dtype %s does not guarantee accuracy.", op::ToString(self->GetDataType()).GetString());
    }

    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* prob, const aclTensor* out)
{
    // 检查self的数据类型是否在tanh算子的支持列表内
    const std::initializer_list<op::DataType> currentDtypeSupportList = GetOutDtypeSupportList();
    const std::initializer_list<op::DataType> currentProbDtypeSupportList = GetProbDtypeSupportList();

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, currentDtypeSupportList, return false);

    // 检查prob的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(prob, currentProbDtypeSupportList, return false);

    // 检查self的数据类型是否和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static bool CheckProb(const aclScalar* prob)
{
    const double probability = prob->ToDouble();
    if (!std::isfinite(probability) || probability > 1.0 || probability < 0.0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "prob should be in range 0<=prob<=1 .");
        return false;
    }

    return true;
}

static bool CheckOffset(int64_t offset)
{
    if (offset < 0 || offset % 4 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offset must be a non-negative multiple of 4, but got %ld.", offset);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* self)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s]",
                ToString(self->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormatTensor(const aclTensor* self, const aclTensor* prob)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(prob->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s], prob [%s]",
                ToString(self->GetStorageFormat()).GetString(), ToString(prob->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SHAPE_LENGTH, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static bool HasDenseNdLayout(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->GetStorageFormat() != op::Format::FORMAT_ND ||
        tensor->GetViewFormat() != op::Format::FORMAT_ND || tensor->GetViewOffset() != 0 ||
        tensor->GetStorageOffset() != 0) {
        return false;
    }

    const auto& viewShape = tensor->GetViewShape();
    const auto& strides = tensor->GetViewStrides();
    if (strides.size() != viewShape.GetDimNum()) {
        return false;
    }

    int64_t expectedStride = 1;
    for (int64_t index = static_cast<int64_t>(viewShape.GetDimNum()) - 1; index >= 0; --index) {
        if (strides[static_cast<size_t>(index)] != expectedStride) {
            return false;
        }
        const int64_t dim = viewShape.GetDim(static_cast<size_t>(index));
        if (dim < 0 || (dim != 0 && expectedStride > std::numeric_limits<int64_t>::max() / dim)) {
            return false;
        }
        expectedStride *= dim;
    }
    return true;
}

static uint64_t GetDtypeBytes(DataType dtype)
{
    switch (dtype) {
        case DataType::DT_UINT8:
        case DataType::DT_INT8:
        case DataType::DT_BOOL:
            return 1U;
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_INT16:
            return 2U;
        case DataType::DT_FLOAT:
        case DataType::DT_INT32:
            return 4U;
        case DataType::DT_DOUBLE:
        case DataType::DT_INT64:
            return 8U;
        default:
            return 0U;
    }
}

static bool CanWriteOutDirectly(const aclTensor* tensor)
{
    if (!HasDenseNdLayout(tensor)) {
        return false;
    }
    const int64_t storageElements = tensor->GetStorageShape().GetShapeSize();
    const int64_t viewElements = tensor->GetViewShape().GetShapeSize();
    const uint64_t dtypeBytes = GetDtypeBytes(tensor->GetDataType());
    if (storageElements < 0 || viewElements < 0 || storageElements != viewElements || dtypeBytes == 0U) {
        return false;
    }

    // DSA writes at least one 128-bit block. Ensure the public output storage
    // can hold the packed mask before aliasing it as UINT8.
    const uint64_t elements = static_cast<uint64_t>(viewElements);
    const uint64_t maskBlocks = elements / 128U + (elements % 128U == 0U ? 0U : 1U);
    if (maskBlocks > UINT64_MAX / 16U) {
        return false;
    }
    const uint64_t maskBytes = maskBlocks * 16U;
    return elements <= UINT64_MAX / dtypeBytes && elements * dtypeBytes >= maskBytes;
}

static bool CheckShapeTensor(const aclTensor* self, const aclTensor* prob, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SHAPE_LENGTH, return false);
    OP_CHECK_MAX_DIM(prob, MAX_SHAPE_LENGTH, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static aclnnStatus CheckParamsTensor(const aclTensor* self, const aclTensor* prob, const aclTensor* out)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullTensor(self, prob, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValidTensor(self, prob, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShapeTensor(self, prob, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    CHECK_RET(CheckFormatTensor(self, prob), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* prob, const aclTensor* out)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, prob, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, prob, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的prob的值是否在范围之内，需要根据api定义校验
    CHECK_RET(CheckProb(prob), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(const aclTensor* self, const aclTensor* prob, int64_t seed,
                                                 int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBernoulliTensor, DFX_IN(self, prob, seed, offset), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParamsTensor(self, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckOffset(offset), ACLNN_ERR_PARAM_INVALID);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty() || prob->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto input_contiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(input_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto prob_contiguous = l0op::Contiguous(prob, uniqueExecutor.get());
    CHECK_RET(prob_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* prob_calc = nullptr;
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_3510) {
        prob_calc = l0op::Cast(prob_contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    } else {
        prob_calc = prob_contiguous;
    }

    // 调用StatelessBernoulli算子kernel
    auto op_out = l0op::StatelessBernoulli(input_contiguous, prob_calc, seed, offset, uniqueExecutor.get());
    CHECK_RET(op_out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto view_copy_result = l0op::ViewCopy(op_out, out, uniqueExecutor.get());
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBernoulliGetWorkspaceSize(const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset,
                                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBernoulli, DFX_IN(self, prob, seed, offset), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckOffset(offset), ACLNN_ERR_PARAM_INVALID);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const aclTensor* opOut = nullptr;
    if (curArch == NpuArch::DAV_2201) {
        if (IsDoubleEqual(prob->ToDouble(), 0)) {
            opOut = UseAivConstantKernel(out->GetDataType(), false) ?
                        l0op::BernoulliConstant(inputContiguous, false, uniqueExecutor.get()) :
                        l0op::ZerosLike(inputContiguous, uniqueExecutor.get());
        } else if (IsDoubleEqual(prob->ToDouble(), 1)) {
            opOut = UseAivConstantKernel(out->GetDataType(), true) ?
                        l0op::BernoulliConstant(inputContiguous, true, uniqueExecutor.get()) :
                        l0op::OnesLike(inputContiguous, uniqueExecutor.get());
        } else {
            auto directOut = CanWriteOutDirectly(out) ? out : nullptr;
            opOut = l0op::BernoulliRandom(inputContiguous, prob->ToDouble(), seed, offset, directOut,
                                          uniqueExecutor.get());
        }
    } else if (curArch == NpuArch::DAV_3510) {
        // 调用StatelessBernoulli算子kernel，ARCH3510统一转成float
        auto probScalar = uniqueExecutor->AllocScalar(static_cast<float>(prob->ToDouble()));
        CHECK_RET(probScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto probTensor = uniqueExecutor.get()->ConvertToTensor(probScalar, probScalar->GetDataType());
        CHECK_RET(probTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = l0op::StatelessBernoulli(inputContiguous, probTensor, seed, offset, uniqueExecutor.get());
    } else {
        auto probTensor = uniqueExecutor.get()->ConvertToTensor(prob, prob->GetDataType());
        CHECK_RET(probTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = l0op::StatelessBernoulli(inputContiguous, probTensor, seed, offset, uniqueExecutor.get());
    }
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Keep the executor-owned result separate from the public output tensor;
    // ViewCopy also handles non-contiguous outputs.
    if (opOut != out) {
        auto viewCopyResult = l0op::ViewCopy(opOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor* selfRef, const aclScalar* prob, int64_t seed,
                                                  int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnBernoulliGetWorkspaceSize(selfRef, prob, seed, offset, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* prob, int64_t seed,
                                                        int64_t offset, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnBernoulliTensorGetWorkspaceSize(selfRef, prob, seed, offset, out, workspaceSize, executor);
}

aclnnStatus aclnnBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBernoulliTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBernoulli);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBernoulliTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceBernoulliTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceBernoulli);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
