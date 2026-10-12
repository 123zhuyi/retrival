/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "aclnn_bernoulli.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "bernoulli_mask.h"
#include "random/stateless_bernoulli/op_api/stateless_bernoulli.h"
#include "math/zero_op/op_api/zero_op.h"
#include "math/ones_like/op_api/ones_like.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
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
static const int64_t VEC_BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910_PROB_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910B_PROB_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,   op::DataType::DT_UINT8, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,  op::DataType::DT_UINT64,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ARCH3510_PROB_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static bool CheckNotNull(const aclTensor* self, const aclScalar* prob, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(prob, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910_93) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase()) {
        return ARCH3510_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGW("Unknown SocVersion.");
        return EMPTY_LIST;
    }
}

static const std::initializer_list<DataType>& GetProbDtypeSupportList()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910) {
        return ASCEND910_PROB_DTYPE_SUPPORT_LIST;
    } else if (socVersion >= SocVersion::ASCEND910B && socVersion <= SocVersion::ASCEND910_93) {
        return ASCEND910B_PROB_DTYPE_SUPPORT_LIST;
    } else if (IsRegBase()) {
        return ARCH3510_PROB_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGW("Unknown SocVersion.");
        return EMPTY_LIST;
    }
}

static bool IsDoubleEqual(double f1, double f2) { return std::abs(f1 - f2) <= std::numeric_limits<double>::epsilon(); }

static bool CheckDtypeValid(const aclTensor* self, const aclScalar* prob, const aclTensor* out)
{
    // 检查self的数据类型是否在Bernoulli算子的支持列表内
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
    const double probValue = prob->ToDouble();
    if (!std::isfinite(probValue) || probValue > 1 || probValue < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "prob should be in range 0<=prob<=1 .");
        return false;
    }

    return true;
}

static bool CheckOffset(int64_t offset)
{
    if (offset % 4 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "offset must be a multiple of 4, but got %ld.", offset);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* tensor, const char* parameterName)
{
    // Private layouts cannot be interpreted by the dense linear alias path or ViewCopy.
    if (op::IsPrivateFormat(tensor->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, %s [%s]",
                parameterName, ToString(tensor->GetStorageFormat()).GetString());
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

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* prob, int64_t offset, const aclTensor* out)
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
    CHECK_RET(CheckFormat(self, "self") && CheckFormat(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查随机数偏移量是否满足接口约束
    CHECK_RET(CheckOffset(offset), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool InferDSAOutShapeV2(const aclIntArray* shape, int64_t& packedBytes)
{
    uint64_t elements = 1;
    for (size_t index = 0; index < shape->Size(); index++) {
        const int64_t dim = (*shape)[index];
        if (dim < 0) {
            return false;
        }
        if (dim == 0) {
            packedBytes = 0;
            return true;
        }
        const uint64_t unsignedDim = static_cast<uint64_t>(dim);
        if (elements > std::numeric_limits<uint64_t>::max() / unsignedDim) {
            return false;
        }
        elements *= unsignedDim;
    }

    const uint64_t maskBlockBytes = static_cast<uint64_t>(VEC_BIT_NUMBER) / static_cast<uint64_t>(UINT8_BIT_NUMBER);
    const uint64_t maskBlocks = (elements - 1) / static_cast<uint64_t>(VEC_BIT_NUMBER) + 1;
    if (maskBlocks > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / maskBlockBytes) {
        return false;
    }
    packedBytes = static_cast<int64_t>(maskBlocks * maskBlockBytes);
    return true;
}

// The direct packed-mask path is used only on DAV_2201. Keep this mapping
// scoped to the output dtypes supported by that architecture.
static uint64_t GetDav2201OutputTypeBytes(DataType dtype)
{
    switch (dtype) {
        case DataType::DT_UINT8:
        case DataType::DT_INT8:
        case DataType::DT_BOOL:
            return 1;
        case DataType::DT_FLOAT16:
        case DataType::DT_BF16:
        case DataType::DT_INT16:
            return 2;
        case DataType::DT_FLOAT:
        case DataType::DT_INT32:
            return 4;
        case DataType::DT_DOUBLE:
        case DataType::DT_INT64:
            return 8;
        default:
            return 0;
    }
}

static bool HasDenseViewLayout(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->GetViewOffset() != 0 || tensor->GetStorageOffset() != 0) {
        return false;
    }
    const auto& viewShape = tensor->GetViewShape();
    const auto& strides = tensor->GetViewStrides();
    if (strides.size() != viewShape.GetDimNum()) {
        return false;
    }
    int64_t expectedStride = 1;
    for (int64_t i = static_cast<int64_t>(viewShape.GetDimNum()) - 1; i >= 0; --i) {
        if (strides[static_cast<size_t>(i)] != expectedStride) {
            return false;
        }
        const int64_t dim = static_cast<int64_t>(viewShape.GetDim(static_cast<size_t>(i)));
        if (dim < 0 || (dim != 0 && expectedStride > std::numeric_limits<int64_t>::max() / dim)) {
            return false;
        }
        expectedStride *= dim;
    }
    return true;
}

static bool CanWriteOutDirectly(const aclTensor* tensor)
{
    if (!HasDenseViewLayout(tensor)) {
        return false;
    }
    // Framework adapters may describe a dense N-D view over a flat 1-D
    // storage shape, so compare element counts rather than shape vectors.
    // The aliased kernel is tiled from the output storage shape and therefore
    // may write every backing element. Only use it when the dense logical view
    // covers the entire storage; a larger backing store must take ViewCopy so
    // bytes outside the logical view remain untouched.
    const int64_t storageElements = tensor->GetStorageShape().GetShapeSize();
    const int64_t viewElements = tensor->GetViewShape().GetShapeSize();
    return storageElements >= 0 && viewElements >= 0 && storageElements == viewElements;
}

static bool LaunchDSAGenBitMask(uint64_t count, uint64_t seed, uint64_t offset, const aclScalar* dropout,
                                aclTensor* out, aclOpExecutor* executor)
{
    if (dropout == nullptr || out == nullptr || executor == nullptr) {
        return false;
    }
    L0_DFX(LaunchDSAGenBitMask, count, seed, offset, dropout);

    auto* args = op::GetOpArgContext(OP_INPUT(count, seed, offset, dropout), OP_OUTPUT(out), OP_ATTR(0));
    if (args == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create DSAGenBitMask argument context.");
        return false;
    }

    static const uint32_t dsaGenBitMaskOpType = op::GenOpTypeId("DSAGenBitMask");
    CreatDSAKernelLauncher("DSAGenBitMask", dsaGenBitMaskOpType, DSAGenBitMaskTaskType, executor, args);
    return true;
}

static aclTensor* CreateInt64BitView(const aclTensor* tensor, aclOpExecutor* executor)
{
    auto bits = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                     tensor->GetViewStrides(), tensor->GetViewOffset());
    CHECK_RET(bits != nullptr, nullptr);
    bits->SetDataType(DataType::DT_INT64);
    return bits;
}

static const aclTensor* ViewCopyWithDoubleSupport(const aclTensor* src, aclTensor* dst, aclOpExecutor* executor)
{
    if (src->GetDataType() != DataType::DT_DOUBLE) {
        return l0op::ViewCopy(src, dst, executor);
    }

    // The A2 AICore ViewCopy kernel moves 64-bit values through its INT64
    // specialization but does not advertise DOUBLE. Reinterpret both tensors
    // as INT64 views so non-contiguous fp64 output preserves the exact bit
    // pattern without a numerical cast or an AICPU fallback.
    auto srcBits = CreateInt64BitView(src, executor);
    CHECK_RET(srcBits != nullptr, nullptr);
    auto dstBits = CreateInt64BitView(dst, executor);
    CHECK_RET(dstBits != nullptr, nullptr);
    return l0op::ViewCopy(srcBits, dstBits, executor);
}

aclnnStatus GetBernoulliByDSA(const aclTensor* input, const aclScalar* prob, int64_t seed, int64_t offset,
                              aclTensor* directOut, const aclTensor*& doMaskOut, aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(input->GetViewShape());
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    CHECK_RET(inputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t shapeSize = 0;
    CHECK_RET(InferDSAOutShapeV2(inputSizeArray, shapeSize) && shapeSize > 0, ACLNN_ERR_PARAM_INVALID);
    auto probScalar = executor->AllocScalar(static_cast<float>(1 - prob->ToDouble()));
    CHECK_RET(probScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* mask = nullptr;
    bool maskAliasesOut = false;
    const uint64_t outputTypeBytes = GetDav2201OutputTypeBytes(input->GetDataType());
    const int64_t outputElementsSigned = input->GetViewShape().GetShapeSize();
    CHECK_RET(outputElementsSigned > 0, ACLNN_ERR_PARAM_INVALID);
    const uint64_t outputElements = static_cast<uint64_t>(outputElementsSigned);
    const uint64_t dsaCount = static_cast<uint64_t>(shapeSize) * static_cast<uint64_t>(UINT8_BIT_NUMBER);
    if (directOut != nullptr && outputTypeBytes != 0 &&
        outputElements <= std::numeric_limits<uint64_t>::max() / outputTypeBytes &&
        outputElements * outputTypeBytes >= static_cast<uint64_t>(shapeSize)) {
        auto aliasedMask = executor->CreateView(directOut, op::Shape{shapeSize}, 0);
        CHECK_RET(aliasedMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
        aliasedMask->SetDataType(op::DataType::DT_UINT8);

        const bool dsaLaunched = LaunchDSAGenBitMask(dsaCount, seed, offset, probScalar, aliasedMask, executor);
        CHECK_RET(dsaLaunched, ACLNN_ERR_INNER_NULLPTR);
        mask = aliasedMask;
        maskAliasesOut = true;
    } else {
        auto allocatedMask = executor->AllocTensor(op::Shape{shapeSize}, op::DataType::DT_UINT8);
        CHECK_RET(allocatedMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(LaunchDSAGenBitMask(dsaCount, seed, offset, probScalar, allocatedMask, executor),
                  ACLNN_ERR_INNER_NULLPTR);
        mask = allocatedMask;
        CHECK_RET(mask != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (directOut != nullptr) {
        doMaskOut = l0op::BernoulliMask(mask, directOut, maskAliasesOut, executor);
    } else {
        doMaskOut = l0op::BernoulliMask(mask, input, executor);
    }
    CHECK_RET(doMaskOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

// This custom package overrides only the scalar-probability APIs named by the
// community task. Tensor-probability APIs continue to resolve from the system
// libopapi, preserving its full dtype/layout behavior instead of shadowing it
// with an unrelated experimental dependency closure.
static aclnnStatus BernoulliGetWorkspaceSizeCommon(const aclTensor* self, const aclScalar* prob, int64_t seed,
                                                   int64_t offset, aclTensor* out, uint64_t* workspaceSize,
                                                   aclOpExecutor** executor)
{
    // 固定写法，参数检查
    auto ret = CheckParams(self, prob, offset, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const aclTensor* opOut = nullptr;
    if (curArch == NpuArch::DAV_2201) {
        // 调用DSAGenBitMask算子kernel
        const aclTensor* doMaskOut = nullptr;
        if (IsDoubleEqual(prob->ToDouble(), 0)) {
            doMaskOut = l0op::ZerosLike(self, uniqueExecutor.get());
        } else if (IsDoubleEqual(prob->ToDouble(), 1)) {
            doMaskOut = l0op::OnesLike(self, uniqueExecutor.get());
        } else {
            aclTensor* directOut = CanWriteOutDirectly(out) ? out : nullptr;
            auto executeResult = GetBernoulliByDSA(self, prob, seed, offset, directOut, doMaskOut,
                                                   uniqueExecutor.get());
            CHECK_RET(executeResult == ACLNN_SUCCESS, executeResult);
        }
        CHECK_RET(doMaskOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = doMaskOut;
    } else if (curArch == NpuArch::DAV_3510) {
        auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用StatelessBernoulli算子kernel，ARCH3510统一转成float
        auto probScalar = uniqueExecutor->AllocScalar(static_cast<float>(prob->ToDouble()));
        CHECK_RET(probScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto probTensor = uniqueExecutor.get()->ConvertToTensor(probScalar, probScalar->GetDataType());
        CHECK_RET(probTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = l0op::StatelessBernoulli(inputContiguous, probTensor, seed, offset, uniqueExecutor.get());
    } else {
        auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto probTensor = uniqueExecutor.get()->ConvertToTensor(prob, prob->GetDataType());
        CHECK_RET(probTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = l0op::StatelessBernoulli(inputContiguous, probTensor, seed, offset, uniqueExecutor.get());
    }
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    if (opOut != out) {
        auto viewCopyResult = ViewCopyWithDoubleSupport(opOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

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
    return BernoulliGetWorkspaceSizeCommon(self, prob, seed, offset, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(const aclTensor* selfRef, const aclScalar* prob, int64_t seed,
                                                  int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnInplaceBernoulli, DFX_IN(selfRef, prob, seed, offset), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    return BernoulliGetWorkspaceSizeCommon(selfRef, prob, seed, offset, out, workspaceSize, executor);
}

aclnnStatus aclnnBernoulli(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBernoulli);
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
