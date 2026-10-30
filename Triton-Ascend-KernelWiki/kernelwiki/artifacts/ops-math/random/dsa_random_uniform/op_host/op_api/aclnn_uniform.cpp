/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl_rt.h"
#include "aclnn_uniform.h"
#include "random/stateless_random_uniform_v2/op_api/stateless_random_uniform_v2.h"
#include "random/stateless_random_uniform_v3/op_api/stateless_random_uniform_v3.h"
#include "random/stateless_uniform/op_api/stateless_uniform.h"
#include "dsa_random_uniform.h"
#include "aclnn_kernels/cast.h"
#include "math/muls/op_api/muls.h"
#include "math/add/op_api/add.h"
#include "../../../../conversion/concat_d/op_api/concat_d.h"
#include "../../../../conversion/pack/op_api/pack.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "../../../random_common/op_api/aclnn_set_pytorch_random.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self)
{
    OP_CHECK_NULL(self, return false);
    return true;
}

inline static bool CheckSocVersionIsSupportBf16(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static inline bool CheckSocVersionIsSupportDSA(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static bool CheckDtypeValid(const aclTensor* self)
{
    // 如果soc是310系列芯片，则不支持DT_BF16，需要校验拦截
    if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Input dtype of aclnnInplaceUniform is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查self的数据类型是否在Uniform算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, double from, double to)
{
    CHECK_RET(CheckNotNull(self), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);
    if (from > to) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "from cannot be greater than to, from is %lf and to is %lf.", from, to);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclScalar* CreateScalar(float input, op::DataType dtype, aclOpExecutor* executor)
{
    op::fp16_t ratioF16;
    op::bfloat16 ratioBf16;
    OP_LOGI("input %f dtype %d", input, dtype);
    switch (dtype) {
        case op::DataType::DT_FLOAT16:
            ratioF16 = input;
            return executor->AllocScalar(&ratioF16.val, op::DataType::DT_FLOAT16);
        case op::DataType::DT_BF16:
            ratioBf16 = input;
            return executor->AllocScalar(&ratioBf16.value, op::DataType::DT_BF16);
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invalid dtype, must be bfloat16 or float16.");
            return nullptr;
    }
}

static aclTensor* ProcessOffsetTensor(const aclTensor* offsetTensor, int64_t offset, aclOpExecutor* executor)
{
    FVector<int64_t> tmpVector = {static_cast<int64_t>(offset)};
    auto offsetTmpTensor = executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), offsetTensor->GetDataType());
    CHECK_RET(offsetTmpTensor != nullptr, nullptr);
    auto offsetAddOut = l0op::Add(offsetTensor, offsetTmpTensor, executor);
    CHECK_RET(offsetAddOut != nullptr, nullptr);
    // concat
    FVector<int64_t> zeroVector = {static_cast<int64_t>(0)};
    auto zeroTensor = executor->ConvertToTensor(zeroVector.data(), zeroVector.size(), offsetTensor->GetDataType());
    CHECK_RET(zeroTensor != nullptr, nullptr);
    FVector<const aclTensor*> tensorListOnce;
    tensorListOnce.emplace_back(zeroTensor);
    tensorListOnce.emplace_back(offsetAddOut);
    auto tensorList = executor->AllocTensorList(tensorListOnce.data(), tensorListOnce.size());
    auto concatTensor = l0op::ConcatD(tensorList, 0, executor);
    CHECK_RET(concatTensor != nullptr, nullptr);

    return concatTensor;
}

// DT_DOUBLE 路径：StatelessRandomUniformV2 + Muls(to-from) + Add(from)
static const aclTensor* uniformDoublePath(const aclTensor* uniformOut, double from, double to, aclOpExecutor* executor)
{
    CHECK_RET(uniformOut != nullptr, nullptr);

    auto mulsOut = l0op::Muls(uniformOut, to - from, executor);
    CHECK_RET(mulsOut != nullptr, nullptr);

    auto fromTensor = executor->ConvertToTensor(&from, 1, mulsOut->GetDataType());
    return l0op::Add(mulsOut, fromTensor, executor);
}

// 非DSA路径的 uniform 计算公共函数
static const aclTensor* uniformDavidPath(const aclTensor* selfRef, uint64_t seed, uint64_t offset, double from,
                                         double to, aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfRef->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 uniform");
            int32_t uniformV3ScaleMode = 0;
            auto fromOut = static_cast<float>(from);
            auto toOut = static_cast<float>(to);
            return l0op::StatelessRandomUniformV3(selfRef, seed, offset, fromOut, toOut, uniformV3ScaleMode, executor);
        }
        return l0op::StatelessUniform(selfRef, seed, offset, from, to, executor);
    } else {
        int32_t alg = 1;
        auto uniformOut = l0op::StatelessRandomUniformV2(selfRef, seed, offset, alg, executor);
        return uniformDoublePath(uniformOut, from, to, executor);
    }
}

// 非DSA路径的 uniform 计算公共函数（Tensor seed/offset 版本）
static const aclTensor* uniformTensorDavidPath(const aclTensor* selfRef, const aclTensor* seedTensor,
                                               const aclTensor* offsetTensor, uint64_t offset, double from, double to,
                                               aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfRef->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 uniform");
            FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
            aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
            auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
            auto resultAddOut = l0op::Add(offsetTensor, tmpTensor, executor);
            CHECK_RET(resultAddOut != nullptr, nullptr);

            int32_t uniformV3ScaleMode = 0;
            auto fromOut = static_cast<float>(from);
            auto toOut = static_cast<float>(to);
            return l0op::StatelessRandomUniformV3(selfRef, seedTensor, resultAddOut, fromOut, toOut, uniformV3ScaleMode,
                                                  executor);
        }
        auto seedInt64 = l0op::Cast(seedTensor, op::DataType::DT_INT64, executor);
        CHECK_RET(seedInt64 != nullptr, nullptr);
        auto offsetInt64 = l0op::Cast(offsetTensor, op::DataType::DT_INT64, executor);
        CHECK_RET(offsetInt64 != nullptr, nullptr);

        FVector<int64_t> offsetVector{static_cast<int64_t>(offset)};
        auto tmpTensor = executor->ConvertToTensor(offsetVector.data(), offsetVector.size(), op::DataType::DT_INT64);
        auto resultAddOut = l0op::Add(offsetInt64, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);

        return l0op::StatelessUniform(selfRef, seedInt64, resultAddOut, from, to, executor);
    } else {
        // V2 路径：保持原有 [0, offset] concat 逻辑
        FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
        aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
        auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
        auto resultAddOut = l0op::Add(offsetTensor, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);

        int32_t alg = 1;
        auto uniformOut = l0op::StatelessRandomUniformV2(selfRef, seedTensor, resultAddOut, alg, executor);
        return uniformDoublePath(uniformOut, from, to, executor);
    }
}

aclnnStatus aclnnInplaceUniformGetWorkspaceSize(const aclTensor* selfRef, double from, double to, uint64_t seed,
                                                uint64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceUniform, DFX_IN(selfRef, from, to, seed, offset), DFX_OUT(selfRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(selfRef, from, to);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto inputShape = op::ToShapeVector(selfRef->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        op::DataType dtype = selfRef->GetDataType();
        aclScalar* fromScalar = nullptr;
        aclScalar* toScalar = nullptr;
        if (dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16) {
            fromScalar = CreateScalar(static_cast<float>(from), selfRef->GetDataType(), uniqueExecutor.get());
            toScalar = CreateScalar(static_cast<float>(to), selfRef->GetDataType(), uniqueExecutor.get());
        } else {
            fromScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(from));
            toScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(to));
        }
        CHECK_RET(fromScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(toScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::DSARandomUniform(inputShapeArray, seed, offset, fromScalar, toScalar, uniqueExecutor.get());
    } else {
        computeOut = uniformDavidPath(selfRef, seed, offset, from, to, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(computeOut, selfRef->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(const aclTensor* selfRef, double from, double to,
                                                      const aclTensor* seedTensor, const aclTensor* offsetTensor,
                                                      uint64_t offset, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceUniformTensor, DFX_IN(selfRef, from, to, seedTensor, offsetTensor, offset),
                   DFX_OUT(selfRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(selfRef, from, to);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto inputShape = op::ToShapeVector(selfRef->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        op::DataType dtype = selfRef->GetDataType();
        aclScalar* fromScalar = nullptr;
        aclScalar* toScalar = nullptr;
        if (dtype == op::DataType::DT_FLOAT16 || dtype == op::DataType::DT_BF16) {
            fromScalar = CreateScalar(static_cast<float>(from), selfRef->GetDataType(), uniqueExecutor.get());
            toScalar = CreateScalar(static_cast<float>(to), selfRef->GetDataType(), uniqueExecutor.get());
        } else {
            fromScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(from));
            toScalar = uniqueExecutor.get()->AllocScalar(static_cast<float>(to));
        }
        CHECK_RET(fromScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(toScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto concatTensor = ProcessOffsetTensor(offsetTensor, offset, uniqueExecutor.get());
        CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::DSARandomUniformTensor(inputShapeArray, seedTensor, concatTensor, fromScalar, toScalar,
                                                  uniqueExecutor.get());
    } else {
        computeOut = uniformTensorDavidPath(selfRef, seedTensor, offsetTensor, offset, from, to, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(computeOut, selfRef->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceUniform(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceUniform);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceUniformTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceUniformTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
