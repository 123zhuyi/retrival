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
#include "aclnn_random.h"
#include "random/stateless_random_uniform_v2/op_api/stateless_random_uniform_v2.h"
#include "random/stateless_random/op_api/stateless_random.h"
#include "random/stateless_random_uniform_v3/op_api/stateless_random_uniform_v3.h"
#include "dsa_random_uniform.h"
#include "opdev/platform.h"
#include "math/round/op_api/round.h"
#include "aclnn_kernels/cast.h"
#include "math/muls/op_api/muls.h"
#include "math/sub/op_api/sub.h"
#include "conversion/view_copy/op_api/view_copy.h"
#include "math/add/op_api/add.h"
#include "../../../../conversion/concat_d/op_api/concat_d.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
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

namespace {
constexpr int32_t FLOAT16_DIGITS = 11;                                   // Number of digits for DT_FLOAT16
constexpr int32_t BF16_DIGITS = 8;                                       // Number of digits for DT_BF16
constexpr int32_t FLOAT32_DIGITS = std::numeric_limits<float_t>::digits; // Number of digits for DT_FLOAT
constexpr int32_t DOUBLE_DIGITS = std::numeric_limits<double>::digits;   // Number of digits for DT_DOUBLE

static aclnnStatus updateFrom(int64_t& from, op::DataType dtype)
{
    op::fp16_t fromF16;
    op::bfloat16 fromBf16;
    OP_LOGI("before updateFrom: from %ld, target dtype %d", from, dtype);
    int32_t digits = 0;
    int64_t fromPlusOne = 0;
    switch (dtype) {
        case op::DataType::DT_FLOAT16:
            fromF16 = from + 1;
            fromPlusOne = static_cast<int64_t>(fromF16);
            digits = FLOAT16_DIGITS;
            break;
        case op::DataType::DT_BF16:
            fromBf16 = from + 1;
            fromPlusOne = static_cast<int64_t>(fromBf16);
            digits = BF16_DIGITS;
            break;
        case op::DataType::DT_FLOAT:
            fromPlusOne = static_cast<int64_t>(static_cast<float_t>(from + 1));
            digits = FLOAT32_DIGITS;
            break;
        case op::DataType::DT_DOUBLE:
            fromPlusOne = static_cast<int64_t>(static_cast<double>(from + 1));
            digits = DOUBLE_DIGITS;
            break;
        default:
            OP_LOGI("dtype must be bfloat16, float16, float32 or double.");
            return ACLNN_SUCCESS;
    }
    if (fromPlusOne < from) {
        int64_t from_ = std::abs(from + 1);
        int32_t n = 0;
        while ((from_ >>= 1) != 0LL) {
            ++n;
        }
        from = fromPlusOne + (1LL << (n - digits + 1));
    }
    OP_LOGI("after updateFrom: from %ld, target dtype %d", from, dtype);
    return ACLNN_SUCCESS;
}

static aclnnStatus updateTo(int64_t& to, op::DataType dtype)
{
    op::fp16_t toF16;
    op::bfloat16 toBf16;
    OP_LOGI("before updateTo: to %ld, target dtype %d", to, dtype);
    int32_t digits = 0;
    int64_t toMinusOne = 0;
    switch (dtype) {
        case op::DataType::DT_FLOAT16:
            toF16 = to - 1;
            toMinusOne = static_cast<int64_t>(toF16);
            digits = FLOAT16_DIGITS;
            break;
        case op::DataType::DT_BF16:
            toBf16 = to - 1;
            toMinusOne = static_cast<int64_t>(toBf16);
            digits = BF16_DIGITS;
            break;
        case op::DataType::DT_FLOAT:
            toMinusOne = static_cast<int64_t>(static_cast<float_t>(to - 1));
            digits = FLOAT32_DIGITS;
            break;
        case op::DataType::DT_DOUBLE:
            toMinusOne = static_cast<int64_t>(static_cast<double>(to - 1));
            digits = DOUBLE_DIGITS;
            break;
        default:
            OP_LOGI("dtype must be bfloat16, float16, float32 or double.");
            return ACLNN_SUCCESS;
    }
    if (toMinusOne >= to) {
        int64_t to_ = std::abs(to - 1);
        int32_t n = 0;
        while ((to_ >>= 1) != 0LL) {
            ++n;
        }
        to = toMinusOne - (1LL << (n - digits + 1));
    }
    OP_LOGI("after updateTo: to %ld, target dtype %d", to, dtype);
    return ACLNN_SUCCESS;
}
} // namespace

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32, op::DataType::DT_INT64,     op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_BF16,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> INT_DTYPE_LIST = {op::DataType::DT_INT32, op::DataType::DT_INT64,
                                                                   op::DataType::DT_INT16, op::DataType::DT_INT8,
                                                                   op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                     op::DataType::DT_BF16};
static bool CheckNotNull(const aclTensor* selfRef)
{
    OP_CHECK_NULL(selfRef, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* selfRef)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, DTYPE_SUPPORT_LIST, return false);
    return true;
}

constexpr size_t MAX_DIM_LEN = 8;
static bool CheckShape(const aclTensor* selfRef)
{
    OP_CHECK_MAX_DIM(selfRef, MAX_DIM_LEN, return false);
    return true;
}

static bool CheckFromToRange(int64_t from, int64_t to)
{
    if (from >= to) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "from %ld must be less than to %ld.", from, to);
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* selfRef)
{
    CHECK_RET(CheckNotNull(selfRef), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckDtypeValid(selfRef), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(selfRef), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline bool CheckSocVersionIsSupportDSA(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static const aclTensor* CastProcess(const aclTensor* selfRef, const aclTensor* computeOut, aclOpExecutor* executor)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(curArch) && CheckType(selfRef->GetDataType(), FLOAT_DTYPE_LIST)) {
        return computeOut;
    }

    if (!CheckType(selfRef->GetDataType(), INT_DTYPE_LIST)) {
        auto castResultInt64 = l0op::Cast(computeOut, op::DataType::DT_INT64, executor);
        CHECK_RET(castResultInt64 != nullptr, nullptr);
        auto castResult = l0op::Cast(castResultInt64, selfRef->GetDataType(), executor);
        return castResult;
    } else {
        auto castResult = l0op::Cast(computeOut, selfRef->GetDataType(), executor);
        return castResult;
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

// DT_DOUBLE 路径：StatelessRandomUniformV2 + Muls(from) + Sub + Muls(to) + Sub
static const aclTensor* randomDoublePath(const aclTensor* uniformOpOut, int64_t from, int64_t to,
                                         aclOpExecutor* executor)
{
    CHECK_RET(uniformOpOut != nullptr, nullptr);

    auto mulFromOut = l0op::Muls(uniformOpOut, from, executor);
    CHECK_RET(mulFromOut != nullptr, nullptr);

    int64_t size = 1;
    auto fromTensor = executor->ConvertToTensor(&from, size, mulFromOut->GetDataType());
    auto temp = l0op::Sub(mulFromOut, fromTensor, executor);
    CHECK_RET(temp != nullptr, nullptr);

    auto mulToOpOut = l0op::Muls(uniformOpOut, to, executor);
    CHECK_RET(mulToOpOut != nullptr, nullptr);

    return l0op::Sub(mulToOpOut, temp, executor);
}

// 非DSA路径的 random 离散均匀计算
// uniformV3ScaleMode: 1 = random (离散)
static const aclTensor* randomDavidPath(const aclTensor* selfRef, int64_t seed, int64_t offset, int64_t from,
                                        int64_t to, aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfRef->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 uniform");
            int32_t uniformV3ScaleMode = 1;
            auto fromOut = static_cast<float>(from);
            auto toOut = static_cast<float>(to);
            return l0op::StatelessRandomUniformV3(selfRef, seed, offset, fromOut, toOut, uniformV3ScaleMode, executor);
        }
        auto randomOpOut = l0op::StatelessRandom(selfRef, seed, offset, from, to, executor);
        if (selfRef->GetDataType() == op::DataType::DT_BOOL && randomOpOut != nullptr) {
            auto castOut = l0op::Cast(randomOpOut, op::DataType::DT_INT32, executor);
            return castOut;
        }
        return randomOpOut;
    } else {
        int32_t alg = 1;
        auto seedVal = static_cast<uint64_t>(seed);
        auto offsetVal = static_cast<uint64_t>(offset);
        auto uniformOpOut = l0op::StatelessRandomUniformV2(selfRef, seedVal, offsetVal, alg, executor);
        return randomDoublePath(uniformOpOut, from, to, executor);
    }
}

// 非DSA路径的 random 离散均匀计算（Tensor seed/offset 版本）
static const aclTensor* randomTensorDavidPath(const aclTensor* selfRef, const aclTensor* seedTensor,
                                              const aclTensor* offsetTensor, int64_t offset, int64_t from, int64_t to,
                                              aclOpExecutor* executor)
{
    // seedTensor/offsetTensor 必须是 int64
    if (seedTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Random seedTensor dtype must be DT_INT64, got %s.",
                op::ToString(seedTensor->GetDataType()).GetString());
        return nullptr;
    }
    if (offsetTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Random offsetTensor dtype must be DT_INT64, got %s.",
                op::ToString(offsetTensor->GetDataType()).GetString());
        return nullptr;
    }

    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfRef->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 uniform");
            // seedTensor/offsetTensor 从 int64 Cast 为 uint64（V3 OpDef 要求 uint64）
            auto randomSeedU64 = l0op::Cast(seedTensor, op::DataType::DT_UINT64, executor);
            CHECK_RET(randomSeedU64 != nullptr, nullptr);
            auto randomOffsetU64 = l0op::Cast(offsetTensor, op::DataType::DT_UINT64, executor);
            CHECK_RET(randomOffsetU64 != nullptr, nullptr);

            FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
            aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
            auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
            auto resultAddOut = l0op::Add(randomOffsetU64, tmpTensor, executor);
            CHECK_RET(resultAddOut != nullptr, nullptr);

            int32_t uniformV3ScaleMode = 1;
            auto fromOut = static_cast<float>(from);
            auto toOut = static_cast<float>(to);
            return l0op::StatelessRandomUniformV3(selfRef, randomSeedU64, resultAddOut, fromOut, toOut,
                                                  uniformV3ScaleMode, executor);
        }
        FVector<int64_t> offsetVector{static_cast<int64_t>(offset)};
        aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 1);
        auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_INT64);
        auto resultAddOut = l0op::Add(offsetTensor, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);
        auto randomOpOut = l0op::StatelessRandom(selfRef, seedTensor, resultAddOut, from, to, executor);
        if (selfRef->GetDataType() == op::DataType::DT_BOOL && randomOpOut != nullptr) {
            auto castOut = l0op::Cast(randomOpOut, op::DataType::DT_INT32, executor);
            return castOut;
        }
        return randomOpOut;
    } else {
        auto randomSeedU64 = l0op::Cast(seedTensor, op::DataType::DT_UINT64, executor);
        CHECK_RET(randomSeedU64 != nullptr, nullptr);
        auto randomOffsetU64 = l0op::Cast(offsetTensor, op::DataType::DT_UINT64, executor);
        CHECK_RET(randomOffsetU64 != nullptr, nullptr);

        FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
        aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
        auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
        auto resultAddOut = l0op::Add(randomOffsetU64, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);
        int32_t alg = 1;
        auto uniformOpOut = l0op::StatelessRandomUniformV2(selfRef, randomSeedU64, resultAddOut, alg, executor);
        return randomDoublePath(uniformOpOut, from, to, executor);
    }
}

aclnnStatus aclnnInplaceRandomGetWorkspaceSize(const aclTensor* selfRef, int64_t from, int64_t to, int64_t seed,
                                               int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRandom, DFX_IN(selfRef, from, to, seed, offset), DFX_OUT(selfRef));
    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckFromToRange(from, to), ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 更新from、to，将from、to移动到下一个最接近且不会超出[from, to)范围的值
    ret = updateFrom(from, selfRef->GetDataType());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = updateTo(to, selfRef->GetDataType());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto inputShape = op::ToShapeVector(selfRef->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto low = uniqueExecutor.get()->AllocScalar(static_cast<float>(from));
        CHECK_RET(low != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto high = uniqueExecutor.get()->AllocScalar(static_cast<float>(to));
        CHECK_RET(high != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::DSARandomUniform(inputShapeArray, seed, offset, low, high, uniqueExecutor.get());
    } else {
        computeOut = randomDavidPath(selfRef, seed, offset, from, to, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfRef->GetDataType() == op::DataType::DT_BOOL) {
        int64_t decimals = 0;
        computeOut = l0op::RoundDecimals(computeOut, decimals, uniqueExecutor.get());
        CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto castResult = CastProcess(selfRef, computeOut, uniqueExecutor.get());
    CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castResult, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceRandom(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRandom);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(const aclTensor* selfRef, int64_t from, int64_t to,
                                                     const aclTensor* seedTensor, const aclTensor* offsetTensor,
                                                     int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRandomTensor, DFX_IN(selfRef, from, to, seedTensor, offsetTensor, offset),
                   DFX_OUT(selfRef));

    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckFromToRange(from, to), ACLNN_ERR_PARAM_INVALID);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 更新from、to，将from、to移动到下一个最接近且不会超出[from, to)范围的值
    ret = updateFrom(from, selfRef->GetDataType());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = updateTo(to, selfRef->GetDataType());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto concatTensor = ProcessOffsetTensor(offsetTensor, offset, uniqueExecutor.get());
        CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputShape = op::ToShapeVector(selfRef->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto low = uniqueExecutor.get()->AllocScalar(static_cast<float>(from));
        CHECK_RET(low != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto high = uniqueExecutor.get()->AllocScalar(static_cast<float>(to));
        CHECK_RET(high != nullptr, ACLNN_ERR_INNER_NULLPTR);

        computeOut = l0op::DSARandomUniformTensor(inputShapeArray, seedTensor, concatTensor, low, high,
                                                  uniqueExecutor.get());
    } else {
        computeOut = randomTensorDavidPath(selfRef, seedTensor, offsetTensor, offset, from, to, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfRef->GetDataType() == op::DataType::DT_BOOL) {
        int64_t decimals = 0;
        computeOut = l0op::RoundDecimals(computeOut, decimals, uniqueExecutor.get());
        CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto castResult = CastProcess(selfRef, computeOut, uniqueExecutor.get());
    CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castResult, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceRandomTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRandomTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRandomWithoutFromToGetWorkspaceSize(const aclTensor* selfRef, int64_t seed, int64_t offset,
                                                            uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRandomWithoutFromTo, DFX_IN(selfRef, seed, offset), DFX_OUT(selfRef));
    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor* computeOut = l0op::StatelessRandomWithoutFromTo(selfContiguous, seed, offset,
                                                                     uniqueExecutor.get());
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfContiguous->GetDataType() == op::DataType::DT_BOOL) {
        int64_t decimals = 0;
        auto castOut = l0op::Cast(computeOut, op::DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::RoundDecimals(castOut, decimals, uniqueExecutor.get());
        CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto castResult = CastProcess(selfContiguous, computeOut, uniqueExecutor.get());
    CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castResult, selfContiguous, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceRandomWithoutFromTo(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRandomWithoutFromTo);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRandomWithoutFromToTensorGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* seedTensor,
                                                                  const aclTensor* offsetTensor, int64_t offset,
                                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRandomWithoutFromToTensor, DFX_IN(selfRef, seedTensor, offsetTensor, offset),
                   DFX_OUT(selfRef));

    auto ret = CheckParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (selfRef->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // seedTensor/offsetTensor 必须是 int64
    if (seedTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Random seedTensor dtype must be DT_INT64, got %s.",
                op::ToString(seedTensor->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (offsetTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Random offsetTensor dtype must be DT_INT64, got %s.",
                op::ToString(offsetTensor->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    FVector<int64_t> offsetVector{static_cast<int64_t>(offset)};
    aclIntArray* offsetList = uniqueExecutor.get()->AllocIntArray(offsetVector.data(), 1);
    auto tmpTensor = uniqueExecutor.get()->ConvertToTensor(offsetList, op::DataType::DT_INT64);
    auto resultAddOut = l0op::Add(offsetTensor, tmpTensor, uniqueExecutor.get());
    CHECK_RET(resultAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* computeOut = l0op::StatelessRandomWithoutFromTo(selfContiguous, seedTensor, resultAddOut,
                                                                     uniqueExecutor.get());
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfContiguous->GetDataType() == op::DataType::DT_BOOL) {
        int64_t decimals = 0;
        auto castOut = l0op::Cast(computeOut, op::DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::RoundDecimals(castOut, decimals, uniqueExecutor.get());
        CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto castResult = CastProcess(selfContiguous, computeOut, uniqueExecutor.get());
    CHECK_RET(castResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castResult, selfContiguous, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceRandomWithoutFromToTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                  aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRandomWithoutFromToTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
