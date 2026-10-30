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
#include "aclnn_normal.h"
#include "math/add/op_api/add.h"
#include "math/mul/op_api/mul.h"
#include "random/stateless_random_normal_v2/op_api/stateless_random_normal_v2.h"
#include "random/stateless_random_normal_v3/op_api/stateless_random_normal_v3.h"
#include "random/stateless_normal/op_api/stateless_normal.h"
#include "dsa_random_normal.h"
#include "../../../../conversion/concat_d/op_api/concat_d.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "../../../random_common/op_api/aclnn_set_pytorch_random.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_INT8,   op::DataType::DT_UINT8,
    op::DataType::DT_BOOL,    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> OUTPUT_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static inline bool CheckSocVersionIsSupportBf16(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static inline bool CheckSocVersionIsSupportDSA(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (CheckSocVersionIsSupportBf16()) {
        return DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline bool CheckDtypeValid(const aclTensor* self)
{
    if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype DT_BF16 not support in current soc version.");
        return false;
    }
    const auto& supportList = GetDtypeSupportList();
    // 检查self的数据类型是否在normal_算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 非浮点数输出不保证精度
    if (!CheckType(self->GetDataType(), OUTPUT_SUPPORT_LIST)) {
        OP_LOGW("Self dtype %s does not guarantee accuracy.", op::ToString(self->GetDataType()).GetString());
    }

    return true;
}

static inline bool CheckNotNull(const aclTensor* self, float std)
{
    OP_CHECK_NULL(self, return false);
    if (std < 0) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Normal expects std >= 0.0, but found std [%f].", std);
        return false;
    }
    return true;
}

static inline bool CheckDimValid(const aclTensor* self)
{
    // 检查self维度是否大于维度8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    return true;
}

/**
 * param类型检查
 */
static aclnnStatus CheckParams(const aclTensor* self, const float std)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, std), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self维度是否满足要求
    CHECK_RET(CheckDimValid(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
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

    return concatTensor;
}

// DT_DOUBLE 路径：StatelessRandomNormalV2 + Mul(std) + Add(mean)
static const aclTensor* normalDoublePath(const aclTensor* selfContiguous, const aclTensor* stateLessOut, float mean,
                                         float std, aclOpExecutor* executor)
{
    CHECK_RET(stateLessOut != nullptr, nullptr);

    // 调用mul_算子kernel function(AI Core算子), 将float -> stdScalar -> stdTensor类型
    auto stdScalar2 = executor->AllocScalar(std);
    auto stdTensor = executor->ConvertToTensor(stdScalar2, selfContiguous->GetDataType());
    auto mulOut = l0op::Mul(stateLessOut, stdTensor, executor);
    CHECK_RET(mulOut != nullptr, nullptr);

    // 调用add_算子kernel function(AI Core算子)， 将float -> meanScalar -> meanTensor类型
    auto meanScalar2 = executor->AllocScalar(mean);
    auto meanTensor = executor->ConvertToTensor(meanScalar2, selfContiguous->GetDataType());
    return l0op::Add(mulOut, meanTensor, executor);
}

// 非DSA路径的 normal 正态分布计算
static const aclTensor* normalDavidPath(const aclTensor* selfContiguous, int64_t seed, int64_t offset, float mean,
                                        float std, aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfContiguous->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 Normal");
            // seed转化为key
            FVector<int64_t, op::MAX_DIM_NUM> key_vec = {seed};
            auto keyArr = executor->AllocIntArray(key_vec.data(), key_vec.size());

            // offset转化为counter
            FVector<int64_t, op::MAX_DIM_NUM> counter_vec = {0, offset};
            auto counterArr = executor->AllocIntArray(counter_vec.data(), counter_vec.size());
            FVector<float> meanVector = {mean};
            auto meanTensorV3 = executor->ConvertToTensor(meanVector.data(), meanVector.size(), op::DataType::DT_FLOAT);

            FVector<float> stdVector = {std};
            auto stdTensorV3 = executor->ConvertToTensor(stdVector.data(), stdVector.size(), op::DataType::DT_FLOAT);
            // 调用normal_算子kernel function(AI Core算子)
            return l0op::StatelessRandomNormalV3(selfContiguous, keyArr, counterArr, meanTensorV3, stdTensorV3,
                                                 executor);
        }
        // 精度等价：kernel 内 z * static_cast<float>(std) + static_cast<float>(mean) 在 fp32 下完成，
        // 最终 static_cast<T>(result) 仅一次 fp32→T 舍入，与原来 L2 层 Mul+Add+外层Cast 等价
        FVector<float> meanVector = {mean};
        auto meanTensor = executor->ConvertToTensor(meanVector.data(), meanVector.size(), op::DataType::DT_FLOAT);
        FVector<float> stdVector = {std};
        auto stdTensor = executor->ConvertToTensor(stdVector.data(), stdVector.size(), op::DataType::DT_FLOAT);

        return l0op::StatelessNormal(selfContiguous, seed, offset, meanTensor, stdTensor, executor);
    } else {
        // V2 路径：seed 转化为 key，offset 转化为 counter
        FVector<int64_t, op::MAX_DIM_NUM> key_vec = {seed};
        auto keyArr = executor->AllocIntArray(key_vec.data(), key_vec.size());

        FVector<int64_t, op::MAX_DIM_NUM> counter_vec = {0, offset};
        auto counterArr = executor->AllocIntArray(counter_vec.data(), counter_vec.size());

        // 调用normal_算子kernel function(AI Cpu算子)
        int64_t alg = 1;
        auto algScalar = executor->AllocScalar(&alg, DataType::DT_INT32);
        const aclTensor* algTensor = executor->ConvertToTensor(algScalar, op::ToOpDataType(ACL_INT32));
        auto stateLessOut = l0op::StatelessRandomNormalV2(selfContiguous, keyArr, counterArr, algTensor, executor);
        return normalDoublePath(selfContiguous, stateLessOut, mean, std, executor);
    }
}

// 非DSA路径的 normal 正态分布计算（Tensor seed/offset 版本）
static const aclTensor* normalTensorDavidPath(const aclTensor* selfContiguous, const aclTensor* seedTensor,
                                              const aclTensor* offsetTensor, int64_t offset, float mean, float std,
                                              aclOpExecutor* executor)
{
    if (seedTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Normal seedTensor dtype must be DT_INT64, got %s.",
                op::ToString(seedTensor->GetDataType()).GetString());
        return nullptr;
    }
    if (offsetTensor->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Normal offsetTensor dtype must be DT_INT64, got %s.",
                op::ToString(offsetTensor->GetDataType()).GetString());
        return nullptr;
    }

    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 &&
        selfContiguous->GetDataType() != DataType::DT_DOUBLE) {
        if (!aclnnGetPytorchRandom()) {
            OP_LOGD("compat mode, use V3 Normal");
            // seedTensor/offsetTensor 从 int64 Cast 为 uint64（V3 OpDef 要求 uint64）
            auto normalSeedU64 = l0op::Cast(seedTensor, op::DataType::DT_UINT64, executor);
            CHECK_RET(normalSeedU64 != nullptr, nullptr);
            auto normalOffsetU64 = l0op::Cast(offsetTensor, op::DataType::DT_UINT64, executor);
            CHECK_RET(normalOffsetU64 != nullptr, nullptr);

            FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
            aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
            auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
            auto resultAddOut = l0op::Add(normalOffsetU64, tmpTensor, executor);
            CHECK_RET(resultAddOut != nullptr, nullptr);
            FVector<float> meanVector = {mean};
            auto meanTensorV3 = executor->ConvertToTensor(meanVector.data(), meanVector.size(), op::DataType::DT_FLOAT);
            FVector<float> stdVector = {std};
            auto stdTensorV3 = executor->ConvertToTensor(stdVector.data(), stdVector.size(), op::DataType::DT_FLOAT);
            return l0op::StatelessRandomNormalV3(selfContiguous, normalSeedU64, resultAddOut, meanTensorV3, stdTensorV3,
                                                 executor);
        }
        // 精度等价论证同 normalDavidPath 改造
        FVector<int64_t> offsetVector{static_cast<int64_t>(offset)};
        auto tmpTensor = executor->ConvertToTensor(offsetVector.data(), offsetVector.size(), op::DataType::DT_INT64);
        auto resultAddOut = l0op::Add(offsetTensor, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);

        FVector<float> meanVector = {mean};
        auto meanTensor = executor->ConvertToTensor(meanVector.data(), meanVector.size(), op::DataType::DT_FLOAT);
        FVector<float> stdVector = {std};
        auto stdTensor = executor->ConvertToTensor(stdVector.data(), stdVector.size(), op::DataType::DT_FLOAT);

        return l0op::StatelessNormal(selfContiguous, seedTensor, resultAddOut, meanTensor, stdTensor, executor);
    } else {
        // V2 路径：保持原有 Cast/concat 逻辑
        auto normalSeedU64 = l0op::Cast(seedTensor, op::DataType::DT_UINT64, executor);
        CHECK_RET(normalSeedU64 != nullptr, nullptr);
        auto normalOffsetU64 = l0op::Cast(offsetTensor, op::DataType::DT_UINT64, executor);
        CHECK_RET(normalOffsetU64 != nullptr, nullptr);

        FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
        aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
        auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_UINT64);
        auto resultAddOut = l0op::Add(normalOffsetU64, tmpTensor, executor);
        CHECK_RET(resultAddOut != nullptr, nullptr);

        int64_t alg = 1;
        auto algScalar = executor->AllocScalar(&alg, DataType::DT_INT32);
        const aclTensor* algTensor = executor->ConvertToTensor(algScalar, op::ToOpDataType(ACL_INT32));
        auto stateLessOut = l0op::StatelessRandomNormalV2(selfContiguous, normalSeedU64, resultAddOut, algTensor,
                                                          executor);
        return normalDoublePath(selfContiguous, stateLessOut, mean, std, executor);
    }
}

/**
 * torch.Tensor.normal_()的API的总体调用流程
 */
aclnnStatus aclnnInplaceNormalGetWorkspaceSize(const aclTensor* selfRef, float mean, float std, int64_t seed,
                                               int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceNormal, DFX_IN(selfRef, mean, std, seed, offset), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, std);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // normal_ API的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (selfRef->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 非要求输出转换为FLOAT
    if (!CheckType(selfContiguous->GetDataType(), OUTPUT_SUPPORT_LIST)) {
        selfContiguous = l0op::Cast(selfContiguous, static_cast<op::DataType>(ACL_FLOAT), uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto inputShape = op::ToShapeVector(selfContiguous->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanScalar = uniqueExecutor.get()->AllocScalar(mean);
        CHECK_RET(meanScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto stdScalar = uniqueExecutor.get()->AllocScalar(std);
        CHECK_RET(stdScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::DSARandomNormal(inputShapeArray, seed, offset, meanScalar, stdScalar, uniqueExecutor.get());
    } else {
        computeOut = normalDavidPath(selfContiguous, seed, offset, mean, std, uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出self的数据类型
    auto castOut = l0op::Cast(computeOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出self上，self可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(const aclTensor* selfRef, float mean, float std,
                                                     const aclTensor* seedTensor, const aclTensor* offsetTensor,
                                                     int64_t offset, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceNormalTensor, DFX_IN(selfRef, mean, std, seedTensor, offsetTensor, offset),
                   DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, std);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // normal_ API的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (selfRef->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 非要求输出转换为FLOAT
    if (!CheckType(selfContiguous->GetDataType(), OUTPUT_SUPPORT_LIST)) {
        selfContiguous = l0op::Cast(selfContiguous, static_cast<op::DataType>(ACL_FLOAT), uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* computeOut = nullptr;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        auto inputShape = op::ToShapeVector(selfContiguous->GetViewShape());
        auto inputShapeArray = uniqueExecutor.get()->AllocIntArray(inputShape.data(), inputShape.size());
        CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanScalar = uniqueExecutor.get()->AllocScalar(mean);
        CHECK_RET(meanScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto stdScalar = uniqueExecutor.get()->AllocScalar(std);
        CHECK_RET(stdScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto concatTensor = ProcessOffsetTensor(offsetTensor, offset, uniqueExecutor.get());
        CHECK_RET(concatTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        computeOut = l0op::DSARandomNormalTensor(inputShapeArray, seedTensor, concatTensor, meanScalar, stdScalar,
                                                 uniqueExecutor.get());
    } else {
        computeOut = normalTensorDavidPath(selfContiguous, seedTensor, offsetTensor, offset, mean, std,
                                           uniqueExecutor.get());
    }
    CHECK_RET(computeOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出self的数据类型
    auto castOut = l0op::Cast(computeOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出self上，self可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceNormal(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceNormal);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceNormalTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceNormalTensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
