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
 * \file aclnn_dropout_gen_mask.cpp
 * \brief
 */

#include "aclnn_dropout_gen_mask.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "random/drop_out_do_mask/op_api/dropout_do_mask.h"
#include "dsa_gen_bit_mask.h"
#include "conversion/fill/op_api/fill.h"
#include "random/stateless_drop_out_gen_mask/op_api/stateless_dropout_gen_mask.h"
#include "math/zero_op/op_api/zero_op.h"
#include "math/add/op_api/add.h"
#include "../../../../conversion/concat_d/op_api/concat_d.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t VEC_BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;
static const int64_t FLOAT_BIT_NUMBER = 32;
static const int8_t MAX_MASK_NUM = -1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclIntArray* input, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* out)
{
    // 检查mask的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, MASK_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckProbability(double prob)
{
    if (prob > 1 || prob < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of prob has to be between 0 and 1, but current is %f.", prob);
        return false;
    }
    return true;
}

static inline bool CheckSocVersionIsSupportDSA(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static inline int64_t InferDSAOutShape(const aclIntArray* shape)
{
    int64_t size = 1;
    for (size_t index = 0; index < shape->Size(); index++) {
        size *= (*shape)[index];
    }
    return (size + VEC_BIT_NUMBER - 1) / VEC_BIT_NUMBER * VEC_BIT_NUMBER / FLOAT_BIT_NUMBER;
}

static inline int64_t InferDSAOutShapeV2(const aclIntArray* shape)
{
    int64_t size = 1;
    for (size_t index = 0; index < shape->Size(); index++) {
        size *= (*shape)[index];
    }
    return (size + VEC_BIT_NUMBER - 1) / VEC_BIT_NUMBER * VEC_BIT_NUMBER / UINT8_BIT_NUMBER;
}

static inline bool CheckShape(const aclIntArray* shape, const aclTensor* out)
{
    int64_t needMaskSize = InferDSAOutShapeV2(shape);
    int64_t maskSize = out->GetViewShape().GetShapeSize();
    if (maskSize != needMaskSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of out has to be %ld, but current is %ld.", needMaskSize, maskSize);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclIntArray* shape, double prob, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(shape, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查p是否符合规则
    CHECK_RET(CheckProbability(prob), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出输出shape
    CHECK_RET(CheckShape(shape, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline const aclTensor* FillScalar(const aclTensor* out, int8_t val, aclOpExecutor* executor)
{
    auto maskShape = out->GetViewShape();
    FVector<int64_t> maskShapeVector;
    for (size_t i = 0; i < maskShape.GetDimNum(); i++) {
        maskShapeVector.push_back(maskShape.GetDim(i));
    }
    auto dims = executor->ConvertToTensor(maskShapeVector.data(), maskShapeVector.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(maskShapeVector.data(), maskShapeVector.size());

    FVector<int8_t> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), op::DataType::DT_INT8);
    auto mask = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(mask != nullptr, nullptr);
    mask->SetFromWorkspace(false);
    mask->SetStorageAddr(out->GetStorageAddr());
    return out;
}

static aclScalar* CreateDropout(float prob, op::DataType dtype, aclOpExecutor* executor)
{
    op::fp16_t ratioF16;
    op::bfloat16 ratioBf16;
    OP_LOGI("prob %f dtype %d", prob, dtype);
    switch (dtype) {
        case op::DataType::DT_FLOAT16:
            ratioF16 = prob;
            return executor->AllocScalar(&ratioF16.val, op::DataType::DT_FLOAT16);
        case op::DataType::DT_BF16:
            ratioBf16 = prob;
            return executor->AllocScalar(&ratioBf16.value, op::DataType::DT_BF16);
        case op::DataType::DT_FLOAT:
            return executor->AllocScalar(prob);
        default:
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Scalar p not implemented for %s, should be in dtype support list %s.",
                op::ToString(dtype).GetString(), op::ToString(P_DTYPE_SUPPORT_LIST).GetString());
            return nullptr;
    }
}

static const aclTensor* ComputeStatelessMask(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclOpExecutor* executor)
{
    FVector<int64_t> seedVector = {seed};
    aclIntArray* seedList = executor->AllocIntArray(seedVector.data(), seedVector.size());
    auto seedTensor = executor->ConvertToTensor(seedList, op::DataType::DT_INT64);

    FVector<int64_t> seed1Vector = {0};
    aclIntArray* seed1List = executor->AllocIntArray(seed1Vector.data(), seed1Vector.size());
    auto seed1Tensor = executor->ConvertToTensor(seed1List, op::DataType::DT_INT64);

    FVector<int64_t> offsetVector = {0, offset};
    aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), offsetVector.size());
    auto offsetTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_INT64);

    auto probTensor =
        executor->ConvertToTensor(executor->AllocScalar(static_cast<float>(1 - prob)), op::DataType::DT_FLOAT);

    return l0op::StatelessDropoutGenMask(shape, probTensor, seedTensor, seed1Tensor, offsetTensor, executor);
}

static const aclTensor* ComputeMask(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclTensor* out, aclOpExecutor* executor)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        op::DataType dtype = out->GetDataType();
        if (dtype == op::DataType::DT_UINT8) {
            dtype = op::DataType::DT_FLOAT;
        }
        auto dropout = CreateDropout(static_cast<float>(prob), dtype, executor);
        CHECK_RET(dropout != nullptr, nullptr);

        int64_t shapeSize = InferDSAOutShape(shape);
        auto outTensorTemp = executor->AllocTensor(op::Shape{shapeSize}, dtype);
        CHECK_RET(outTensorTemp != nullptr, nullptr);
        outTensorTemp->SetFromWorkspace(false);
        outTensorTemp->SetStorageAddr(out->GetStorageAddr());
        l0op::DSAGenBitMask(shapeSize * FLOAT_BIT_NUMBER, seed, offset, dropout, outTensorTemp, executor);
        return out;
    } else {
        return ComputeStatelessMask(shape, prob, seed, offset, executor);
    }
}

static const aclTensor* ComputeMaskV2(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, op::DataType dtype, aclOpExecutor* executor)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201) {
        int64_t shapeSize = InferDSAOutShapeV2(shape);
        auto dropout = CreateDropout(static_cast<float>(prob), dtype, executor);
        CHECK_RET(dropout != nullptr, nullptr);
        return l0op::DSAGenBitMask(shapeSize * UINT8_BIT_NUMBER, seed, offset, dropout, executor);
    } else {
        return ComputeStatelessMask(shape, prob, seed, offset, executor);
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

    return concatTensor;
}

static const aclTensor* ComputeMaskV2Tensor(
    const aclIntArray* shape, double prob, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    op::DataType dtype, aclOpExecutor* executor)
{
    if (IsRegBase()) {
        FVector<int64_t> seed1Vector = {0};
        aclIntArray* seed1List = executor->AllocIntArray(seed1Vector.data(), seed1Vector.size());
        auto seed1Tensor = executor->ConvertToTensor(seed1List, op::DataType::DT_INT64);

        FVector<int64_t> offsetVector{0, static_cast<int64_t>(offset)};
        aclIntArray* offsetList = executor->AllocIntArray(offsetVector.data(), 2);
        auto tmpTensor = executor->ConvertToTensor(offsetList, op::DataType::DT_INT64);
        auto offset1Tensor = l0op::Add(offsetTensor, tmpTensor, executor);
        CHECK_RET(offset1Tensor != nullptr, nullptr);

        auto probTensor = executor->ConvertToTensor(executor->AllocScalar(static_cast<float>(1 - prob)), op::DataType::DT_FLOAT);

        return l0op::StatelessDropoutGenMask(shape, probTensor, seedTensor, seed1Tensor, offset1Tensor, executor);
    } else {
        int64_t shapeSize = InferDSAOutShapeV2(shape);
        auto dropout = CreateDropout(static_cast<float>(prob), dtype, executor);
        CHECK_RET(dropout != nullptr, nullptr);
        auto concatTensor = ProcessOffsetTensor(offsetTensor, offset, executor);
        CHECK_RET(concatTensor != nullptr, nullptr);
        return l0op::DSAGenBitMaskTensor(shapeSize * UINT8_BIT_NUMBER, seedTensor, concatTensor, dropout, executor);
    }
}

static bool IsDoubleEqual(double f1, double f2)
{
    return std::abs(f1 - f2) <= std::numeric_limits<double>::epsilon();
}

static inline const aclTensor* DoMask(
    const aclTensor* inputContiguous, const aclTensor* mask, double prob, aclOpExecutor* executor)
{
    if (IsDoubleEqual(prob, 0)) {
        return inputContiguous;
    } else if (IsDoubleEqual(prob, 1)) {
        return l0op::ZerosLike(inputContiguous, executor);
    } else {
        FVector<float> probVector = {static_cast<float>(1 - prob)};
        auto probTensor =
            executor->ConvertToTensor(probVector.data(), probVector.size(), inputContiguous->GetDataType());
        return l0op::DropoutDoMask(inputContiguous, mask, probTensor, executor);
    }
}

aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDropoutGenMask, DFX_IN(shape, prob, seed, offset), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(shape, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* mask;
    if (IsDoubleEqual(prob, 0)) {
        mask = FillScalar(out, MAX_MASK_NUM, uniqueExecutor.get());
    } else if (IsDoubleEqual(prob, 1)) {
        mask = FillScalar(out, 0, uniqueExecutor.get());
    } else {
        mask = ComputeMask(shape, prob, seed, offset, out, uniqueExecutor.get());
    }
    CHECK_RET(mask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskViewCopyResult = l0op::ViewCopy(mask, out, uniqueExecutor.get());
    CHECK_RET(maskViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutGenMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutGenMask);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(
    const aclIntArray* shape, double prob, int64_t seed, int64_t offset, aclDataType probDataType, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDropoutGenMaskV2, DFX_IN(shape, prob, seed, offset, probDataType), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(shape, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* mask;
    if (IsDoubleEqual(prob, 0)) {
        mask = FillScalar(out, MAX_MASK_NUM, uniqueExecutor.get());
    } else if (IsDoubleEqual(prob, 1)) {
        mask = FillScalar(out, 0, uniqueExecutor.get());
    } else {
        op::DataType dtype = ToOpDataType(probDataType);
        mask = ComputeMaskV2(shape, prob, seed, offset, dtype, uniqueExecutor.get());
    }
    CHECK_RET(mask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(mask, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutGenMaskV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutGenMaskV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(
    const aclIntArray* shape, double prob, const aclTensor* seedTensor, const aclTensor* offsetTensor, int64_t offset,
    aclDataType probDataType, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnDropoutGenMaskV2Tensor, DFX_IN(shape, prob, seedTensor, offsetTensor, offset, probDataType), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(shape, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* mask;
    if (IsDoubleEqual(prob, 0)) {
        mask = FillScalar(out, MAX_MASK_NUM, uniqueExecutor.get());
    } else if (IsDoubleEqual(prob, 1)) {
        mask = FillScalar(out, 0, uniqueExecutor.get());
    } else {
        op::DataType dtype = ToOpDataType(probDataType);
        mask = ComputeMaskV2Tensor(shape, prob, seedTensor, offsetTensor, offset, dtype, uniqueExecutor.get());
    }
    CHECK_RET(mask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(mask, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutGenMaskV2Tensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutGenMaskV2Tensor);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
