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
 * \file stateless_bernoulli.cpp
 * \brief
 */

#include "aclnn_bernoulli.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "random/stateless_bernoulli/op_api/stateless_bernoulli.h"
#include "random/drop_out_do_mask/op_api/dropout_do_mask.h"
#include "dsa_gen_bit_mask.h"
#include "math/zero_op/op_api/zero_op.h"
#include "math/ones_like/op_api/ones_like.h"
#include "conversion/fill/op_api/fill.h"
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

static bool IsDoubleEqual(double f1, double f2)
{
    return std::abs(f1 - f2) <= std::numeric_limits<double>::epsilon();
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
    if (!CheckType(self->GetDataType(), ASCEND910B_PROB_DTYPE_SUPPORT_LIST)) {
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
    // 检查y的数据类型是否在支持列表内
    if (prob->ToDouble() > 1 || prob->ToDouble() < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "prob should be in range 0<=prob<=1 .");
        return false;
    }

    return true;
}

static bool CheckFormat(const aclTensor* self)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s]",
            ToString(self->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckFormatTensor(const aclTensor* self, const aclTensor* prob)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(prob->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW, self [%s], prob [%s]",
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

static const aclTensor* CreateTensor(float val, op::DataType dtype, aclOpExecutor* executor)
{
    FVector<float> tmpVector = {static_cast<float>(val)};
    OP_LOGI("val %f dtype %d", val, dtype);
    switch (dtype) {
        case op::DataType::DT_BF16:
            return executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), op::DataType::DT_BF16);
        case op::DataType::DT_FLOAT:
            return executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), op::DataType::DT_FLOAT);
        default:
            return executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), op::DataType::DT_FLOAT16);
    }
}

static inline int64_t InferDSAOutShapeV2(const aclIntArray* shape)
{
    int64_t size = 1;
    for (size_t index = 0; index < shape->Size(); index++) {
        size *= (*shape)[index];
    }
    return (size + VEC_BIT_NUMBER - 1) / VEC_BIT_NUMBER * VEC_BIT_NUMBER / UINT8_BIT_NUMBER;
}

aclnnStatus GetBernoulliByDSA(
    const aclTensor* inputContiguous, const aclScalar* prob, int64_t seed, int64_t offset, const aclTensor*& doMaskOut,
    aclOpExecutor* executor)
{
    auto inputShape = op::ToShapeVector(inputContiguous->GetViewShape());
    auto dims = executor->ConvertToTensor(inputShape.data(), inputShape.size(), DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputSizeArray = executor->AllocIntArray(inputShape.data(), inputShape.size());
    CHECK_RET(inputSizeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t shapeSize = InferDSAOutShapeV2(inputSizeArray);
    auto probScalar = executor->AllocScalar(static_cast<float>(1 - prob->ToDouble()));
    CHECK_RET(probScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto mask = l0op::DSAGenBitMask(shapeSize * UINT8_BIT_NUMBER, seed, offset, probScalar, executor);
    CHECK_RET(mask != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto oneTensor = CreateTensor(1, inputContiguous->GetDataType(), executor);
    CHECK_RET(oneTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 根据shape、dtype生成全1数组
    auto inputOnes = l0op::Fill(dims, oneTensor, inputSizeArray, executor);
    CHECK_RET(inputOnes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 仅支持bf16、fp16和fp32，prob = 1从而不进行缩放
    doMaskOut = l0op::DropoutDoMask(inputOnes, mask, oneTensor, executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBernoulliTensor, DFX_IN(self, prob, seed, offset), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParamsTensor(self, prob, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

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

aclnnStatus aclnnBernoulliGetWorkspaceSize(
    const aclTensor* self, const aclScalar* prob, int64_t seed, int64_t offset, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBernoulli, DFX_IN(self, prob, seed, offset), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, prob, out);
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

    auto inputContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const aclTensor* opOut = nullptr;
    if (curArch == NpuArch::DAV_2201) {
        // 调用DSAGenBitMask算子kernel
        const aclTensor* doMaskOut = nullptr;
        if (IsDoubleEqual(prob->ToDouble(), 0)) {
            doMaskOut = l0op::ZerosLike(inputContiguous, uniqueExecutor.get());
        } else if (IsDoubleEqual(prob->ToDouble(), 1)) {
            doMaskOut = l0op::OnesLike(inputContiguous, uniqueExecutor.get());
        } else {
            auto executeResult =
                GetBernoulliByDSA(inputContiguous, prob, seed, offset, doMaskOut, uniqueExecutor.get());
            CHECK_RET(executeResult == ACLNN_SUCCESS, executeResult);
        }
        CHECK_RET(doMaskOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        opOut = l0op::Cast(doMaskOut, out->GetDataType(), uniqueExecutor.get());
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

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(opOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(
    const aclTensor* selfRef, const aclScalar* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnBernoulliGetWorkspaceSize(selfRef, prob, seed, offset, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(
    const aclTensor* selfRef, const aclTensor* prob, int64_t seed, int64_t offset, uint64_t* workspaceSize,
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

aclnnStatus aclnnInplaceBernoulliTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
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
