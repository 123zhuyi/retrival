/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sinc.h"
#include "sinc.h"
#include "math/ones_like/op_api/ones_like.h"
#include "math/equal/op_api/equal.h"
#include "math/sin/op_api/sin.h"
#include "math/select/op_api/select.h"
#include "math/mul/op_api/mul.h"
#include "math/real_div/op_api/realdiv.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr double PI = 3.14159265358979323846;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64,  op::DataType::DT_BOOL,
    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_OUT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                   op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SINC_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在sin算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_OUT_LIST, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
                op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

static bool CheckInplaceDtypeValid(const aclTensor* self)
{
    // 检查inplace下self的数据类型是否在out的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_OUT_LIST, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && self->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s is unsupported by the current SOC version [%s].",
                op::ToString(self->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    // self的维度必须小于 9
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_INNER_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(const aclTensor* self)
{
    CHECK_RET(CheckInplaceDtypeValid(self), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSincGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Sinc算子的空tensor在kernel中不支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        OP_LOGD("empty input tensor");
        // 根据实际支持情况补充
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto castDtype = selfContiguous->GetDataType();

    const aclTensor* sincOut = nullptr;
    if (IsRegBase() && CheckType(castDtype, DTYPE_SINC_SUPPORT_LIST)) {
        sincOut = l0op::Sinc(selfContiguous, uniqueExecutor.get());
        CHECK_RET(sincOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        if (!CheckType(castDtype, DTYPE_OUT_LIST)) {
            castDtype = op::DataType::DT_FLOAT;
        }
        // 调用cast算子将不支持的类型转化为float
        auto selfCast = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
        CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto zerosTensor = l0op::ZerosLike(selfCast, uniqueExecutor.get());
        CHECK_RET(zerosTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto equalResult = l0op::Equal(selfCast, zerosTensor, uniqueExecutor.get());
        CHECK_RET(equalResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto onesTensor = l0op::OnesLike(selfCast, uniqueExecutor.get());
        CHECK_RET(onesTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto selectOut = l0op::SelectV2(equalResult, onesTensor, selfCast, uniqueExecutor.get());
        CHECK_RET(selectOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto piTensor = uniqueExecutor.get()->ConvertToTensor(&PI, 1, selectOut->GetDataType());
        auto mulOut = l0op::Mul(selectOut, piTensor, uniqueExecutor.get());
        CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto sinOut = l0op::Sin(mulOut, uniqueExecutor.get());
        CHECK_RET(sinOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto divOut = l0op::RealDiv(sinOut, mulOut, uniqueExecutor.get());
        CHECK_RET(divOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        sincOut = l0op::SelectV2(equalResult, onesTensor, divOut, uniqueExecutor.get());
        CHECK_RET(sincOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(sincOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSincGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSinc, DFX_IN(self), DFX_OUT(out));
    return ExecSincGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceSincGetWorkspaceSize(aclTensor* selfRef, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceSinc, DFX_IN(selfRef), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckInplaceParams(selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecSincGetWorkspaceSize(selfRef, out, workspaceSize, executor);
}

aclnnStatus aclnnSinc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSinc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceSinc(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnInplaceSinc);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
