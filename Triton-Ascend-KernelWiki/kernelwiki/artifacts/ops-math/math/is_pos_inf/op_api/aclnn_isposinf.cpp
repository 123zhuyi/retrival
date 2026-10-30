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
 * \file aclnn_isposinf.cpp
 * \brief
 */

#include "aclnn_isposinf.h"
#include "aclnn_kernels/contiguous.h"
#include "isposinf.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_BOOL,  op::DataType::DT_INT32,   op::DataType::DT_INT64,
    op::DataType::DT_INT16, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_INT16,   op::DataType::DT_INT8, op::DataType::DT_UINT8};

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto curSocVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (curSocVersion >= SocVersion::ASCEND910B && curSocVersion <= SocVersion::ASCEND910E) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    } else if (curSocVersion == SocVersion::ASCEND310P) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, ASCEND310P_DTYPE_SUPPORT_LIST, return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnIsPosInf is not supported on this device.");
        return false;
    }
    // out的dtype必须为BOOL
    OP_CHECK_DTYPE_NOT_MATCH(out, op::DataType::DT_BOOL, return false);
    OP_LOGD("Data type check successful");
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查self和out的维度匹配关系
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查IsPosInf计算的输出dtype能否cast为out的dtype
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static FVector<int64_t> GetTmpDim(const aclTensor* out)
{
    FVector<int64_t> tmp;
    if (out->GetViewShape().GetDimNum() != 0) {
        size_t dimNum = out->GetViewShape().GetDimNum();
        for (size_t idx = 0; idx < dimNum; idx++) {
            int64_t tmpVal = out->GetViewShape().GetDim(idx);
            tmp.push_back(tmpVal);
        }
    } else {
        tmp.push_back(1);
    }
    return tmp;
}

static const aclTensor* FillTensor(aclTensor* out, bool val, aclOpExecutor* executor)
{
    FVector<int64_t> tmp = GetTmpDim(out);
    const aclTensor* dims = executor->ConvertToTensor(tmp.data(), tmp.size(), op::ToOpDataType(ACL_INT64));
    aclIntArray* shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());
    FVector<bool> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    if (dims == nullptr || shapeArray == nullptr || valTensor == nullptr) {
        return nullptr;
    }
    return l0op::Fill(dims, valTensor, shapeArray, executor);
}

aclnnStatus aclnnIsPosInfGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnIsPosInf, DFX_IN(self), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非empty，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* IsPosInfResult = nullptr;
    // self的dtype为整型，必定有界，out值全为false，调用fill给out赋值
    if (!IsFloatingType(self->GetDataType()) && !IsComplexType(self->GetDataType())) {
        OP_LOGD("Input data is integer, all output are false");
        IsPosInfResult = FillTensor(out, false, uniqueExecutor.get());
        CHECK_RET(IsPosInfResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // self为浮点型，调用l0算子IsPosInf进行计算,输出dtype与self一致
    if (IsFloatingType(self->GetDataType())) {
        OP_LOGD("Input data is float point");
        IsPosInfResult = l0op::IsPosInf(selfContiguous, uniqueExecutor.get());
        CHECK_RET(IsPosInfResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(IsPosInfResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIsPosInf(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIsPosInf);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
