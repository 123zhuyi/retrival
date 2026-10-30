/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "aclnn_pdist_forward.h"
#include "pdist.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "../../../../conversion/fill/op_api/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16
};

static bool CheckNotNull(const aclTensor *self, const aclScalar *pScalar, const aclTensor *out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(pScalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out) {
    OP_CHECK_WRONG_DIMENSION(self, 2, return false);
    OP_CHECK_WRONG_DIMENSION(out, 1, return false);

    int64_t N = self->GetViewShape().GetDim(0);
    op::Shape expectOutShape = {N * (N - 1) / 2};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectOutShape, return false);
    return true;
}

static bool CheckPValid(const aclScalar *pScalar) {
    float pVal = pScalar->ToFloat();
    if (pVal < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pScalar only supports non-negative p values.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar *pScalar, const aclTensor *out) {
    // 检查参数是否为空指针
    CHECK_COND(CheckNotNull(self, pScalar, out), ACLNN_ERR_INNER_NULLPTR, "CheckNotNull failed!");

    // 检查输入的数据类型是否支持在API支持的数据类型范围之内， 需要根据api定义校验
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 检查数据排布是否满足要求
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    // 检查系数p是否符合要求
    CHECK_COND(CheckPValid(pScalar), ACLNN_ERR_PARAM_INVALID, "CheckPValid failed!");

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(int64_t shape, aclTensor *out, float val, aclOpExecutor *executor) {
    FVector<int64_t> tmp = {shape};
    auto dims = executor->ConvertToTensor(tmp.data(), tmp.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(tmp.data(), tmp.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static float CalculateValP(const aclScalar *pScalar) {
    float pVal = pScalar->ToFloat();
    return static_cast<float>(pVal);
}

aclnnStatus aclnnPdistForwardGetWorkspaceSize(const aclTensor* self, const aclScalar* pScalar,
                                              aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnPdistForward, DFX_IN(self, pScalar), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, pScalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 对于空tensor在kernal中的处理
    if (self->IsEmpty()) {
        int64_t row = self->GetViewShape().GetDim(0);
        int64_t shape = row * (row - 1) / 2;
        ret = FillScalar(shape, out, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 行向量只有一行，返回空tensor
    if (self->GetViewShape().GetDim(0) == 1) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入的self转换为连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Pdist算子kernel
    float pVal = CalculateValP(pScalar);
    auto pdistOut = l0op::Pdist(selfContiguous, pVal, uniqueExecutor.get());
    CHECK_RET(pdistOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果转化为out的数据类型
    auto castOut = l0op::Cast(pdistOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPdistForward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnPdistForward);

    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif

