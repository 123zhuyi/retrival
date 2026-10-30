/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sign_bits_pack.h"
#include "sign_bits_pack.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/framework_op.h"
#include "op_api/aclnn_check.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_NUM_1D = 1;
static const int64_t DIM_NUM_2D = 2;
static const int64_t PACK_SIZE = 8;

static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = { 
    op::DataType::DT_UINT8};

static bool CheckNotNullPtr(const aclTensor *self, aclTensor *out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckSocVersionIsSupport(void) {
  return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out)
{   
    bool isSupport = CheckSocVersionIsSupport();
    if(!isSupport) {   
        auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "SignBitsPack is unsupported by the current SOC version [%s]",
            op::ToString(socVersion).GetString());
        return false;
    }
    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *out)
{
    // 检查self的format是否为ND
    if (self->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self format should be ND. Actual: self is [%s].",
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    // 检查out的format是否为ND
    if (out->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out format should be ND. Actual: out is [%s].",
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support format [%s].", op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *out)
{
    size_t selfdimNum = self->GetViewShape().GetDimNum();
    if (selfdimNum != DIM_NUM_1D) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dims %zu should be 1.", selfdimNum);
        return false;
    }

    size_t outdimNum = out->GetViewShape().GetDimNum();
    if (outdimNum != DIM_NUM_2D) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dims %zu should be 2.", outdimNum);
        return false;
    }
    return true;
}

static bool CheckValue(const aclTensor *self, int64_t size, const aclTensor *out)
{
    for (size_t i = 0; i < out->GetViewShape().GetDimNum(); i++) {
        if (out->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of out is negative.");
            return false;
        }
    }

    size_t selfdim = self->GetViewShape().GetDim(0);
    auto ysize = (selfdim + 7) / 8;
    if(size <= 0)
    {   
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "size value must bigger zero.");
        return false;
    }

    if(size != 0 && ysize % size != 0){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "all must need be divisible by size");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t size, aclTensor *out)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNullPtr(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查参数值是否合法
    CHECK_RET(CheckValue(self, size, out), ACLNN_ERR_PARAM_INVALID);

    // 检查数据维度是否合法
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSignBitsPackGetWorkspaceSize(const aclTensor *self, int64_t size, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSignBitsPack, DFX_IN(self, size), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, size, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果是空tensor，直接返回
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *calcOut = nullptr;
    calcOut = l0op::SignBitsPack(selfContiguous, size, uniqueExecutor.get());
    CHECK_RET(calcOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(calcOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSignBitsPack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSignBitsPack);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
