/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_logaddexp2.h"
#include "logaddexp.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const float LOG_BASE = 2.0f;
const float LOG_SCALE = 1.0f;
const float LOG_SHIFT = 0.0f;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST_910 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, 
    op::DataType::DT_UINT16, op::DataType::DT_INT32, op::DataType::DT_UINT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16, 
    op::DataType::DT_UINT16, op::DataType::DT_INT32, op::DataType::DT_UINT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_INT16,
    op::DataType::DT_UINT16, op::DataType::DT_INT32, op::DataType::DT_UINT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static const std::initializer_list<DataType>& GetInputDtypeSupportList() {
  if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
    return INPUT_DTYPE_SUPPORT_LIST_910B;
  } else {
    return INPUT_DTYPE_SUPPORT_LIST_910;
  }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other, const aclTensor *out) {
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND950) {
        // 检查self和other的数据类型是否为支持的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(self, INPUT_DTYPE_SUPPORT_LIST_950, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(other, INPUT_DTYPE_SUPPORT_LIST_950, return false);
        // 检查out的数据类型是否为支持的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(out, ASCEND950_DTYPE_SUPPORT_LIST, return false);
    } else {
        auto outDtypeSupportList = GetDtypeSupportList();
        auto inputDtypeSupportList = GetInputDtypeSupportList();

        // 检查self和other的数据类型是否为支持的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(self, inputDtypeSupportList, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(other, inputDtypeSupportList, return false);

        // 检查out的数据类型是否为支持的数据类型
        OP_CHECK_DTYPE_NOT_SUPPORT(out, outDtypeSupportList, return false);
    }

    if (self->GetDataType() != out->GetDataType()) {
        // 检查self的数据类型能否转换为输出的数据类型
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);
    }

    if (other->GetDataType() != out->GetDataType()) {
        // 检查other的数据类型能否转换为输出的数据类型
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), out->GetDataType(), return false);
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
    // 检查self和other的维度是否大于8
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查self和other是否能broadcast，如可以求broadcast之后的shape
    OP_CHECK_BROADCAST(self, other, return false);
    op::Shape broadcastShape;
    BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);

    // 检查self和other经过broadcast后的shape是否与out的shape一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of self and output should be equal. self [%s], out [%s].",
            ToString(self->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (other->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of other and output should be equal. other [%s], out [%s].",
            ToString(other->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的shape
    CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

    if (IsRegBase()) {
        CHECK_RET(CheckFormat(self, other, out), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static void WarnFormat(const aclTensor* self, const aclTensor* target){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  ge::Format targetStorageFormat = target->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND || targetStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnLogAddExp2 only support format ND.");
  }
}

aclnnStatus aclnnLogAddExp2GetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                            uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    
    L2_DFX_PHASE_1(aclnnLogAddExp2, DFX_IN(self, other), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (!IsRegBase()) {
        WarnFormat(self, other);
    }

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || other->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入self转换成连续的tensor，并将其数据类型转换成out的数据类型
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfCasted = l0op::Cast(selfContiguous, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other转换成连续的tensor，并将其数据类型转换成out的数据类型
    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherCasted = l0op::Cast(otherContiguous, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 按计算图执行具体LogAddExp2计算
    auto finalOut = l0op::LogAddExp(selfCasted, otherCasted, LOG_BASE, LOG_SCALE, LOG_SHIFT, uniqueExecutor.get());
    CHECK_RET(finalOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(finalOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogAddExp2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnLogAddExp2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
