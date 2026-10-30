/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_pow_tensor_tensor.h"
#include "pow.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Pow 算子的完整计算流程如下:
 * self                               exponent
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             Pow(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_INT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *exponent, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(exponent, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionIsSupportBf16(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *exponent) {
  if (!CheckSocVersionIsSupportBf16() &&
     (self->GetDataType() == op::DataType::DT_BF16 || exponent->GetDataType() == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of pow is not support bfloat16 in current socversion.");
    return false;
  }
  if ((self->GetDataType() == op::DataType::DT_BOOL) && (exponent->GetDataType() == op::DataType::DT_BOOL)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self and exponent dtype are bool is not supported.");
    return false;
  }

  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(exponent, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *exponent, const aclTensor *out) {
  // 检查self和exponent能否做数据类型推导
  op::DataType promoteType = op::PromoteType(self->GetDataType(), exponent->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and exponent dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(exponent->GetDataType()).GetString());
    return false;
  }

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *exponent, const aclTensor *out) {

  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(exponent, MAX_DIM_LEN, return false);

  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, exponent, broadcastShape, return false);

  if (broadcastShape != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static void CheckFormat(const aclTensor* self, const aclTensor* exponent) {
    ge::Format selfStorageFormat = self->GetStorageFormat();
    ge::Format exponentStorageFormat = exponent->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ || 
        exponentStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnPowTensorTensor doesn't support format NZ.");
    }
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *exponent, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, exponent, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, exponent), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和exponent能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, exponent, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查双输入是否能broadcast
  CHECK_RET(CheckShape(self, exponent, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPowTensorTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *exponent, aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnPowTensorTensor, DFX_IN(self, exponent), DFX_OUT(out));

// 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, exponent, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 检查格式
  CheckFormat(self, exponent);

  // pow算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty() || exponent->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // Pow算子需要对self和exponent两个输入做隐式数据类型转换，根据具体算子语义按需调用
  auto promoteType = op::PromoteType(self->GetDataType(), exponent->GetDataType());

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入exponent转换成连续的tensor
  auto exponentContiguous = l0op::Contiguous(exponent, uniqueExecutor.get());
  CHECK_RET(exponentContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入exponent的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto exponentCasted = l0op::Cast(exponentContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(exponentCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Pow算子kernel
  auto powOpOut = l0op::Pow(selfCasted, exponentCasted, uniqueExecutor.get());
  CHECK_RET(powOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(powOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplacePowTensorTensorGetWorkspaceSize(const aclTensor *self,
                                                        const aclTensor *exponent,
                                                        uint64_t *workspaceSize,
                                                        aclOpExecutor **executor) {
  auto out = const_cast<aclTensor*>(self);
  return aclnnPowTensorTensorGetWorkspaceSize(self, exponent, out, workspaceSize, executor);
}

aclnnStatus aclnnInplacePowTensorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplacePowTensorTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnPowTensorTensor(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnPowTensorTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif