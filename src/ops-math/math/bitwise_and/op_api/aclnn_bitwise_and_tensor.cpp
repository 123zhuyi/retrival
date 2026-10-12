/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bitwise_and_tensor.h"
#include "math/logical_and/op_api/logical_and.h"
#include "bitwiseand.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

constexpr int BITWISE_AND_MAX_TENSOR_DIM = 8;

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* BitwiseAndTensor算子的完整计算流程如下:
 * self.dtype == bool
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             LogicalAnd(workspace_4)
 *                    |
 *              Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 * self.dtype == INT
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             BitwiseAnd(workspace_4)
 *                    |
 *              Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_BOOL, op::DataType::DT_UINT16};

static bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other) {
  // 检查self的数据类型是否在and算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查other的数据类型是否在and算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 检查self和other能否做数据类型推导
  op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
    return false;
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

static bool CheckFormat(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 如果输入格式是私有格式，记录日志，直接报错
  if (op::IsPrivateFormat(self->GetStorageFormat()) ||
      op::IsPrivateFormat(other->GetStorageFormat()) ||
      op::IsPrivateFormat(out->GetStorageFormat())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND,NCHW,NHWC,HWCN,NDHWC,NCDHW.");
    return false;
  }
  return true;
}

static bool CheckShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, BITWISE_AND_MAX_TENSOR_DIM, return false);
  OP_CHECK_MAX_DIM(other, BITWISE_AND_MAX_TENSOR_DIM, return false);

  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查数据格式是否支持
  CHECK_RET(CheckFormat(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 5. 检查双输入是否能broadcast
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBitwiseAndTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBitwiseAndTensor, DFX_IN(self, other), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // BitwiseAnd算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty() || other->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // BitwiseAnd算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
  auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行BitwiseAnd计算
  const aclTensor *andOpOut = nullptr;
  if (promoteType == op::DataType::DT_BOOL) {
	andOpOut = l0op::LogicalAnd(selfCasted, otherCasted, uniqueExecutor.get());
  } else {
	andOpOut = l0op::BitwiseAnd(selfCasted, otherCasted, uniqueExecutor.get());
  }
  CHECK_RET(andOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(andOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceBitwiseAndTensorGetWorkspaceSize(const aclTensor *selfRef, const aclTensor *other,
    uint64_t *workspaceSize, aclOpExecutor **executor) {
  auto out = const_cast<aclTensor *>(selfRef);
  return aclnnBitwiseAndTensorGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnBitwiseAndTensor(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBitwiseAndTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBitwiseAndTensor(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceBitwiseAndTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
