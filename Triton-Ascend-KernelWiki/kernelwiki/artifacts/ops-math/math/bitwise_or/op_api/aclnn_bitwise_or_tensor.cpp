/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bitwise_or_tensor.h"
#include "math/logical_or/op_api/logical_or.h"
#include "bitwise_or.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
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

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_UINT8, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64,
  op::DataType::DT_BOOL};

static bool CheckShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // self、other、out的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

  // 对self和other的shape做广播，如果广播成功，则判断广播后的shape与out的shape是否相等
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);

  return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 检查self和other能否做数据类型推导
  op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
    return false;
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

// 两个tensor同时为空时返回true
static bool HasEmptyTensor(const aclTensor *self, const aclTensor *other) {
  if (self->IsEmpty() || other->IsEmpty()) {
    return true;
  }

  return false;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other) {
  // 检查self的数据类型是否在LogicalOr或者BitwiseOr算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查other的数据类型是否在LogicalOr或者BitwiseOr算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, DTYPE_SUPPORT_LIST, return false);

  return true;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // self、other、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和other是否能做数据类型推导，并且推导出的数据类型能否转换为out的数据类型
  CHECK_RET(CheckPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                          uint64_t* workspaceSize, aclOpExecutor** executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (HasEmptyTensor(self, other)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 对self和other两个输入做隐式数据类型转换
  auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型
  auto castSelf = l0op::Cast(contiguousSelf, promoteType, uniqueExecutor.get());
  CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的Tensor
  auto contiguousOther = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(contiguousOther != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型
  auto castOther = l0op::Cast(contiguousOther, promoteType, uniqueExecutor.get());
  CHECK_RET(castOther != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用LogicalOr或者BitwiseOr算子kernel
  const aclTensor *bitwiseOrOut = nullptr;
  if (promoteType == op::DataType::DT_BOOL) {
    bitwiseOrOut = l0op::LogicalOr(castSelf, castOther, uniqueExecutor.get());
  } else {
    bitwiseOrOut = l0op::BitwiseOr(castSelf, castOther, uniqueExecutor.get());
  }
  CHECK_RET(bitwiseOrOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(bitwiseOrOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBitwiseOrTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBitwiseOrTensor, DFX_IN(self, other), DFX_OUT(out));
  return GetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnBitwiseOrTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBitwiseOrTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBitwiseOrTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceBitwiseOrTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return GetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBitwiseOrTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceBitwiseOrTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
