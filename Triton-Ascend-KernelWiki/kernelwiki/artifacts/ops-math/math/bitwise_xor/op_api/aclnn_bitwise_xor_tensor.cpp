/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bitwise_xor_tensor.h"
#include "bitwise_xor.h"
#include "math/not_equal/op_api/not_equal.h"
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
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_DEFAULT = {
  DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BOOL};

static const std::initializer_list<DataType> REGBASE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BOOL,
  DataType::DT_UINT16, DataType::DT_UINT32, DataType::DT_UINT64};

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // self、other、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *self, const aclTensor *other, const aclTensor *out,
                                   DataType &promoteType) {
  auto supportList = IsRegBase() ? REGBASE_DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_DEFAULT;
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查other的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

  // 检查self和other能否做数据类型推导
  promoteType = PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            ToString(self->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
    return false;
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // self、other、out的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

  // 对self和other的shape做广播，如果广播成功，则判断广播后的shape与out的shape是否相等
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, const aclTensor *out,
                               DataType &promoteType) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型是否符合要求
  CHECK_RET(CheckDtypeValid(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否符合要求
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  DataType promoteType = DataType::DT_UNDEFINED;
  auto ret = CheckParams(self, other, out, promoteType);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty() || other->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的Tensor
  auto contiguousOther = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(contiguousOther != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用NotEqual或者BitwiseXor算子kernel
  const aclTensor *bitwiseXorTensorOut = nullptr;
  if (promoteType == DataType::DT_BOOL) {
    bitwiseXorTensorOut = l0op::NotEqual(contiguousSelf, contiguousOther, uniqueExecutor.get());
  } else {
    // 将输入self的数据类型转换成隐式数据类型
    auto castSelf = l0op::Cast(contiguousSelf, promoteType, uniqueExecutor.get());
    CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型
    auto castOther = l0op::Cast(contiguousOther, promoteType, uniqueExecutor.get());
    CHECK_RET(castOther != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bitwiseXorTensorOut = l0op::BitwiseXor(castSelf, castOther, uniqueExecutor.get());
  }
  CHECK_RET(bitwiseXorTensorOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(bitwiseXorTensorOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBitwiseXorTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBitwiseXorTensor, DFX_IN(self, other), DFX_OUT(out));
  return GetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnBitwiseXorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBitwiseXorTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBitwiseXorTensorGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceBitwiseXorTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return GetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBitwiseXorTensor(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceBitwiseXorTensor);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
