/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_bitwise_xor_scalar.h"
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
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

static op::DataType CombineCategories(const op::DataType higher, const op::DataType lower) {
  if (higher == op::DataType::DT_BOOL) {
    return op::PromoteType(higher, lower);
  }
  if (higher != op::DataType::DT_UNDEFINED) {
    return higher;
  }
  return lower;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_DEFAULT = {
  DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BOOL};

static const std::initializer_list<DataType> REGBASE_DTYPE_SUPPORT_LIST = {
  DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT32, DataType::DT_INT64, DataType::DT_UINT8, DataType::DT_BOOL,
  DataType::DT_UINT16, DataType::DT_UINT32, DataType::DT_UINT64};

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  return true;
}

static inline bool CheckNotNull(const aclTensor *self, const aclScalar *other, const aclTensor *out) {
  // self、other、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static DataType GetPromoteDType(const aclTensor *self, const aclScalar *other) {
  auto promoteType = DataType::DT_UNDEFINED;
  if (IsRegBase()) {
    promoteType = CombineCategories(self->GetDataType(), other->GetDataType());
    return promoteType;
  }

  if (other->GetDataType() == DataType::DT_INT64) {
    int64_t v = other->ToInt64();
    if (v >= INT32_MIN && v <= INT32_MAX) {
      promoteType = PromoteType(self->GetDataType(), DataType::DT_INT32);
    } else {
      promoteType = PromoteType(self->GetDataType(), DataType::DT_INT64);
    }
  } else {
    promoteType = PromoteType(self->GetDataType(), other->GetDataType());
  }
  return promoteType;
}

static inline bool CheckDtypeValid(const aclTensor *self, const aclScalar *other, const aclTensor *out,
                                   DataType &promoteType) {
  auto supportList = IsRegBase() ? REGBASE_DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST_DEFAULT;
  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查other的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

  // 检查self和other能否做数据类型推导
  promoteType = GetPromoteDType(self, other);
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dtype %s and other dtype %s can not promote dtype.",
            ToString(self->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
    return false;
  }

  // 检查推导后的数据类型能否转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclScalar *other, const aclTensor *out,
                               DataType &promoteType) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查数据类型是否正确
  CHECK_RET(CheckDtypeValid(self, other, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  DataType promoteType = DataType::DT_UNDEFINED;
  auto ret = CheckParams(self, other, out, promoteType);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果self是空tensor，则out也是空tensor，直接返回
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other转换为数据类型为promoteType的tensor
  const aclTensor *otherTensor = (uniqueExecutor.get())->ConvertToTensor(other, promoteType);
  CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用NotEqual或者BitwiseXor算子kernel
  const aclTensor *bitwiseXorScalarOut = nullptr;
  if (promoteType == DataType::DT_BOOL) {
    bitwiseXorScalarOut = l0op::NotEqual(contiguousSelf, otherTensor, uniqueExecutor.get());
  } else {
    // 将输入self的数据类型转换成隐式数据类型
    auto castSelf = l0op::Cast(contiguousSelf, promoteType, uniqueExecutor.get());
    CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bitwiseXorScalarOut = l0op::BitwiseXor(castSelf, otherTensor, uniqueExecutor.get());
  }
  CHECK_RET(bitwiseXorScalarOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(bitwiseXorScalarOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnBitwiseXorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                                  uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnBitwiseXorScalar, DFX_IN(self, other), DFX_OUT(out));
  return GetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnBitwiseXorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnBitwiseXorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceBitwiseXorScalarGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceBitwiseXorScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return GetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceBitwiseXorScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceBitwiseXorScalar);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
