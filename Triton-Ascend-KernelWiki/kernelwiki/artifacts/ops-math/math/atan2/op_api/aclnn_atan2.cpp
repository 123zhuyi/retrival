/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_atan2.h"
#include "atan2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// API支持的所有数据类型列表
static const std::initializer_list<DataType> ASCEND910_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8,  DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8};

static const std::initializer_list<DataType> ASCEND910B_INPUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_INT8,  DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,   DataType::DT_BOOL,   DataType::DT_UINT8, DataType::DT_BF16};

// 当前算子（包括aicore,aicpu）支持数据类型列表
static const std::initializer_list<DataType> ASCEND910_OPERATOR_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> ASCEND910B_OPERATOR_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_DOUBLE, DataType::DT_BF16};

static inline const std::initializer_list<DataType>& GetDtypeSupportList(bool isInput) {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return isInput ? ASCEND910B_INPUT_DTYPE_SUPPORT_LIST : ASCEND910B_OPERATOR_SUPPORT_LIST;
  } else {
    return isInput ? ASCEND910_INPUT_DTYPE_SUPPORT_LIST : ASCEND910_OPERATOR_SUPPORT_LIST;
  }
}

// 检查输入和输出的数据类型是否在算子的支持列表内
static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* other) {
  auto supportList = GetDtypeSupportList(true);
  // 检查输入的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
  return true;
}

// 检查输入和输出是否是空指针
static inline bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
  // 检查输入是否是空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);

  // 检查输入是否是空指针
  OP_CHECK_NULL(out, return false);

  return true;
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
  // self的维度必须小于等于8
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  // other的维度必须小于等于8
  OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
  return true;
}

static inline aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入的数据shape是否符合条件
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

// 类型推导
static inline DataType InferDtype(const DataType selfDtype, const DataType otherDtype) {
  DataType promoteType = PromoteType(selfDtype, otherDtype);
  auto supportList = GetDtypeSupportList(false);
  if (!CheckType(promoteType, supportList)) {
    OP_LOGD("SelfDtype dtype %s ,otherDtype dtype %s ,promoteType %s,inferDtype type float.",
            ToString(selfDtype).GetString(), ToString(otherDtype).GetString(), ToString(promoteType).GetString());
    return DataType::DT_FLOAT;
  }
  return promoteType;
}

static aclnnStatus ExecAtan2GetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                             uint64_t* workspaceSize, aclOpExecutor** executor) {
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (self->IsEmpty() || other->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  if (self->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGW("Format only support ND");
  }
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto targetDtype = InferDtype(self->GetDataType(), other->GetDataType());

  auto selfCast = l0op::Cast(selfContiguous, targetDtype, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto otherCast = l0op::Cast(otherContiguous, targetDtype, uniqueExecutor.get());
  CHECK_RET(otherCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto kernelOut = l0op::Atan2(selfCast, otherCast, uniqueExecutor.get());
  CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto castOut = l0op::Cast(kernelOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  // 需要把 uniqueExecutor持有executor转移给executor
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAtan2GetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnAtan2, DFX_IN(self, other), DFX_OUT(out));
  return ExecAtan2GetWorkspaceSize(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceAtan2GetWorkspaceSize(aclTensor* selfRef, aclTensor* other, uint64_t* workspaceSize,
                                              aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplaceAtan2, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  auto out = const_cast<aclTensor*>(selfRef);
  return ExecAtan2GetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnAtan2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnAtan2);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceAtan2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  // 固定写法，调用框架能力，完成计算
  L2_DFX_PHASE_2(aclnnInplaceAtan2);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
